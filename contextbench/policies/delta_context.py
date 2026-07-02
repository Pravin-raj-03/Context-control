r"""ΔContext Policy — the payoff of the measurement and control infrastructure.

Version 1 is a **checkpoint-anchored selector**:
- ``observe()`` runs the semantic pipeline (embed -> metric -> accumulate -> decide)
  as each turn arrives. It emits ``SNAPSHOT`` at semantic boundaries, leaving a trail
  of checkpoint anchor turns.
- ``build_context()`` divides the budget. It spends up to ``landmark_fraction`` of
  the budget keeping the most recent checkpoint turns (landmarks of past topics), and
  the remaining budget keeping the most recent contiguous tail.

When zero checkpoints fire, it behaves identically to a sliding window. Because it
retains checkpoints rather than summarising them, no LLM summarizer is needed on
the critical path.
"""

from __future__ import annotations

from typing import Sequence

from contextbench.context.accumulator import ACCUMULATOR_REGISTRY
from contextbench.context.checkpoint_manager import CHECKPOINT_REGISTRY, Checkpoint
from contextbench.context.config import DeltaContextConfig
from contextbench.context.controller import CONTROLLER_REGISTRY, ControlAction, ControlSignals
from contextbench.context.semantic_metric import METRIC_REGISTRY
from contextbench.context.state import SemanticState, SentenceTransformerEmbedder, WindowedCentroidConstructor
from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import ContextResult, Message, TokenCounter
from contextbench.policies.base import ContextPolicy, register_policy


@register_policy("delta_context")
class DeltaContextPolicy(ContextPolicy):
    """The ΔContext policy (Version 1)."""

    def __init__(self, config: PolicyConfig, token_counter: TokenCounter) -> None:
        super().__init__(config, token_counter)
        self._cfg = DeltaContextConfig(**self._config.model_extra)

        # 1. State Constructor
        embedder = SentenceTransformerEmbedder(
            model_name=self._cfg.embedder_model, device=self._cfg.embedder_device
        )
        self._state_constructor = WindowedCentroidConstructor(embedder, window=self._cfg.window)

        # 2-3. Metric
        self._metric = METRIC_REGISTRY.create(self._cfg.metric)

        # 4. Accumulator
        if self._cfg.accumulator == "displacement":
            self._accumulator = ACCUMULATOR_REGISTRY.create(self._cfg.accumulator, self._metric)
        else:
            self._accumulator = ACCUMULATOR_REGISTRY.create(self._cfg.accumulator)

        # 5. Controller
        self._controller = CONTROLLER_REGISTRY.create(
            self._cfg.controller,
            change_threshold=self._cfg.change_threshold,
            ignore_below=self._cfg.ignore_below,
            budget_pressure_threshold=self._cfg.budget_pressure_threshold,
            min_snapshot_gap=self._cfg.min_snapshot_gap,
        )

        # 6. Checkpoint Manager
        self._checkpoints = CHECKPOINT_REGISTRY.create(self._cfg.checkpoint_manager)

        # Incremental internal state
        self._turn_index = 0
        self._last_state: SemanticState | None = None
        self._last_checkpoint_turn = -1

        # Cache for counting message tokens exactly like sliding window does
        self._count_cache: dict[Message, int] = {}

    def reset(self) -> None:
        """Clear all conversation state."""
        self._state_constructor.reset()
        self._accumulator.reset()
        self._checkpoints.reset()
        self._turn_index = 0
        self._last_state = None
        self._last_checkpoint_turn = -1
        self._count_cache.clear()

    def observe(self, message: Message) -> None:
        """Ingest one history turn, update semantic tracking, and drop checkpoints."""
        text = f"{message.speaker or message.role}: {message.content}"
        token_count = self._count_message(message)
        timestamp = float(self._turn_index)

        current_state = self._state_constructor.update(
            text, turn_index=self._turn_index, token_count=token_count, timestamp=timestamp
        )

        if self._last_state is None:
            delta = 0.0
            acc = self._accumulator.update(0.0, current_state)
        else:
            delta = self._metric.distance(self._last_state, current_state)
            acc = self._accumulator.update(delta, current_state)

        # In observe(), we are deciding whether to drop a snapshot anchor.
        # We pass a massive budget so budget_pressure is ~0, meaning we don't trigger COMPRESS here.
        signals = ControlSignals(
            accumulated_change=acc,
            instantaneous_delta=delta,
            context_tokens=0,
            budget=1_000_000,
            turns_since_checkpoint=self._turn_index - self._last_checkpoint_turn,
            state=current_state,
        )

        decision = self._controller.decide(signals)

        if decision.action == ControlAction.SNAPSHOT:
            checkpoint = Checkpoint(
                checkpoint_id=self._checkpoints.next_id(),
                state=current_state,
                turn_index=self._turn_index,
                token_count=current_state.token_count,
                timestamp=timestamp,
                accumulated_change=acc,
                turn_span=(self._last_checkpoint_turn + 1, self._turn_index),
                meta={"decision_reason": decision.reason},
            )
            self._checkpoints.add(checkpoint)
            self._accumulator.reset(checkpoint_state=current_state)
            self._last_checkpoint_turn = self._turn_index

        self._last_state = current_state
        self._turn_index += 1

    def build_context(
        self, history: Sequence[Message], query: str | None, budget: int
    ) -> ContextResult:
        """Select checkpoint anchors + recent tail within budget."""
        if budget <= 0 or not history:
            return ContextResult(messages=(), token_count=0, dropped_messages=len(history))

        landmark_budget = int(budget * self._cfg.landmark_fraction)
        all_ckpts = self._checkpoints.all()

        # 1. Collect landmarks from most recent backwards
        selected_landmarks: list[Message] = []
        landmark_tokens = 0
        landmark_turn_indices = set()

        for ckpt in reversed(all_ckpts):
            turn_idx = ckpt.turn_index
            if turn_idx >= len(history):
                continue  # robustness
            msg = history[turn_idx]
            cost = self._count_message(msg)
            if landmark_tokens + cost <= landmark_budget:
                selected_landmarks.append(msg)
                landmark_tokens += cost
                landmark_turn_indices.add(turn_idx)
            else:
                break

        # 2. Collect recent tail from the remaining budget
        tail_budget = budget - landmark_tokens
        selected_tail: list[Message] = []
        tail_tokens = 0

        for turn_idx in range(len(history) - 1, -1, -1):
            if turn_idx in landmark_turn_indices:
                continue
            msg = history[turn_idx]
            cost = self._count_message(msg)
            if tail_tokens + cost <= tail_budget:
                selected_tail.append(msg)
                tail_tokens += cost
            else:
                break

        # 3. Combine and sort
        combined = selected_landmarks + selected_tail
        combined.sort(key=lambda m: m.turn_id)

        # 4. Exact-fit guarantee (trim oldest from tail if needed)
        exact_tokens = self._token_counter.count_messages(combined)
        while combined and exact_tokens > budget:
            # We must drop something. We should ideally drop from the tail end, but they are sorted chronologically.
            # Usually we pop the oldest message to keep recency. 
            # If the oldest message is a landmark, we might accidentally drop it, but budget strictness wins.
            combined.pop(0)
            exact_tokens = self._token_counter.count_messages(combined)

        dropped = len(history) - len(combined)

        meta = {
            "landmarks_selected": len(selected_landmarks),
            "tail_selected": len(selected_tail),
            "checkpoints_total": len(all_ckpts),
            "landmark_tokens": landmark_tokens,
            "tail_tokens": tail_tokens,
            "landmark_budget": landmark_budget,
            "tail_budget": tail_budget,
        }

        return ContextResult(
            messages=tuple(combined),
            token_count=exact_tokens,
            dropped_messages=dropped,
            policy_meta=meta,
        )

    def _count_message(self, message: Message) -> int:
        cached = self._count_cache.get(message)
        if cached is None:
            cached = self._token_counter.count_message(message)
            self._count_cache[message] = cached
        return cached
