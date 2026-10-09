r"""ΔContext Policy — the payoff of the measurement and control infrastructure.

Version 1 is a **checkpoint-anchored selector**:
- ``observe()`` runs the semantic pipeline (embed -> metric -> accumulate -> decide)
  as each turn arrives. It emits ``SNAPSHOT`` at semantic boundaries, leaving a trail
  of checkpoint anchor turns.
- ``build_context()`` divides the budget. It spends up to ``landmark_fraction`` of
  the budget keeping checkpoint turns (landmarks of past topics), and the remaining
  budget keeping the most recent contiguous tail.

When zero checkpoints fire, it behaves identically to a sliding window. Because it
retains checkpoints rather than summarising them, no LLM summarizer is needed on
the critical path.

Landmark selection is **query-aware** when a query is given (Version 1.1): each
checkpoint already carries the semantic-state embedding it fired on, so ranking
landmarks by relevance to the current question costs one query embedding plus
O(checkpoints) cosine distances. This closes a real structural gap: earlier
versions ignored ``query`` entirely and returned the same context for every
question in a conversation, while :class:`~contextbench.policies.retrieval.RetrievalPolicy`
re-searches per question. Falls back to pure recency (the original behaviour)
when there's no query or no checkpoints.

Version 1.2 (``expand_landmark_search``, on by default) goes further: rather than
ranking only the sparse checkpoint pool, it ranks *every observed turn*. This is
not the O(full history) cost it sounds like -- ``observe()`` already embeds every
turn to build the windowed-centroid state, so caching that per-turn embedding and
reusing it at query time costs nothing extra beyond the ranking's own cosine
distances (cheap numpy ops), matching RetrievalPolicy's recall exactly while
retaining checkpoints for what they're for: the context-rot diagnostics and, when
``expand_landmark_search`` is off, the cheaper checkpoint-only search mode.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from contextbench.context.accumulator import ACCUMULATOR_REGISTRY
from contextbench.context.checkpoint_manager import CHECKPOINT_REGISTRY, Checkpoint
from contextbench.context.config import DeltaContextConfig
from contextbench.context.controller import CONTROLLER_REGISTRY, ControlAction, ControlSignals
from contextbench.context.semantic_metric import METRIC_REGISTRY, SemanticChangeMetric
import contextbench.context.embedding_metric  # Register cosine metric
from contextbench.context.state import SemanticState, SentenceTransformerEmbedder, WindowedCentroidConstructor
from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import ContextResult, Message, TokenCounter
from contextbench.policies.base import ContextPolicy, register_policy

logger = logging.getLogger(__name__)


def rank_checkpoints_by_relevance(
    checkpoints: Sequence[Checkpoint], query_state: SemanticState, metric: SemanticChangeMetric
) -> list[Checkpoint]:
    """Sort checkpoints by semantic closeness to ``query_state``, closest first.

    Pure and dependency-free of the policy/embedder so it's unit-testable without
    a real SentenceTransformer: callers just need any :class:`SemanticState` for
    the query (in production, the embedded question text) and the same metric
    ΔContext already uses for turn-to-turn distance.
    """
    return sorted(checkpoints, key=lambda c: metric.distance(query_state, c.state))


def rank_turn_indices_by_relevance(
    turn_embeddings: dict[int, np.ndarray], query_state: SemanticState, metric: SemanticChangeMetric
) -> list[int]:
    """Sort observed turn indices by semantic closeness to ``query_state``, closest first.

    Ranks on each turn's own single-turn embedding (not the windowed-centroid
    state ΔContext uses for change detection), so a turn's relevance reflects its
    own content rather than a blend with its neighbours. Pure and unit-testable
    without a real embedder, same as :func:`rank_checkpoints_by_relevance`.
    """
    def _distance(turn_index: int) -> float:
        candidate = SemanticState(
            embedding=turn_embeddings[turn_index], token_count=0,
            turn_index=turn_index, timestamp=float(turn_index),
        )
        return metric.distance(query_state, candidate)

    return sorted(turn_embeddings.keys(), key=_distance)


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
        elif self._cfg.accumulator == "ema":
            self._accumulator = ACCUMULATOR_REGISTRY.create(
                self._cfg.accumulator, alpha=self._cfg.ema_alpha
            )
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

        # Context-rot diagnostic: a never-reset EMA of instantaneous delta,
        # tracked independently of whichever accumulator drives checkpoint
        # triggering (see accumulator.py's module docstring for why these are
        # kept separate). A persistently near-zero value, or a long streak of
        # turns below `ignore_below`, means the conversation isn't
        # semantically progressing turn to turn.
        self._rot_ema = 0.0
        self._stagnant_streak = 0
        self._max_stagnant_streak = 0

        # Per-turn embedding cache for query-aware landmark search. Populated in
        # observe() as a side effect of embedding for the windowed-centroid state
        # (see rank_turn_indices_by_relevance's docstring) -- no extra embedding
        # cost, just retaining what would otherwise be discarded after each
        # update() call folds it into the centroid.
        self._turn_embeddings: dict[int, np.ndarray] = {}

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
        self._rot_ema = 0.0
        self._stagnant_streak = 0
        self._max_stagnant_streak = 0
        self._turn_embeddings.clear()
        self._count_cache.clear()

    def observe(self, message: Message) -> None:
        """Ingest one history turn, update semantic tracking, and drop checkpoints."""
        text = f"{message.speaker or message.role}: {message.content}"
        token_count = self._count_message(message)
        timestamp = float(self._turn_index)

        raw_embedding = np.asarray(self._state_constructor.embedder.embed(text), dtype=np.float32)
        self._turn_embeddings[self._turn_index] = raw_embedding
        current_state = self._state_constructor.update(
            text, turn_index=self._turn_index, token_count=token_count, timestamp=timestamp,
            embedding=raw_embedding,
        )

        if self._last_state is None:
            delta = 0.0
            acc = self._accumulator.update(0.0, current_state)
        else:
            delta = self._metric.distance(self._last_state, current_state)
            acc = self._accumulator.update(delta, current_state)

        # Context-rot bookkeeping — independent of the checkpoint trigger above.
        self._rot_ema = (
            self._cfg.rot_ema_alpha * delta + (1.0 - self._cfg.rot_ema_alpha) * self._rot_ema
        )
        self._stagnant_streak = self._stagnant_streak + 1 if delta < self._cfg.ignore_below else 0
        if self._stagnant_streak > self._max_stagnant_streak:
            self._max_stagnant_streak = self._stagnant_streak
        if self._stagnant_streak == self._cfg.rot_streak_threshold:
            logger.info(
                "context_rot detected at turn %d: %d consecutive turns with delta < %.3f "
                "(rot_ema=%.4f)",
                self._turn_index, self._stagnant_streak, self._cfg.ignore_below, self._rot_ema,
            )

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

        # 1. Rank landmark candidate turn indices, best-first:
        #    - query given + query-aware landmarks enabled + full-history search
        #      on (default): rank EVERY observed turn by relevance, using
        #      embeddings already cached in observe() -- matches
        #      RetrievalPolicy's recall at no extra embedding cost.
        #    - query given + query-aware + full-history off: rank only the
        #      (much smaller) checkpoint pool by relevance -- the cheaper V1.1
        #      behaviour, kept for the efficiency-focused ablation.
        #    - no usable query: fall back to recency over checkpoints, as V1.
        if query and self._cfg.query_aware_landmarks and self._turn_embeddings:
            query_embedding = self._state_constructor.embedder.embed(query)
            query_state = SemanticState(
                embedding=np.asarray(query_embedding, dtype=np.float32),
                token_count=0, turn_index=-1, timestamp=-1.0,
            )
            if self._cfg.expand_landmark_search:
                candidate_turn_indices = rank_turn_indices_by_relevance(
                    self._turn_embeddings, query_state, self._metric
                )
            else:
                candidate_turn_indices = [
                    c.turn_index
                    for c in rank_checkpoints_by_relevance(all_ckpts, query_state, self._metric)
                ]
            query_aware = True
        else:
            candidate_turn_indices = [c.turn_index for c in reversed(all_ckpts)]
            query_aware = False

        # 2. Greedily fill the landmark budget in ranked order. Skipping (not
        # breaking on) an over-budget candidate matters here: unlike pure
        # recency order, relevance order has no reason to have cost increase
        # down the list, so stopping at the first miss would waste budget.
        selected_landmarks: list[Message] = []
        landmark_tokens = 0
        landmark_turn_indices = set()

        for turn_idx in candidate_turn_indices:
            if turn_idx >= len(history):
                continue  # robustness
            msg = history[turn_idx]
            cost = self._count_message(msg)
            if landmark_tokens + cost <= landmark_budget:
                selected_landmarks.append(msg)
                landmark_tokens += cost
                landmark_turn_indices.add(turn_idx)

        # Landmarks are picked by relevance/recency but must still render in
        # chronological order like every other selected message.
        selected_landmarks.sort(key=lambda m: m.turn_id)

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
            # Context-rot diagnostics (see observe()): a persistently near-zero
            # rot_ema, or a long stagnant_streak, means the conversation isn't
            # semantically progressing turn to turn.
            "rot_ema": self._rot_ema,
            "stagnant_streak": self._stagnant_streak,
            "max_stagnant_streak": self._max_stagnant_streak,
            "context_rot_detected": self._max_stagnant_streak >= self._cfg.rot_streak_threshold,
            "query_aware_landmarks_used": query_aware,
            "landmark_search_scope": (
                "full_history" if query_aware and self._cfg.expand_landmark_search
                else "checkpoints"
            ),
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
