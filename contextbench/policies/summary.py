"""Summary context policy — a baseline that condenses dropped history.

This baseline simulates a summarization memory. True LLM-based summarization 
of dropped turns during a benchmark sweep would require tens of thousands of 
generations, making it computationally prohibitive.

Instead, this policy uses a fast, heuristic extractive summary (taking the first 
~60 characters of each dropped turn). This provides the model with condensed 
pointers to past topics without exceeding the benchmark's time constraints.

It spends a fraction of the budget on the summary and the rest on the recent tail.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field

from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import ContextResult, Message, TokenCounter
from contextbench.policies.base import ContextPolicy, register_policy


class SummaryConfig(BaseModel):
    summary_fraction: float = Field(default=0.3, ge=0.0, le=1.0)


@register_policy("summary")
class SummaryPolicy(ContextPolicy):
    """Maintains a heuristic summary of dropped turns."""

    def __init__(self, config: PolicyConfig, token_counter: TokenCounter) -> None:
        super().__init__(config, token_counter)
        self._cfg = SummaryConfig(**self._config.model_extra)
        self._count_cache: dict[Message, int] = {}

    def reset(self) -> None:
        self._count_cache.clear()

    def build_context(
        self, history: Sequence[Message], query: str | None, budget: int
    ) -> ContextResult:
        if budget <= 0 or not history:
            return ContextResult(messages=(), token_count=0, dropped_messages=len(history))

        # We first figure out how much of the tail we can fit if we reserve the summary budget
        summary_budget = int(budget * self._cfg.summary_fraction)
        tail_budget = budget - summary_budget

        selected_tail: list[Message] = []
        tail_tokens = 0
        tail_start_idx = len(history)

        for turn_idx in range(len(history) - 1, -1, -1):
            msg = history[turn_idx]
            cost = self._count_message(msg)
            if tail_tokens + cost <= tail_budget:
                selected_tail.append(msg)
                tail_tokens += cost
                tail_start_idx = turn_idx
            else:
                break

        selected_tail.reverse()

        dropped_messages = history[:tail_start_idx]

        # Build the summary message, budget-capped from the start so it can
        # never blow past summary_budget and force the trim loop below to
        # delete it wholesale (the bug this replaces: a summary that didn't
        # fit was previously dropped entirely, silently sending the model an
        # EMPTY context on every turn where anything had to be summarized).
        summary_message = (
            self._build_summary_message(dropped_messages, summary_budget)
            if dropped_messages and summary_budget > 0
            else None
        )

        combined = []
        if summary_message is not None:
            combined.append(summary_message)
        combined.extend(selected_tail)

        # Exact-fit guarantee (safety net only — the summary is already
        # budget-capped above, so in practice this trims tail overflow from
        # tokenization-boundary rounding, not the summary itself).
        exact_tokens = self._token_counter.count_messages(combined)

        while combined and exact_tokens > budget:
            # Prefer dropping from the tail to preserve the summary; only
            # fall back to dropping the summary if it's the sole survivor.
            if len(combined) > 1:
                combined.pop(1)  # pop oldest from tail
            else:
                combined.pop(0)  # last resort: pop the summary itself
            exact_tokens = self._token_counter.count_messages(combined)

        dropped_count = len(history) - (len(combined) - (1 if summary_message in combined else 0))

        meta = {
            "tail_selected": len(selected_tail),
            "dropped_summarized": len(dropped_messages),
            "summary_budget": summary_budget,
            "tail_budget": tail_budget,
        }

        return ContextResult(
            messages=tuple(combined),
            token_count=exact_tokens,
            dropped_messages=dropped_count,
            policy_meta=meta,
        )

    def _count_message(self, message: Message) -> int:
        cached = self._count_cache.get(message)
        if cached is None:
            cached = self._token_counter.count_message(message)
            self._count_cache[message] = cached
        return cached

    def _build_summary_message(
        self, dropped_messages: Sequence[Message], summary_budget: int
    ) -> Message | None:
        """Build a heuristic extractive summary that fits within ``summary_budget``.

        Stride-samples the dropped turns across the *whole* dropped span, not
        just its earliest slice, so the summary gives condensed pointers to
        topics from across the conversation rather than exhausting its budget
        describing only the opening turns. Accumulates bullets against a
        running token estimate and stops before exceeding the budget, so the
        result is guaranteed to fit and is never deleted downstream by the
        exact-fit trim (the bug this replaces).
        """
        header = "Summary of earlier conversation:"
        running_tokens = self._token_counter.count_message(
            Message(role="system", content=header, turn_id=-1, speaker="System")
        )
        if running_tokens > summary_budget:
            return None  # even the header alone doesn't fit; skip the summary

        # Estimate how many bullets plausibly fit, then stride-sample the
        # dropped span evenly so early *and* late dropped topics both get a
        # shot at inclusion.
        avg_line_tokens = max(1, self._token_counter.count_text(self._summary_line(dropped_messages[0])))
        budget_left = summary_budget - running_tokens
        max_bullets = max(1, budget_left // avg_line_tokens)
        stride = max(1, len(dropped_messages) // max_bullets)

        lines: list[str] = []
        for m in dropped_messages[::stride]:
            line = self._summary_line(m)
            line_tokens = self._token_counter.count_text(line)
            if running_tokens + line_tokens > summary_budget:
                break
            lines.append(line)
            running_tokens += line_tokens

        if not lines:
            return None

        text = header + "\n" + "\n".join(lines)
        return Message(role="system", content=text, turn_id=-1, speaker="System")

    @staticmethod
    def _summary_line(m: Message) -> str:
        prefix = m.speaker or m.role
        content = m.content.strip().replace("\n", " ")
        if len(content) > 60:
            content = content[:57] + "..."
        return f"- {prefix}: {content}"
