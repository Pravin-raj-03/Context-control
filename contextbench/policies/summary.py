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
        
        # Build the summary message
        if not dropped_messages:
            summary_message = None
        else:
            lines = []
            for m in dropped_messages:
                prefix = m.speaker or m.role
                content = m.content.strip().replace('\n', ' ')
                if len(content) > 60:
                    content = content[:57] + "..."
                lines.append(f"- {prefix}: {content}")
                
            summary_text = "Summary of earlier conversation:\n" + "\n".join(lines)
            summary_message = Message(
                role="system",
                content=summary_text,
                turn_id=-1,
                speaker="System"
            )

        combined = []
        if summary_message:
            combined.append(summary_message)
        combined.extend(selected_tail)

        # Exact-fit guarantee
        exact_tokens = self._token_counter.count_messages(combined)
        
        while combined and exact_tokens > budget:
            # If we need to drop something, we drop from the tail to preserve the summary.
            # But if the tail is empty, we must drop the summary.
            if len(combined) > 1:
                combined.pop(1) # pop oldest from tail
            else:
                combined.pop(0) # pop summary
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
