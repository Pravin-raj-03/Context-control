"""Sliding-window context policy — the Phase 1 baseline.

The simplest possible policy and the canonical *stateless* one: keep the newest
whole messages that fit the token budget, drop the rest. Because the runner owns
the history and passes it into :meth:`build_context`, this policy needs no
internal history and no :meth:`observe` hook — it reads the history it is handed.

Budget guarantee
----------------
Greedy filling uses the cheap, additive :meth:`TokenCounter.count_message`, which
slightly *under*-counts a rendered block (inter-line separators add tokens). To
keep the hard invariant ``token_count <= budget``, the greedily chosen set is then
measured exactly with :meth:`TokenCounter.count_messages` and the oldest messages
are trimmed until the exact block fits. Messages are never truncated — a message
is atomic — so if even the newest single message exceeds the budget the returned
context is empty.
"""

from __future__ import annotations

from typing import Sequence

from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import ContextResult, Message, TokenCounter
from contextbench.policies.base import ContextPolicy, register_policy


@register_policy("sliding_window")
class SlidingWindowPolicy(ContextPolicy):
    """Keep the most recent messages that fit within the token budget."""

    def __init__(self, config: PolicyConfig, token_counter: TokenCounter) -> None:
        """Initialize the policy and its per-message count cache."""
        super().__init__(config, token_counter)
        # Cache per-message token counts: the same frozen Message objects are
        # scored repeatedly across the many QA of one conversation.
        self._count_cache: dict[Message, int] = {}

    def reset(self) -> None:
        """Clear the count cache before a new conversation."""
        self._count_cache.clear()

    def build_context(
        self, history: Sequence[Message], query: str | None, budget: int
    ) -> ContextResult:
        """Select the newest whole messages that fit ``budget`` tokens.

        Args:
            history: The full conversation so far, chronological. Not mutated.
            query: Ignored — the sliding window is query-agnostic.
            budget: Maximum context tokens.

        Returns:
            A :class:`ContextResult` whose ``token_count`` is ``<= budget`` and
            whose ``messages`` are the most recent contiguous, chronologically
            ordered subset of ``history``.
        """
        if budget <= 0 or not history:
            return self._empty_result(len(history), budget)

        # Greedy newest -> oldest fill using the additive per-message estimate.
        selected_reversed: list[Message] = []
        running = 0
        for message in reversed(history):
            cost = self._count_message(message)
            if running + cost > budget:
                break
            selected_reversed.append(message)
            running += cost
        selected = list(reversed(selected_reversed))

        # Exact-fit guarantee: the true block cost may exceed the additive sum
        # (separators); drop from the oldest end until the block fits.
        token_count = self._token_counter.count_messages(selected)
        while selected and token_count > budget:
            selected.pop(0)
            token_count = self._token_counter.count_messages(selected)

        dropped = len(history) - len(selected)
        meta = {
            "budget": budget,
            "selected_messages": len(selected),
            "dropped_messages": dropped,
            "oldest_turn_id": selected[0].turn_id if selected else None,
            "newest_turn_id": selected[-1].turn_id if selected else None,
        }
        return ContextResult(
            messages=tuple(selected),
            token_count=token_count,
            dropped_messages=dropped,
            policy_meta=meta,
        )

    def _empty_result(self, history_len: int, budget: int) -> ContextResult:
        return ContextResult(
            messages=(),
            token_count=0,
            dropped_messages=history_len,
            policy_meta={
                "budget": budget,
                "selected_messages": 0,
                "dropped_messages": history_len,
                "oldest_turn_id": None,
                "newest_turn_id": None,
            },
        )

    def _count_message(self, message: Message) -> int:
        cached = self._count_cache.get(message)
        if cached is None:
            cached = self._token_counter.count_message(message)
            self._count_cache[message] = cached
        return cached
