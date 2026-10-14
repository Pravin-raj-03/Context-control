"""Precomputed context policy — replays turn selections made offline.

``scripts/memory_lab.py --export`` writes, for each retrieval method, the turn
indices it selects for every question. This policy returns exactly those turns,
so methods explored offline (including ones that need GPU-side preprocessing
such as LLMLingua-2 compression) are answered through the identical prompt
builder, reader and scorer as every other policy.

Selections are keyed by ``"<question>\\x1f<history length>"``: the question text
alone can repeat across LongMemEval histories, the pair does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import ContextResult, Message, TokenCounter
from contextbench.policies.base import ContextPolicy, register_policy

KEY_SEP = "\x1f"


def selection_key(question: str, history_length: int) -> str:
    return f"{question}{KEY_SEP}{history_length}"


class PrecomputedConfig(BaseModel):
    selection_file: str
    method: str


@register_policy("precomputed")
class PrecomputedPolicy(ContextPolicy):
    """Returns the turns a precomputed selection lists for this question."""

    def __init__(self, config: PolicyConfig, token_counter: TokenCounter) -> None:
        super().__init__(config, token_counter)
        self._cfg = PrecomputedConfig(**self._config.model_extra)
        table = json.loads(Path(self._cfg.selection_file).read_text(encoding="utf-8"))
        if self._cfg.method not in table:
            raise KeyError(f"method {self._cfg.method!r} not in {self._cfg.selection_file}")
        self._selection: dict[str, list[int]] = table[self._cfg.method]

    def reset(self) -> None:
        """Stateless: the selection table does not change between conversations."""

    def build_context(
        self, history: Sequence[Message], query: str | None, budget: int
    ) -> ContextResult:
        ids = self._selection.get(selection_key(query or "", len(history)))
        if ids is None:
            raise KeyError(f"no precomputed selection for question {query!r} ({len(history)} turns)")
        chosen = [history[i] for i in sorted(set(ids)) if 0 <= i < len(history)]
        tokens = self._token_counter.count_messages(chosen)
        dropped_for_fit = 0
        while chosen and tokens > budget:  # exact-fit guarantee, as in the retrieval policy
            chosen.pop(0)
            dropped_for_fit += 1
            tokens = self._token_counter.count_messages(chosen)
        return ContextResult(
            messages=tuple(chosen),
            token_count=tokens,
            dropped_messages=len(history) - len(chosen),
            policy_meta={"precomputed_method": self._cfg.method, "selected": len(chosen),
                         "dropped_for_fit": dropped_for_fit},
        )
