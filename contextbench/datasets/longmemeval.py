"""LongMemEval loader.

LongMemEval (Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term
Interactive Memory", ICLR 2025) gives each of its 500 questions its own chat
history: a "haystack" of user-assistant sessions on mostly unrelated topics,
with very different lengths, a few of which hold the answer. That makes it the
counterpart to LoCoMo for this benchmark: LoCoMo's sessions are one continuing
conversation between two people with similar-sized sessions, while
LongMemEval's topic segments are irregular, which is where detecting topic
boundaries from content could beat cutting the history into fixed-size chunks.

Normalization performed here
----------------------------
* **One sample per question**: the haystack sessions, in the given
  (chronological) order, become one turn stream; ``sample_id`` is the
  ``question_id``.
* **Turns** keep their ``user`` / ``assistant`` roles and are labelled
  ``User`` / ``Assistant``. Each carries ``meta["dia_id"] = "<session_id>:<n>"``
  and its session's date in ``meta["session_date"]``.
* **Evidence** is the turn-level ``has_answer`` labels, as ``dia_id``s, so
  evidence recall works exactly as on LoCoMo. Questions without turn-level
  labels get no evidence and are left out of evidence recall.
* **Abstention questions** (``question_id`` ending in ``_abs``) ask about
  something the history never says; they are flagged adversarial, so the
  ``locomo_qa`` evaluator scores them on abstention.
* **Categories** are integers (``QAItem.category`` is an int) mapped from
  ``question_type`` by :data:`QUESTION_TYPES`; the type name is kept in
  ``QAItem.meta``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from contextbench.core.config_schema import DatasetConfig
from contextbench.core.types import Message, QAItem
from contextbench.datasets.base import ConversationSample, Dataset, register_dataset

QUESTION_TYPES: dict[str, int] = {
    "single-session-user": 1,
    "single-session-assistant": 2,
    "single-session-preference": 3,
    "multi-session": 4,
    "temporal-reasoning": 5,
    "knowledge-update": 6,
}
"""``question_type`` -> integer category used in reports."""

_SPEAKERS = {"user": "User", "assistant": "Assistant"}


@register_dataset("longmemeval")
class LongMemEvalDataset(Dataset):
    """Loader for a LongMemEval JSON file (``_s``, ``_m`` or ``_oracle``)."""

    def __init__(self, config: DatasetConfig) -> None:
        """Load and parse the LongMemEval JSON.

        Raises:
            FileNotFoundError: If ``config.path`` does not exist.
        """
        super().__init__(config)
        path = Path(config.path)
        if not path.is_file():
            raise FileNotFoundError(
                f"LongMemEval file not found at '{path}'. Download it from the "
                "HuggingFace dataset xiaowu0162/longmemeval-cleaned."
            )
        with path.open(encoding="utf-8") as handle:
            raw: list[dict[str, Any]] = json.load(handle)

        if config.qa_categories is not None:
            wanted = set(config.qa_categories)
            raw = [r for r in raw if QUESTION_TYPES.get(r["question_type"]) in wanted]
        if config.num_samples is not None:
            raw = raw[: config.num_samples]

        self._samples: tuple[ConversationSample, ...] = tuple(self._build_sample(r) for r in raw)

    def __iter__(self) -> Iterator[ConversationSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def _build_sample(raw: dict[str, Any]) -> ConversationSample:
        question_id = str(raw["question_id"])
        messages: list[Message] = []
        evidence: list[str] = []
        for session_id, date, session in zip(
            raw["haystack_session_ids"], raw["haystack_dates"], raw["haystack_sessions"]
        ):
            for n, turn in enumerate(session):
                dia_id = f"{session_id}:{n}"
                role = "assistant" if turn["role"] == "assistant" else "user"
                messages.append(Message(
                    role=role,
                    content=str(turn["content"]),
                    turn_id=len(messages),
                    speaker=_SPEAKERS[role],
                    session_id=str(session_id),
                    meta={"dia_id": dia_id, "session_date": date},
                ))
                if turn.get("has_answer"):
                    evidence.append(dia_id)

        question_type = str(raw["question_type"])
        abstention = question_id.endswith("_abs")
        qa = QAItem(
            question=str(raw["question"]),
            answer=str(raw["answer"]),
            evidence=tuple(evidence),
            category=QUESTION_TYPES.get(question_type),
            qa_id=question_id,
            meta={
                "adversarial": abstention,
                "question_type": question_type,
                "question_date": raw.get("question_date"),
                "answer_session_ids": tuple(raw.get("answer_session_ids") or ()),
            },
        )
        return ConversationSample(
            sample_id=question_id,
            messages=tuple(messages),
            qa_items=(qa,),
            speakers=("User", "Assistant"),
            meta={"num_sessions": len(raw["haystack_sessions"]),
                  "question_type": question_type},
        )
