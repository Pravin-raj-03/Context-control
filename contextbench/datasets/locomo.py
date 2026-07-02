"""LoCoMo dataset loader.

LoCoMo (Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM
Agents") is a collection of very long, multi-session dialogues between two
speakers, each accompanied by QA probes across several reasoning categories.

Normalization performed here
----------------------------
* **Sessions** are concatenated in numeric order (``session_1`` ... ``session_N``,
  sorted by integer, not lexically) into one chronological turn stream, with each
  turn's ``session_id`` and session date preserved on the message.
* **Roles**: the two speakers are mapped ``speaker_a -> "user"`` and
  ``speaker_b -> "assistant"``. This mapping is *nominal* — the QA task answers
  questions *about* the transcript, so the prompt builder renders turns by
  speaker label, not by chat role. The role is retained for generality and
  faithful reconstruction.
* **Image turns**: turns carrying a BLIP caption are, by default, rendered as
  ``"<text> [shared a photo: <caption>]"`` so a text-only model retains the
  visual context. Raw ``img_url`` / ``blip_caption`` / ``query`` are also kept in
  message metadata. Controlled by ``dataset.include_image_captions``.
* **Answers**: adversarial questions (category 5) store their gold in
  ``adversarial_answer`` rather than ``answer``; the loader reads whichever is
  present and coerces integer answers to strings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from contextbench.core.config_schema import DatasetConfig
from contextbench.core.types import Message, QAItem, Role
from contextbench.datasets.base import ConversationSample, Dataset, register_dataset

_SESSION_RE = re.compile(r"^session_(\d+)$")
_IMAGE_CAPTION_TEMPLATE = "{text} [shared a photo: {caption}]"


@register_dataset("locomo")
class LoCoMoDataset(Dataset):
    """Loader for the LoCoMo JSON file.

    Parses the whole file eagerly in ``__init__`` (the file is small) and caches
    immutable :class:`ConversationSample` objects.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Load and parse the LoCoMo JSON.

        Args:
            config: Dataset config providing ``path`` and optional
                ``num_samples`` / ``qa_categories`` / ``include_image_captions``.

        Raises:
            FileNotFoundError: If ``config.path`` does not exist.
        """
        super().__init__(config)
        path = Path(config.path)
        if not path.is_file():
            raise FileNotFoundError(
                f"LoCoMo data file not found at '{path}'. Download the dataset and "
                "point dataset.path at it (see README)."
            )
        with path.open(encoding="utf-8") as handle:
            raw_samples: list[dict[str, Any]] = json.load(handle)

        if config.num_samples is not None:
            raw_samples = raw_samples[: config.num_samples]

        self._samples: tuple[ConversationSample, ...] = tuple(
            self._build_sample(raw) for raw in raw_samples
        )

    def __iter__(self) -> Iterator[ConversationSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    # -- parsing ---------------------------------------------------------------

    def _build_sample(self, raw: dict[str, Any]) -> ConversationSample:
        conv = raw["conversation"]
        sample_id = str(raw.get("sample_id", id(raw)))
        speaker_a = conv.get("speaker_a")
        speaker_b = conv.get("speaker_b")

        session_keys = sorted(
            (k for k in conv if _SESSION_RE.match(k)),
            key=lambda k: int(_SESSION_RE.match(k).group(1)),  # type: ignore[union-attr]
        )

        messages: list[Message] = []
        session_dates: dict[str, Any] = {}
        turn_id = 0
        for session_key in session_keys:
            date = conv.get(f"{session_key}_date_time")
            session_dates[session_key] = date
            for turn in conv[session_key]:
                messages.append(
                    self._build_message(turn, session_key, date, speaker_a, speaker_b, turn_id)
                )
                turn_id += 1

        qa_items = self._build_qa_items(raw.get("qa", []), sample_id)

        return ConversationSample(
            sample_id=sample_id,
            messages=tuple(messages),
            qa_items=qa_items,
            speakers=tuple(s for s in (speaker_a, speaker_b) if s is not None),
            meta={
                "session_dates": session_dates,
                "num_sessions": len(session_keys),
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
            },
        )

    def _build_message(
        self,
        turn: dict[str, Any],
        session_id: str,
        date: Any,
        speaker_a: str | None,
        speaker_b: str | None,
        turn_id: int,
    ) -> Message:
        speaker = turn.get("speaker")
        role: Role = "user" if speaker == speaker_a else "assistant"

        text = str(turn.get("text", ""))
        caption = turn.get("blip_caption")
        if caption and self._config.include_image_captions:
            content = _IMAGE_CAPTION_TEMPLATE.format(text=text, caption=caption)
        else:
            content = text

        meta: dict[str, Any] = {"dia_id": turn.get("dia_id"), "session_date": date}
        for key in ("img_url", "blip_caption", "query"):
            if key in turn:
                meta[key] = turn[key]

        return Message(
            role=role,
            content=content,
            turn_id=turn_id,
            speaker=speaker,
            session_id=session_id,
            meta=meta,
        )

    def _build_qa_items(self, raw_qa: list[dict[str, Any]], sample_id: str) -> tuple[QAItem, ...]:
        categories = self._config.qa_categories
        items: list[QAItem] = []
        for index, qa in enumerate(raw_qa):
            category = qa.get("category")
            if categories is not None and category not in categories:
                continue
            items.append(self._build_qa(qa, sample_id, index))
        return tuple(items)

    @staticmethod
    def _build_qa(qa: dict[str, Any], sample_id: str, index: int) -> QAItem:
        # Gold answer: prefer the explicit `answer`; adversarial (category 5)
        # questions carry theirs in `adversarial_answer`. A few category-5 items
        # provide both, so the gold source and the adversarial label are tracked
        # independently — the label follows the dataset's own category.
        if "answer" in qa:
            gold = str(qa["answer"])
            answer_source = "answer"
        elif "adversarial_answer" in qa:
            gold = str(qa["adversarial_answer"])
            answer_source = "adversarial_answer"
        else:
            raise ValueError(
                f"QA {sample_id}#{index} has neither 'answer' nor 'adversarial_answer'."
            )

        category = qa.get("category")
        evidence = tuple(str(e) for e in (qa.get("evidence") or ()))
        return QAItem(
            question=str(qa["question"]),
            answer=gold,
            evidence=evidence,
            category=category,
            qa_id=f"{sample_id}::q{index}",
            meta={"adversarial": category == 5, "answer_source": answer_source},
        )
