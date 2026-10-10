"""Conversation history store and replay planning.

Two small, orthogonal pieces:

* :class:`Conversation` is the mutable, authoritative history the
  :class:`~contextbench.engine.conversation_runner.ConversationRunner` owns and
  grows turn by turn. It is the single source of truth passed into
  ``policy.build_context`` — policies never accumulate their own copy of the raw
  history.
* :class:`ConversationReplayer` is a *stateless* planner: given a
  :class:`~contextbench.datasets.base.ConversationSample`, it yields the turns to
  replay and marks the checkpoints at which QA should be evaluated. It owns the
  *timing* of evaluation and nothing else — no history, no policy, no model.

Separating the two keeps the runner's loop trivial::

    conv = Conversation()
    for step in replayer.iter_steps(sample):
        conv.append(step.message)
        policy.observe(step.message)
        for qa in step.checkpoint_qa:
            ...  # build_context(conv.history, ...) -> prompt -> model -> score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from contextbench.core.config_schema import QATiming
from contextbench.core.types import Message, QAItem
from contextbench.datasets.base import ConversationSample
from contextbench.evaluation.evidence import parse_evidence_ids


class Conversation:
    """A mutable, ordered store of conversation turns.

    Grows via :meth:`append` as turns are replayed. Exposes history as an
    immutable snapshot so a policy cannot mutate the runner's state.
    """

    def __init__(self) -> None:
        """Create an empty conversation."""
        self._history: list[Message] = []

    def append(self, message: Message) -> None:
        """Append one turn to the history."""
        self._history.append(message)

    def reset(self) -> None:
        """Clear all turns so the instance can be reused for a new conversation."""
        self._history.clear()

    @property
    def history(self) -> tuple[Message, ...]:
        """An immutable snapshot of the turns so far, in chronological order.

        Returns a fresh tuple on each access so callers cannot mutate internal
        state. Conversations here are short enough (hundreds of turns) that the
        copy is negligible relative to a model forward pass.
        """
        return tuple(self._history)

    @property
    def last(self) -> Message | None:
        """The most recently appended turn, or ``None`` if empty."""
        return self._history[-1] if self._history else None

    def __len__(self) -> int:
        return len(self._history)

    def __iter__(self) -> Iterator[Message]:
        return iter(self._history)

    def __repr__(self) -> str:
        return f"Conversation(turns={len(self._history)})"


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """One step of a conversation replay.

    Attributes:
        message: The turn to append to the conversation at this step.
        turn_index: 0-based index of this turn within the conversation.
        is_last: Whether this is the final turn of the conversation.
        checkpoint_qa: QA probes to evaluate *after* appending ``message``.
            Empty on non-checkpoint steps.
    """

    message: Message
    turn_index: int
    is_last: bool
    checkpoint_qa: tuple[QAItem, ...] = ()

    @property
    def is_checkpoint(self) -> bool:
        """Whether any QA should be evaluated at this step."""
        return bool(self.checkpoint_qa)


class ConversationReplayer:
    """Plans the turn-by-turn replay of a sample and the QA checkpoint timing.

    The replayer is stateless: it derives the plan purely from the sample and the
    configured timing, so a single instance can plan any number of samples.
    """

    def __init__(self, qa_timing: QATiming = "end") -> None:
        """Create a replayer.

        Args:
            qa_timing: When to evaluate QA. ``"end"`` (Phase 1) fires a single
                checkpoint after the whole conversation is replayed.
                ``"per_session"`` is reserved for streaming evaluation.
        """
        self._qa_timing = qa_timing

    def iter_steps(self, sample: ConversationSample) -> Iterator[ReplayStep]:
        """Yield the ordered replay steps for a sample.

        Args:
            sample: The conversation and its QA probes.

        Yields:
            :class:`ReplayStep` objects in chronological order. Under ``"end"``
            timing, only the final step carries ``checkpoint_qa`` (all probes).

        Raises:
            NotImplementedError: If ``qa_timing`` is not yet supported.
        """
        if self._qa_timing == "end":
            yield from self._iter_end(sample)
        elif self._qa_timing == "per_session":
            yield from self._iter_per_session(sample)
        else:  # pragma: no cover - guarded by config validation
            raise NotImplementedError(f"Unknown qa_timing '{self._qa_timing}'.")

    @staticmethod
    def _iter_end(sample: ConversationSample) -> Iterator[ReplayStep]:
        messages = sample.messages
        n = len(messages)
        if n == 0:
            # No turns to replay; nothing can be evaluated against empty history.
            return
        last_index = n - 1
        for index, message in enumerate(messages):
            is_last = index == last_index
            yield ReplayStep(
                message=message,
                turn_index=index,
                is_last=is_last,
                checkpoint_qa=sample.qa_items if is_last else (),
            )

    @staticmethod
    def _iter_per_session(sample: ConversationSample) -> Iterator[ReplayStep]:
        """Ask each probe as soon as its evidence is complete.

        A probe is evaluated after the last turn of the session that holds its
        latest evidence turn, i.e. mid-conversation, with that evidence among
        the most recent turns. Probes without evidence in the history are asked
        at the end. Probes due at the same turn keep their dataset order.
        """
        messages = sample.messages
        n = len(messages)
        if n == 0:
            return
        index_of = {m.meta.get("dia_id"): i for i, m in enumerate(messages)}
        session_end: dict[int, int] = {}
        end = n - 1
        for i in range(n - 1, -1, -1):
            if i < n - 1 and messages[i].session_id != messages[i + 1].session_id:
                end = i
            session_end[i] = end

        due: dict[int, list[QAItem]] = {}
        for qa in sample.qa_items:
            hits = [index_of[d] for d in parse_evidence_ids(qa.evidence) if d in index_of]
            at = session_end[max(hits)] if hits else n - 1
            due.setdefault(at, []).append(qa)

        for index, message in enumerate(messages):
            yield ReplayStep(
                message=message,
                turn_index=index,
                is_last=index == n - 1,
                checkpoint_qa=tuple(due.get(index, ())),
            )

    @property
    def qa_timing(self) -> QATiming:
        """The configured QA timing."""
        return self._qa_timing
