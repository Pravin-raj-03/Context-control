"""Shared data primitives and the token-counter protocol.

These types are deliberately free of any dependency on models, datasets, or
policies so that every other subpackage can import them without creating
cycles. Instances are immutable (``frozen``) because a conversation history is a
record of facts that must not mutate underneath a policy mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

Role = Literal["system", "user", "assistant"]
"""Normalized message roles. Dataset-specific speaker names are preserved
separately on :attr:`Message.speaker`; ``role`` is what the prompt builder and
chat template consume."""


@dataclass(frozen=True, slots=True)
class Message:
    """A single conversation turn.

    Attributes:
        role: Normalized role driving the chat template.
        content: The utterance text.
        turn_id: Global 0-based index of this turn within its conversation,
            assigned at load time. Defines chronological order.
        speaker: Original dataset speaker name (e.g. ``"Caroline"``), or ``None``
            for synthetic messages such as the system prompt.
        session_id: Identifier of the session this turn belongs to (e.g.
            ``"session_1"``), or ``None`` if the dataset is not sessioned.
        meta: Auxiliary provenance (e.g. LoCoMo ``dia_id``, timestamps, image
            captions). Excluded from equality and hashing so two turns compare
            equal on their semantic content alone.
    """

    role: Role
    content: str
    turn_id: int
    speaker: str | None = None
    session_id: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class ContextResult:
    """The context a policy selected for one generation.

    Returning a structured result rather than a bare message list lets each
    policy attach diagnostics (via :attr:`policy_meta`) that become the
    qualitative evidence in the paper: which turns retrieval pulled, the summary
    text a summarizer used, ΔContext change scores, and so on.

    Attributes:
        messages: The selected turns, in chronological order, that the prompt
            builder will render. Messages are atomic — never truncated.
        token_count: Exact token count of ``messages`` as they will be rendered,
            measured with the same tokenizer the model uses. Guaranteed to be
            ``<= budget``.
        dropped_messages: How many history turns the policy excluded to satisfy
            the budget. Useful for measuring information loss.
        policy_meta: Free-form, policy-specific diagnostics. Excluded from
            equality; logged verbatim per generation.
    """

    messages: tuple[Message, ...]
    token_count: int
    dropped_messages: int = 0
    policy_meta: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def num_messages(self) -> int:
        """Number of selected messages."""
        return len(self.messages)


@dataclass(frozen=True, slots=True)
class QAItem:
    """A question/answer probe evaluated against a replayed conversation.

    Attributes:
        question: The natural-language question.
        answer: The gold answer, normalized to a string at load time.
        evidence: Dataset evidence identifiers (e.g. LoCoMo ``dia_id`` strings)
            supporting the answer. May be empty (e.g. adversarial questions).
        category: Dataset-specific question category id, or ``None``.
        qa_id: Stable identifier for this probe within its conversation.
        meta: Auxiliary fields carried through for later analysis.
    """

    question: str
    answer: str
    evidence: tuple[str, ...] = ()
    category: int | None = None
    qa_id: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False)


@runtime_checkable
class TokenCounter(Protocol):
    """Injected into every policy so all policies count tokens identically.

    A single implementation (the model's tokenizer wrapper) is shared across a
    benchmark run. This is the core fairness guarantee of ContextBench: two
    policies compared under the same budget are measured with the same ruler.

    Implementations must count tokens *as the model will actually see them*,
    including any per-message role/formatting overhead introduced by the chat
    template.
    """

    def count_text(self, text: str) -> int:
        """Return the token count of a raw string."""
        ...

    def count_message(self, message: Message) -> int:
        """Return the token count of one rendered message, role overhead included."""
        ...

    def count_messages(self, messages: Sequence[Message]) -> int:
        """Return the exact token count of a rendered message sequence.

        This may differ from the sum of :meth:`count_message` over the sequence
        because chat templates add sequence-level tokens (e.g. a leading BOS).
        """
        ...
