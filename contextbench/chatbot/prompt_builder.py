"""Prompt assembly.

The :class:`PromptBuilder` turns the three ingredients the framework agreed on —
a system prompt, the policy-selected messages, and the question — into the
chat-role message list the model consumes. It is deliberately **policy-agnostic**:
it receives already-selected messages and knows nothing about *how* they were
chosen.

Critically, it renders the transcript with the *same*
:class:`~contextbench.chatbot.formatting.TranscriptFormatter` instance the
tokenizer counts with, so the tokens a policy charged against its budget are
exactly the transcript tokens that end up in the prompt.
"""

from __future__ import annotations

from typing import Sequence

from contextbench.chatbot.formatting import TranscriptFormatter
from contextbench.chatbot.tokenizer import ChatMessage
from contextbench.core.config_schema import ExperimentConfig
from contextbench.core.types import Message

_EMPTY_TRANSCRIPT_PLACEHOLDER = "(no prior conversation)"


class PromptBuilder:
    """Assembles the ``[system, user]`` chat prompt for a QA generation."""

    def __init__(
        self,
        formatter: TranscriptFormatter,
        system_prompt: str,
        question_template: str,
    ) -> None:
        """Create a prompt builder.

        Args:
            formatter: The transcript formatter — must be the same instance given
                to the tokenizer so counted tokens equal prompted tokens.
            system_prompt: The system message content.
            question_template: A format string with ``{transcript}`` and
                ``{question}`` fields that produces the user message content.

        Raises:
            ValueError: If ``question_template`` is missing a required field.
        """
        for required in ("{transcript}", "{question}"):
            if required not in question_template:
                raise ValueError(
                    f"question_template must contain {required}; got: {question_template!r}"
                )
        self._formatter = formatter
        self._system_prompt = system_prompt
        self._question_template = question_template

    @classmethod
    def from_config(
        cls, config: ExperimentConfig, formatter: TranscriptFormatter
    ) -> PromptBuilder:
        """Construct from the experiment config and the shared formatter."""
        return cls(
            formatter=formatter,
            system_prompt=config.system_prompt,
            question_template=config.question_template,
        )

    def build(
        self,
        context_messages: Sequence[Message],
        question: str,
        *,
        system_prompt: str | None = None,
    ) -> list[ChatMessage]:
        """Assemble the chat-role prompt.

        Args:
            context_messages: The policy-selected messages, in chronological order.
            question: The question to answer.
            system_prompt: Optional override of the configured system prompt.

        Returns:
            A two-element list ``[{"role": "system", ...}, {"role": "user", ...}]``
            ready for the tokenizer's chat template.
        """
        transcript = (
            self._formatter.render(context_messages)
            if context_messages
            else _EMPTY_TRANSCRIPT_PLACEHOLDER
        )
        user_content = self._question_template.format(
            transcript=transcript, question=question
        )
        return [
            {"role": "system", "content": system_prompt or self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    @property
    def formatter(self) -> TranscriptFormatter:
        """The transcript formatter used for rendering."""
        return self._formatter

    @property
    def system_prompt(self) -> str:
        """The configured system prompt."""
        return self._system_prompt
