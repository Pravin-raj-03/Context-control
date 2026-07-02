"""Transcript rendering shared by token counting and prompt building.

The single most important fairness property of ContextBench is that the tokens a
policy *counts* against its budget are exactly the tokens the model *sees*. That
only holds if one component renders a conversation message to text and both the
token counter and the prompt builder use it. :class:`TranscriptFormatter` is that
component.

For LoCoMo-style QA the model answers questions *about* a dialogue, so messages
are rendered as speaker-labelled transcript lines (``"Caroline: ..."``) rather
than as chat turns; the chat template is applied once, later, around the whole
system+question prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from contextbench.core.types import Message


@dataclass(frozen=True, slots=True)
class TranscriptFormatter:
    """Renders messages to a plain-text transcript.

    Attributes:
        line_template: Format string with ``{speaker}`` and ``{content}`` fields
            applied per message.
        separator: String joining consecutive rendered lines.
    """

    line_template: str = "{speaker}: {content}"
    separator: str = "\n"

    def format_line(self, message: Message) -> str:
        """Render a single message to one transcript line.

        The dataset speaker name is used when present, falling back to the
        normalized role so synthetic messages still render sensibly.
        """
        speaker = message.speaker or message.role
        return self.line_template.format(speaker=speaker, content=message.content)

    def render(self, messages: Sequence[Message]) -> str:
        """Render a message sequence to a single transcript block."""
        return self.separator.join(self.format_line(m) for m in messages)
