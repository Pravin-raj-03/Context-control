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

With ``include_dates`` on, each line is prefixed with its session date
(``"[8 May 2023] Caroline: ..."``). LoCoMo's temporal questions are answered
relative to those dates, so without them a whole question category is
unanswerable for every policy. The prefix is part of the rendered line, so it is
counted against the budget like any other transcript token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from contextbench.core.types import Message

# LoCoMo session stamps look like "1:56 pm on 8 May, 2023"; the time of day is
# never asked about, so only the calendar date is rendered.
_LOCOMO_DATE_RE = re.compile(r"(\d{1,2}) (\w+),? (\d{4})")


def compact_date(raw: str) -> str:
    """Reduce a session timestamp to its calendar date, e.g. ``"8 May 2023"``.

    Unrecognized formats are returned stripped but otherwise unchanged.
    """
    match = _LOCOMO_DATE_RE.search(raw)
    if match is None:
        return raw.strip()
    day, month, year = match.groups()
    return f"{day} {month} {year}"


@dataclass(frozen=True, slots=True)
class TranscriptFormatter:
    """Renders messages to a plain-text transcript.

    Attributes:
        line_template: Format string with ``{speaker}`` and ``{content}`` fields
            applied per message.
        separator: String joining consecutive rendered lines.
        include_dates: Prefix each line with its session date (from
            ``message.meta["session_date"]``) when one is present.
        date_template: Format string with a ``{date}`` field for that prefix.
    """

    line_template: str = "{speaker}: {content}"
    separator: str = "\n"
    include_dates: bool = False
    date_template: str = "[{date}] "

    def format_line(self, message: Message) -> str:
        """Render a single message to one transcript line.

        The dataset speaker name is used when present, falling back to the
        normalized role so synthetic messages still render sensibly.
        """
        speaker = message.speaker or message.role
        line = self.line_template.format(speaker=speaker, content=message.content)
        if self.include_dates:
            date = message.meta.get("session_date")
            if date:
                line = self.date_template.format(date=compact_date(str(date))) + line
        return line

    def render(self, messages: Sequence[Message]) -> str:
        """Render a message sequence to a single transcript block."""
        return self.separator.join(self.format_line(m) for m in messages)
