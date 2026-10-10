"""Evidence recall: did the policy put the answer-bearing turns in front of the model?

LoCoMo labels each QA probe with the ``dia_id``s of the turns that support its
answer. Evidence recall is the fraction of those turns present in the context a
policy selected. It scores the *policy* directly, independent of how well the
model then reads the context, so it separates "the policy dropped the answer"
from "the model missed it" -- and it needs no generation at all, which makes
selection-only sweeps take minutes instead of hours.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from contextbench.core.types import Message

# Evidence strings are usually one id ("D2:3" in LoCoMo, "<session>:<n>" in
# LongMemEval), but a few LoCoMo entries pack several into one string
# ("D8:6; D9:17"), so each string is split on separators.
_SEPARATOR_RE = re.compile(r"[;,\s]+")


def parse_evidence_ids(evidence: Iterable[str]) -> set[str]:
    """Split raw evidence strings into individual turn ids."""
    return {dia_id for raw in evidence for dia_id in _SEPARATOR_RE.split(raw) if dia_id}


def evidence_turn_ids(evidence: Iterable[str], history: Sequence[Message]) -> tuple[int, ...]:
    """``turn_id``s of the evidence turns present in ``history``, ascending."""
    wanted = parse_evidence_ids(evidence)
    return tuple(sorted(m.turn_id for m in history if m.meta.get("dia_id") in wanted))


def evidence_recall(
    evidence: Iterable[str],
    selected: Sequence[Message],
    history: Sequence[Message],
) -> float | None:
    """Fraction of a probe's evidence turns included in ``selected``.

    Only evidence ids that actually occur in ``history`` count toward the
    denominator: an id that no policy could ever select (a dataset typo, or a
    turn not yet replayed) would otherwise penalize every policy equally and
    just add noise.

    Returns:
        A value in ``[0, 1]``, or ``None`` when the probe has no reachable
        evidence (so it is excluded from the mean rather than scored 0).
    """
    available = {m.meta.get("dia_id") for m in history}
    wanted = parse_evidence_ids(evidence) & available
    if not wanted:
        return None
    present = {m.meta.get("dia_id") for m in selected}
    return len(wanted & present) / len(wanted)
