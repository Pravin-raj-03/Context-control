"""Abstention detection: did the model say the answer isn't in the conversation?

LoCoMo's adversarial questions (category 5) ask about something the conversation
never says -- typically a fact about one speaker re-attributed to the other. The
dataset's ``adversarial_answer`` field holds that trap answer, so scoring F1
against it rewards falling for the trap. The correct behaviour is to abstain,
and LoCoMo's own evaluation scores these probes by whether the prediction does.

Detection is phrase-based and deliberately broad over the ways a model says "not
in the conversation" ("not mentioned", "doesn't say", "no information", "I don't
know", ...). It is also applied to answerable questions, where it measures
over-abstention.
"""

from __future__ import annotations

import re

_NEGATED_VERB = r"(?:mention|say|specify|state|provide|discuss|indicate|reveal|talk about)"
_NEGATED_PARTICIPLE = r"(?:mentioned|specified|stated|provided|discussed|given|available|known|clear)"

_ABSTENTION_RE = re.compile(
    "|".join([
        rf"(?:\bnot|n't) (?:been |explicitly |directly )?{_NEGATED_PARTICIPLE}\b",
        rf"\b(?:do|does|did)(?: not|n't) {_NEGATED_VERB}",
        r"\b(?:do|does|did)(?: not|n't) know\b",
        r"\bno (?:information|mention|indication|details?|record|evidence|reference)\b",
        r"\bnothing (?:is |was )?(?:mentioned|said)\b",
        r"\b(?:cannot|can't|can not) be (?:determined|answered|inferred)\b",
        r"\bunable to (?:determine|answer|find)\b",
        r"\bunknown\b",
    ]),
    re.IGNORECASE,
)


def is_abstention(prediction: str) -> bool:
    """Return whether ``prediction`` declines to answer from the conversation."""
    return bool(_ABSTENTION_RE.search(prediction.replace("’", "'")))
