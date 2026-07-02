"""SQuAD-style answer normalization, shared by exact match and F1.

Normalizing both prediction and gold the same way is what makes string-overlap
metrics meaningful: ``"The Answer."`` and ``"answer"`` should score as identical.
The steps mirror the canonical SQuAD ``normalize_answer``: lowercase, strip
punctuation, drop the articles a/an/the, and collapse whitespace.
"""

from __future__ import annotations

import re
import string

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Return the SQuAD-normalized form of ``text``."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def get_tokens(text: str) -> list[str]:
    """Return the normalized whitespace tokens of ``text``."""
    return normalize_answer(text).split()
