"""Exact-match metric."""

from __future__ import annotations

from contextbench.evaluation.normalization import normalize_answer


def exact_match_score(prediction: str, gold: str) -> float:
    """Return 1.0 if prediction and gold match after normalization, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(gold))
