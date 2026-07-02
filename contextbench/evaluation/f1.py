"""Token-level F1 metric (SQuAD-style)."""

from __future__ import annotations

from collections import Counter

from contextbench.evaluation.normalization import get_tokens


def f1_score(prediction: str, gold: str) -> float:
    """Return the token-level F1 between prediction and gold.

    Follows the SQuAD convention: when either side has no tokens, F1 is 1.0 only
    if both are empty, otherwise 0.0.
    """
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(gold)

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
