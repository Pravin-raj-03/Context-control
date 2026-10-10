"""Topic-segmentation quality metrics against a reference segmentation.

Segmentations are given as *boundary turns* (the last turn of every segment
except the final one) over a history of ``n`` turns.

* :func:`pk` -- Beeferman et al. (1999): the probability that two turns ``k``
  apart are wrongly judged to be in the same / different segments. Lower is
  better.
* :func:`window_diff` -- Pevzner & Hearst (2002): the share of ``k``-turn
  windows whose number of boundaries differs between reference and
  hypothesis. Lower is better; it also penalizes near-miss clusters.
* :func:`boundary_prf` -- precision / recall / F1 of boundaries, each
  hypothesis boundary matched to at most one reference boundary within
  ``tolerance`` turns.

``k`` defaults to half the mean reference segment length, the usual choice.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _segment_ids(boundaries: Sequence[int], n: int) -> np.ndarray:
    ids = np.zeros(n, dtype=np.int64)
    for b in sorted(set(boundaries)):
        if 0 <= b < n - 1:
            ids[b + 1:] += 1
    return ids


def default_k(reference: Sequence[int], n: int) -> int:
    """Half the mean reference segment length, at least 1."""
    segments = len(set(b for b in reference if 0 <= b < n - 1)) + 1
    return max(1, int(round(n / segments / 2)))


def pk(reference: Sequence[int], hypothesis: Sequence[int], n: int, k: int | None = None) -> float:
    """Beeferman's Pk (lower is better)."""
    k = default_k(reference, n) if k is None else k
    if n <= k:
        return 0.0
    ref, hyp = _segment_ids(reference, n), _segment_ids(hypothesis, n)
    same_ref = ref[:-k] == ref[k:]
    same_hyp = hyp[:-k] == hyp[k:]
    return float(np.mean(same_ref != same_hyp))


def window_diff(reference: Sequence[int], hypothesis: Sequence[int], n: int,
                k: int | None = None) -> float:
    """Pevzner & Hearst's WindowDiff (lower is better)."""
    k = default_k(reference, n) if k is None else k
    if n <= k:
        return 0.0
    ref_marks = np.zeros(n - 1)
    hyp_marks = np.zeros(n - 1)
    ref_marks[[b for b in set(reference) if 0 <= b < n - 1]] = 1
    hyp_marks[[b for b in set(hypothesis) if 0 <= b < n - 1]] = 1
    # boundaries in the window of gaps [i, i + k)
    ref_counts = np.convolve(ref_marks, np.ones(k), mode="valid")
    hyp_counts = np.convolve(hyp_marks, np.ones(k), mode="valid")
    return float(np.mean(ref_counts != hyp_counts))


def boundary_prf(reference: Sequence[int], hypothesis: Sequence[int],
                 tolerance: int = 0) -> tuple[float, float, float]:
    """Boundary precision, recall and F1 with one-to-one matching within ``tolerance``."""
    ref = sorted(set(reference))
    hyp = sorted(set(hypothesis))
    pairs = sorted(
        (abs(h - r), h, r) for h in hyp for r in ref if abs(h - r) <= tolerance
    )
    used_h: set[int] = set()
    used_r: set[int] = set()
    for _, h, r in pairs:
        if h not in used_h and r not in used_r:
            used_h.add(h)
            used_r.add(r)
    matched = len(used_h)
    precision = matched / len(hyp) if hyp else 0.0
    recall = matched / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
