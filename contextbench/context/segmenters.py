"""Baseline topic segmenters, for comparison with ΔContext's checkpoints.

Each function takes the turn embeddings of a history (one row per turn, in
order) and returns *boundary turns*: the last turn of every segment except the
final one, the convention :mod:`contextbench.policies.ranking` uses. All of them
work on the same single-turn embeddings ΔContext uses, so a comparison isolates
the segmentation rule.

* :func:`fixed_boundaries` -- every ``size`` turns; no semantic signal.
* :func:`adjacent_threshold_boundaries` -- online: cut wherever two consecutive
  turns are farther apart than ``threshold``.
* :func:`semantic_chunker_boundaries` -- the "semantic chunking" used by RAG
  toolkits (e.g. LangChain's ``SemanticChunker``): cut where the consecutive
  distance exceeds a percentile of all consecutive distances in the history.
  Offline -- it needs the whole history to set its threshold.
* :func:`texttiling_boundaries` -- TextTiling (Hearst, 1997) over embeddings:
  similarity between adjacent blocks of turns, depth scores at each gap, and
  Hearst's cutoff (mean depth minus half a standard deviation). Offline.

ΔContext itself segments online, as turns arrive, from its own checkpoints.
"""

from __future__ import annotations

import numpy as np


def _normalized(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def consecutive_distances(embeddings: np.ndarray) -> np.ndarray:
    """``d[i]`` = cosine distance between turns ``i-1`` and ``i`` (``d[0] = 0``)."""
    matrix = _normalized(embeddings)
    d = np.zeros(len(matrix))
    if len(matrix) > 1:
        d[1:] = 1.0 - np.sum(matrix[1:] * matrix[:-1], axis=1)
    return d


def fixed_boundaries(n_turns: int, size: int) -> list[int]:
    """Segments of ``size`` consecutive turns."""
    return list(range(size - 1, n_turns - 1, size))


def adjacent_threshold_boundaries(embeddings: np.ndarray, threshold: float) -> list[int]:
    """Cut before every turn whose distance to the previous turn exceeds ``threshold``."""
    d = consecutive_distances(embeddings)
    return [i - 1 for i in range(1, len(d)) if d[i] > threshold]


def semantic_chunker_boundaries(embeddings: np.ndarray, percentile: float) -> list[int]:
    """Cut where the consecutive distance exceeds its ``percentile`` over the history."""
    d = consecutive_distances(embeddings)
    if len(d) < 2:
        return []
    threshold = float(np.percentile(d[1:], percentile))
    return [i - 1 for i in range(1, len(d)) if d[i] > threshold]


def texttiling_boundaries(embeddings: np.ndarray, block: int = 3,
                          cutoff_std: float = 0.5) -> list[int]:
    """TextTiling over turn embeddings with Hearst's depth-score cutoff.

    The gap before turn ``i`` is scored by the cosine similarity between the
    mean embedding of the ``block`` turns before it and of the ``block`` turns
    from it on. A gap's depth is how far its score dips below the nearest peaks
    on either side; gaps deeper than ``mean - cutoff_std * std`` of all depths
    (Hearst's cutoff is ``cutoff_std = 0.5``), and deeper than their neighbours
    (local maxima), become boundaries.
    """
    matrix = _normalized(embeddings)
    n = len(matrix)
    if n < 2 * block + 1:
        return []
    gaps = np.arange(block, n - block + 1)
    scores = np.empty(len(gaps))
    for j, i in enumerate(gaps):
        left = matrix[i - block:i].mean(axis=0)
        right = matrix[i:i + block].mean(axis=0)
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        scores[j] = float(left @ right / denom) if denom > 0 else 0.0

    depth = np.zeros(len(scores))
    for j, s in enumerate(scores):
        left_peak = s
        for k in range(j - 1, -1, -1):
            if scores[k] < left_peak:
                break
            left_peak = scores[k]
        right_peak = s
        for k in range(j + 1, len(scores)):
            if scores[k] < right_peak:
                break
            right_peak = scores[k]
        depth[j] = (left_peak - s) + (right_peak - s)

    cutoff = depth.mean() - cutoff_std * depth.std()
    boundaries = []
    for j, i in enumerate(gaps):
        is_local_max = (j == 0 or depth[j] >= depth[j - 1]) and (
            j == len(depth) - 1 or depth[j] >= depth[j + 1])
        if depth[j] > cutoff and is_local_max:
            boundaries.append(int(i) - 1)
    return boundaries
