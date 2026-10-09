"""Turn ranking shared by the query-aware policies."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def rank_turns_with_segment_prior(
    turn_embeddings: dict[int, np.ndarray],
    boundary_turns: Sequence[int],
    query_embedding: np.ndarray,
    weight: float,
) -> list[int]:
    """Rank observed turns by own relevance plus ``weight`` x their segment's relevance.

    ``boundary_turns`` split the conversation into segments: each segment runs
    up to and including a boundary turn, and the turns after the last boundary
    form one more, still-open segment. ΔContext passes its checkpoint turns
    (semantic topic shifts); the retrieval baseline can pass fixed-size chunk
    ends or session ends instead, as controls. A segment's relevance is the cosine
    between the query and the normalized mean of its turns' embeddings. Scoring
    is ``cos(q, turn) + weight * cos(q, segment)``, highest first; ``weight=0``
    reduces to plain per-turn cosine ranking.

    Vectorized and pure (no embedder or policy state), so it is unit-testable
    with hand-built vectors.
    """
    indices = np.asarray(sorted(turn_embeddings), dtype=np.int64)
    matrix = np.stack([turn_embeddings[int(i)] for i in indices]).astype(np.float64)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    query = np.asarray(query_embedding, dtype=np.float64)
    query = query / max(float(np.linalg.norm(query)), 1e-12)

    score = matrix @ query
    if weight > 0.0:
        boundaries = np.asarray(sorted(boundary_turns), dtype=np.int64)
        segment = np.searchsorted(boundaries, indices, side="left")
        for segment_id in np.unique(segment):
            members = segment == segment_id
            centroid = matrix[members].mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0.0:
                score[members] += weight * float(centroid @ query) / norm

    order = np.argsort(-score, kind="stable")
    return [int(i) for i in indices[order]]
