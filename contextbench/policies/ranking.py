"""Turn ranking shared by the query-aware policies."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class TopicFirstRanking:
    """Result of :func:`rank_topic_first`."""

    order: list[int]
    """Turn indices: the top segments' turns by similarity, then the rest."""
    top_spans: list[tuple[int, int]]
    """``(first, last)`` turn index of each selected segment, best first."""
    comparisons: int
    """Query similarities computed: one per segment plus one per turn inside
    the selected segments (what a topic-indexed store searches)."""
    segments: int
    """Number of segments in the store."""


def rank_topic_first(
    turn_embeddings: dict[int, np.ndarray],
    boundary_turns: Sequence[int],
    query_embedding: np.ndarray,
    top_k: int,
) -> TopicFirstRanking:
    """Two-stage, topic-first ranking over a history stored as topic segments.

    Stage 1 ranks segments by the cosine between the query and each segment's
    centroid and keeps the ``top_k`` best. Stage 2 ranks only the turns inside
    those segments by their own cosine to the query. Segments follow the same
    convention as :func:`rank_turns_with_segment_prior` (each runs up to and
    including a boundary turn).

    So a context budget is never left unused, the order continues with every
    other turn by its own similarity; ``comparisons`` counts only the two
    stages, which is all a topic-indexed store would search when the selected
    topics fill the budget.
    """
    indices = np.asarray(sorted(turn_embeddings), dtype=np.int64)
    matrix = np.stack([turn_embeddings[int(i)] for i in indices]).astype(np.float64)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    query = np.asarray(query_embedding, dtype=np.float64)
    query = query / max(float(np.linalg.norm(query)), 1e-12)

    segment = np.searchsorted(np.asarray(sorted(boundary_turns), dtype=np.int64),
                              indices, side="left")
    segment_ids = np.unique(segment)
    centroid_scores = []
    for segment_id in segment_ids:
        centroid = matrix[segment == segment_id].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroid_scores.append(float(centroid @ query) / norm if norm > 0.0 else -1.0)
    ranked_segments = segment_ids[np.argsort(-np.asarray(centroid_scores), kind="stable")]
    chosen = ranked_segments[:top_k]

    turn_score = matrix @ query
    in_chosen = np.isin(segment, chosen)
    first = np.argsort(-np.where(in_chosen, turn_score, -np.inf), kind="stable")[: int(in_chosen.sum())]
    rest = np.argsort(-np.where(in_chosen, -np.inf, turn_score), kind="stable")[: int((~in_chosen).sum())]
    spans = []
    for segment_id in chosen:
        members = indices[segment == segment_id]
        spans.append((int(members[0]), int(members[-1])))
    return TopicFirstRanking(
        order=[int(i) for i in indices[np.concatenate([first, rest])]],
        top_spans=spans,
        comparisons=len(segment_ids) + int(in_chosen.sum()),
        segments=len(segment_ids),
    )
