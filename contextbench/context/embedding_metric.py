r"""Cosine distance metric — Module 3 (Version 1 semantic change metric).

.. math::  \delta_t = 1 - \frac{C_{t-1} \cdot C_t}{\lVert C_{t-1}\rVert \, \lVert C_t\rVert}

Range :math:`[0, 2]`: 0 for identical direction, 1 for orthogonal, 2 for
opposite. For L2-normalized embeddings the cosine is a dot product, so this is
cheap. Degenerate zero-norm states yield 0 (no measurable change).

Complexity: :math:`O(d)` per call. Intended future siblings: Euclidean,
Mahalanobis, angular distance, or learned metrics — each a new
:class:`SemanticChangeMetric` registered under its own name.
"""

from __future__ import annotations

import numpy as np

from contextbench.context.semantic_metric import SemanticChangeMetric, register_metric
from contextbench.context.state import SemanticState


@register_metric("cosine")
class CosineDistanceMetric(SemanticChangeMetric):
    """Cosine distance between the two states' embeddings."""

    def distance(self, previous: SemanticState, current: SemanticState) -> float:
        a = np.asarray(previous.embedding, dtype=np.float64)
        b = np.asarray(current.embedding, dtype=np.float64)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        cosine = float(np.dot(a, b) / (norm_a * norm_b))
        cosine = max(-1.0, min(1.0, cosine))  # guard float drift outside [-1, 1]
        return 1.0 - cosine
