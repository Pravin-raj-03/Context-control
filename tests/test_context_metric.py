"""Tests for Modules 2-3: semantic change metric interface + cosine distance."""

from __future__ import annotations

import numpy as np
import pytest

from contextbench.context.embedding_metric import CosineDistanceMetric
from contextbench.context.semantic_metric import METRIC_REGISTRY, SemanticChangeMetric
from contextbench.context.state import SemanticState


def _state(vec):
    return SemanticState(embedding=np.array(vec, dtype=np.float32),
                         token_count=1, turn_index=0, timestamp=0.0)


def test_cosine_registered_and_is_metric():
    metric = METRIC_REGISTRY.create("cosine")
    assert isinstance(metric, SemanticChangeMetric) and isinstance(metric, CosineDistanceMetric)


@pytest.mark.parametrize("a,b,expected", [
    ([1.0, 0.0], [1.0, 0.0], 0.0),   # identical
    ([1.0, 0.0], [0.0, 1.0], 1.0),   # orthogonal
    ([1.0, 0.0], [-1.0, 0.0], 2.0),  # opposite
    ([2.0, 0.0], [5.0, 0.0], 0.0),   # scale-invariant
])
def test_cosine_distance_values(a, b, expected):
    assert CosineDistanceMetric().distance(_state(a), _state(b)) == pytest.approx(expected)


def test_cosine_zero_vector_is_zero():
    assert CosineDistanceMetric().distance(_state([0.0, 0.0]), _state([1.0, 0.0])) == 0.0


def test_cosine_non_negative_and_symmetric():
    m = CosineDistanceMetric()
    a, b = _state([0.3, 0.7, 0.1]), _state([0.9, 0.1, 0.4])
    d1 = m.distance(a, b)
    d2 = m.distance(b, a)
    assert d1 >= 0 and d1 == pytest.approx(d2)
