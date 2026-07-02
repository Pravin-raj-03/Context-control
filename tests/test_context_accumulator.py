"""Tests for Module 4: semantic change accumulators (Sum vs Displacement)."""

from __future__ import annotations

import numpy as np
import pytest

from contextbench.context.accumulator import (ACCUMULATOR_REGISTRY, DisplacementAccumulator,
                                              SemanticChangeAccumulator, SumAccumulator)
from contextbench.context.embedding_metric import CosineDistanceMetric
from contextbench.context.state import SemanticState


def _state(vec):
    return SemanticState(embedding=np.array(vec, dtype=np.float32),
                         token_count=1, turn_index=0, timestamp=0.0)


def test_registry_has_both():
    assert "sum" in ACCUMULATOR_REGISTRY and "displacement" in ACCUMULATOR_REGISTRY
    assert isinstance(ACCUMULATOR_REGISTRY.create("sum"), SemanticChangeAccumulator)


def test_sum_accumulates_and_resets():
    acc = SumAccumulator()
    assert acc.update(0.3, _state([1, 0])) == pytest.approx(0.3)
    assert acc.update(0.5, _state([0, 1])) == pytest.approx(0.8)
    assert acc.value == pytest.approx(0.8)
    acc.reset()
    assert acc.value == 0.0


def test_displacement_measures_from_reference():
    metric = CosineDistanceMetric()
    acc = DisplacementAccumulator(metric)
    a, b = _state([1.0, 0.0]), _state([0.0, 1.0])
    acc.reset(a)                       # reference = a
    assert acc.update(0.0, a) == pytest.approx(0.0)   # distance to itself
    assert acc.update(0.0, b) == pytest.approx(1.0)   # orthogonal -> 1


def test_drift_and_return_distinguishes_the_two():
    """The core research distinction: A -> B -> A.

    Sum accumulates 2*d(A,B); displacement returns to ~0.
    """
    metric = CosineDistanceMetric()
    A, B = _state([1.0, 0.0]), _state([0.0, 1.0])
    d_ab = metric.distance(A, B)  # = 1.0

    summ = SumAccumulator()
    summ.update(metric.distance(A, B), B)  # step A->B
    summ.update(metric.distance(B, A), A)  # step B->A
    assert summ.value == pytest.approx(2 * d_ab)  # churn remembered

    disp = DisplacementAccumulator(metric)
    disp.reset(A)
    disp.update(0.0, B)  # displaced
    disp.update(0.0, A)  # returned
    assert disp.value == pytest.approx(0.0)  # net displacement zero


def test_displacement_first_update_is_reference_when_unset():
    acc = DisplacementAccumulator(CosineDistanceMetric())
    # no reset -> first observed state becomes the implicit reference
    assert acc.update(0.0, _state([1, 0])) == pytest.approx(0.0)
