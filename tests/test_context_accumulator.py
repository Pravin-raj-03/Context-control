"""Tests for Module 4: semantic change accumulators (Sum vs Displacement)."""

from __future__ import annotations

import numpy as np
import pytest

from contextbench.context.accumulator import (ACCUMULATOR_REGISTRY, DisplacementAccumulator,
                                              EMAAccumulator, SemanticChangeAccumulator,
                                              SumAccumulator)
from contextbench.context.embedding_metric import CosineDistanceMetric
from contextbench.context.state import SemanticState


def _state(vec):
    return SemanticState(embedding=np.array(vec, dtype=np.float32),
                         token_count=1, turn_index=0, timestamp=0.0)


def test_registry_has_all_three():
    assert "sum" in ACCUMULATOR_REGISTRY
    assert "displacement" in ACCUMULATOR_REGISTRY
    assert "ema" in ACCUMULATOR_REGISTRY
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


# -- EMA accumulator --

def test_ema_registered_and_rejects_bad_alpha():
    assert isinstance(ACCUMULATOR_REGISTRY.create("ema"), EMAAccumulator)
    with pytest.raises(ValueError):
        EMAAccumulator(alpha=0.0)
    with pytest.raises(ValueError):
        EMAAccumulator(alpha=1.5)


def test_ema_weighted_average_matches_formula():
    acc = EMAAccumulator(alpha=0.5)
    s = _state([1, 0])
    assert acc.update(0.4, s) == pytest.approx(0.5 * 0.4)
    assert acc.update(0.2, s) == pytest.approx(0.5 * 0.2 + 0.5 * 0.2)


def test_ema_resets_to_zero():
    acc = EMAAccumulator(alpha=0.5)
    acc.update(1.0, _state([1, 0]))
    assert acc.value > 0.0
    acc.reset()
    assert acc.value == 0.0


def test_ema_damps_single_turn_noise_more_than_sum():
    """The whole point of offering EMA as a trigger accumulator: one noisy
    turn should move it less than Sum, which counts every delta in full."""
    summ, ema = SumAccumulator(), EMAAccumulator(alpha=0.3)
    s = _state([1, 0])
    summ.update(0.5, s)
    ema.update(0.5, s)
    assert ema.value < summ.value


def test_ema_decays_toward_zero_during_a_quiet_stretch():
    """A long run of ~zero deltas should pull EMA back down — this is the
    signal a context-rot detector reads."""
    acc = EMAAccumulator(alpha=0.3)
    s = _state([1, 0])
    acc.update(0.9, s)
    high = acc.value
    for _ in range(20):
        acc.update(0.0, s)
    assert acc.value < high * 0.01
