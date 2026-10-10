"""Tests for the baseline segmenters and the segmentation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from contextbench.context.segmenters import (adjacent_threshold_boundaries,
                                             consecutive_distances, fixed_boundaries,
                                             semantic_chunker_boundaries,
                                             texttiling_boundaries)
from contextbench.evaluation.segmentation import boundary_prf, pk, window_diff


def _three_topics(length: int = 8) -> np.ndarray:
    """Turns on three orthogonal topics, ``length`` turns each, with mild noise."""
    rng = np.random.default_rng(0)
    rows = []
    for topic in range(3):
        base = np.zeros(6)
        base[topic] = 1.0
        for _ in range(length):
            rows.append(base + 0.05 * rng.standard_normal(6))
    return np.asarray(rows)


def test_consecutive_distances_peak_at_topic_switches():
    d = consecutive_distances(_three_topics())
    assert d[0] == 0.0
    assert set(np.argsort(-d)[:2]) == {8, 16}


def test_fixed_boundaries():
    assert fixed_boundaries(10, 4) == [3, 7]
    assert fixed_boundaries(8, 4) == [3]


@pytest.mark.parametrize("segment", [
    lambda e: adjacent_threshold_boundaries(e, 0.5),
    lambda e: semantic_chunker_boundaries(e, 90),
    lambda e: texttiling_boundaries(e, block=3),
])
def test_segmenters_find_clear_topic_switches(segment):
    assert set(segment(_three_topics())) >= {7, 15}


def test_pk_and_windowdiff_are_zero_for_a_perfect_match_and_positive_otherwise():
    reference = [7, 15]
    assert pk(reference, reference, 24) == 0.0
    assert window_diff(reference, reference, 24) == 0.0
    assert pk(reference, [3, 11, 19], 24) > 0.0
    assert window_diff(reference, [], 24) > 0.0


def test_boundary_prf_matches_one_to_one_within_tolerance():
    assert boundary_prf([7, 15], [7, 15]) == (1.0, 1.0, 1.0)
    p, r, f = boundary_prf([7, 15], [8, 9, 20], tolerance=1)
    assert (p, r) == (pytest.approx(1 / 3), pytest.approx(1 / 2))
    assert boundary_prf([7], [], tolerance=2) == (0.0, 0.0, 0.0)
