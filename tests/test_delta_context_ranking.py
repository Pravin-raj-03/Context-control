"""Tests for query-aware landmark ranking (hermetic; no real embedder needed).

``rank_checkpoints_by_relevance`` is the pure core of ΔContext's query-awareness
fix: previously ``build_context`` ignored ``query`` entirely and returned the
same context for every question in a conversation, while ``RetrievalPolicy``
re-searches per question -- a real structural gap, not just a tuning one. This
tests the ranking logic directly, independent of the real SentenceTransformer
embedder ``DeltaContextPolicy`` loads (which would make these tests slow and
network-dependent).
"""

from __future__ import annotations

import numpy as np
import pytest

from contextbench.context.checkpoint_manager import Checkpoint
from contextbench.context.embedding_metric import CosineDistanceMetric
from contextbench.context.state import SemanticState
from contextbench.policies.delta_context import (rank_checkpoints_by_relevance,
                                                  rank_turn_indices_by_relevance,
                                                  rank_turns_with_segment_prior)


def _state(vec, turn=0):
    return SemanticState(embedding=np.array(vec, dtype=np.float32), token_count=1,
                         turn_index=turn, timestamp=float(turn))


def _checkpoint(cid, vec, turn):
    return Checkpoint(checkpoint_id=cid, state=_state(vec, turn), turn_index=turn,
                      token_count=10, timestamp=float(turn), accumulated_change=1.0,
                      turn_span=(0, turn))


def test_ranks_closest_checkpoint_first():
    metric = CosineDistanceMetric()
    # Three checkpoints pointing in different directions; query aligned with c2.
    c1 = _checkpoint(0, [1.0, 0.0], turn=5)
    c2 = _checkpoint(1, [0.0, 1.0], turn=50)   # oldest by recency, but query-relevant
    c3 = _checkpoint(2, [-1.0, 0.0], turn=95)  # most recent, but query-irrelevant
    query = _state([0.0, 1.0])  # exactly matches c2

    ranked = rank_checkpoints_by_relevance([c1, c2, c3], query, metric)
    assert ranked[0] is c2  # relevance wins over recency


def test_recency_order_is_not_assumed():
    """The whole point: this must be able to reorder away from recency."""
    metric = CosineDistanceMetric()
    old_but_relevant = _checkpoint(0, [1.0, 0.0], turn=1)
    new_but_irrelevant = _checkpoint(1, [-1.0, 0.0], turn=999)
    query = _state([1.0, 0.0])

    ranked = rank_checkpoints_by_relevance(
        [new_but_irrelevant, old_but_relevant], query, metric
    )
    assert ranked[0] is old_but_relevant


def test_empty_checkpoints_returns_empty():
    metric = CosineDistanceMetric()
    assert rank_checkpoints_by_relevance([], _state([1.0, 0.0]), metric) == []


def test_ties_keep_all_checkpoints():
    metric = CosineDistanceMetric()
    a = _checkpoint(0, [1.0, 0.0], turn=1)
    b = _checkpoint(1, [1.0, 0.0], turn=2)
    query = _state([1.0, 0.0])
    ranked = rank_checkpoints_by_relevance([a, b], query, metric)
    assert set(ranked) == {a, b}
    assert len(ranked) == 2


# -- full-history turn ranking (expand_landmark_search) --

def test_rank_turn_indices_prefers_relevance_over_recency():
    metric = CosineDistanceMetric()
    turn_embeddings = {
        3: np.array([1.0, 0.0], dtype=np.float32),    # old, relevant
        50: np.array([0.0, 1.0], dtype=np.float32),   # newer, irrelevant
        99: np.array([-1.0, 0.0], dtype=np.float32),  # newest, opposite
    }
    query = _state([1.0, 0.0])
    ranked = rank_turn_indices_by_relevance(turn_embeddings, query, metric)
    assert ranked[0] == 3
    assert ranked[-1] == 99


def test_rank_turn_indices_covers_every_observed_turn_not_just_checkpoints():
    """The whole point of expand_landmark_search: the candidate pool is every
    turn ΔContext has seen, not the much sparser checkpoint set."""
    metric = CosineDistanceMetric()
    turn_embeddings = {i: np.array([float(i), 1.0], dtype=np.float32) for i in range(50)}
    query = _state([0.0, 1.0])
    ranked = rank_turn_indices_by_relevance(turn_embeddings, query, metric)
    assert set(ranked) == set(range(50))
    assert len(ranked) == 50


def test_rank_turn_indices_empty_is_empty():
    metric = CosineDistanceMetric()
    assert rank_turn_indices_by_relevance({}, _state([1.0, 0.0]), metric) == []


def _turns(*vectors):
    return {i: np.array(v, dtype=np.float32) for i, v in enumerate(vectors)}


def test_segment_prior_weight_zero_is_plain_cosine_ranking():
    turns = _turns([1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [0.8, 0.6])
    query = np.array([1.0, 0.0])
    plain = rank_turn_indices_by_relevance(turns, _state(query), CosineDistanceMetric())
    assert rank_turns_with_segment_prior(turns, [1], query, 0.0) == plain


def test_segment_prior_breaks_ties_toward_the_relevant_segment():
    # Turns 1 and 3 are equally similar to the query on their own, but turn 1
    # sits in a segment (turns 0-1, checkpoint at 1) that is about the query,
    # while turn 3's segment (turns 2-3) is not.
    turns = _turns([1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [0.6, 0.8])
    query = np.array([1.0, 0.0])
    ranked = rank_turns_with_segment_prior(turns, [1], query, 0.5)
    assert ranked.index(1) < ranked.index(3)
    assert ranked[0] == 0


def test_segment_prior_treats_turns_after_last_checkpoint_as_one_segment():
    turns = _turns([0.0, 1.0], [1.0, 0.0], [0.6, 0.8])
    # checkpoint at turn 0 -> segments {0} and the open tail {1, 2}
    ranked = rank_turns_with_segment_prior(turns, [0], np.array([1.0, 0.0]), 1.0)
    assert ranked == [1, 2, 0]
