"""Tests for Module 1: semantic state construction (hermetic; FakeEmbedder)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from contextbench.context.state import (Embedder, SemanticState, StateConstructor,
                                        WindowedCentroidConstructor)


def test_fake_embedder_satisfies_protocol(embedder):
    assert isinstance(embedder, Embedder)
    v = embedder.embed("hello world")
    assert v.shape == (embedder.dim,) and v.dtype == np.float32


def test_constructor_is_a_state_constructor(embedder):
    c = WindowedCentroidConstructor(embedder, window=3)
    assert isinstance(c, StateConstructor)
    assert c.window == 3 and c.current is None


def test_update_returns_state_with_metadata(embedder):
    c = WindowedCentroidConstructor(embedder, window=3)
    state = c.update("hello there", turn_index=0, token_count=5, timestamp=1.0)
    assert isinstance(state, SemanticState)
    assert state.turn_index == 0
    assert state.token_count == 5
    assert state.timestamp == 1.0
    assert state.embedding.shape == (embedder.dim,)
    assert c.current is state


def test_window_one_is_last_turn(embedder):
    c = WindowedCentroidConstructor(embedder, window=1)
    c.update("first turn", turn_index=0, token_count=2, timestamp=0.0)
    state = c.update("second different turn", turn_index=1, token_count=3, timestamp=1.0)
    assert np.allclose(state.embedding, embedder.embed("second different turn"))
    assert state.token_count == 3  # only the single windowed turn


def test_centroid_is_mean_of_window(embedder):
    c = WindowedCentroidConstructor(embedder, window=2)
    c.update("alpha", turn_index=0, token_count=1, timestamp=0.0)
    state = c.update("beta", turn_index=1, token_count=1, timestamp=1.0)
    expected = np.mean([embedder.embed("alpha"), embedder.embed("beta")], axis=0)
    assert np.allclose(state.embedding, expected, atol=1e-6)
    assert state.token_count == 2  # window token aggregate


def test_window_slides_and_aggregates_tokens(embedder):
    c = WindowedCentroidConstructor(embedder, window=2)
    c.update("a", turn_index=0, token_count=10, timestamp=0.0)
    c.update("b", turn_index=1, token_count=20, timestamp=1.0)
    state = c.update("c", turn_index=2, token_count=30, timestamp=2.0)
    # window now holds turns b, c
    expected = np.mean([embedder.embed("b"), embedder.embed("c")], axis=0)
    assert np.allclose(state.embedding, expected, atol=1e-6)
    assert state.token_count == 50


def test_reset_clears_state(embedder):
    c = WindowedCentroidConstructor(embedder, window=2)
    c.update("x", turn_index=0, token_count=1, timestamp=0.0)
    c.reset()
    assert c.current is None
    # after reset the window starts fresh
    state = c.update("y", turn_index=0, token_count=1, timestamp=0.0)
    assert np.allclose(state.embedding, embedder.embed("y"))


def test_state_embedding_excluded_from_equality(embedder):
    a = SemanticState(embedding=embedder.embed("p"), token_count=1, turn_index=0, timestamp=0.0)
    b = SemanticState(embedding=embedder.embed("q"), token_count=1, turn_index=0, timestamp=0.0)
    assert a == b  # differing embeddings, identical metadata -> equal (no ndarray truth error)


def test_state_is_frozen(embedder):
    s = SemanticState(embedding=embedder.embed("p"), token_count=1, turn_index=0, timestamp=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.token_count = 2  # type: ignore[misc]


def test_invalid_window_rejected(embedder):
    with pytest.raises(ValueError):
        WindowedCentroidConstructor(embedder, window=0)


def test_similar_texts_are_closer_than_dissimilar(embedder):
    """Sanity: the fake embedder is semantic-ish, so state moves meaningfully."""
    base = embedder.embed("the cat sat on the mat")
    similar = embedder.embed("the cat sat on the rug")
    different = embedder.embed("quantum chromodynamics lagrangian")
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos(base, similar) > cos(base, different)
