"""Tests for the sliding-window policy using the hermetic WordCounter."""

from __future__ import annotations

import pytest

from contextbench.core.config_schema import PolicyConfig
from contextbench.policies.sliding_window import SlidingWindowPolicy


@pytest.fixture
def policy(counter):
    return SlidingWindowPolicy(PolicyConfig(name="sliding_window"), counter)


@pytest.mark.parametrize("budget", [0, 1, 3, 5, 8, 13, 21, 50, 1000])
def test_budget_never_exceeded(policy, counter, messages, budget):
    result = policy.build_context(messages, None, budget)
    assert result.token_count <= budget
    assert result.token_count == counter.count_messages(result.messages)


def test_selection_is_newest_contiguous_suffix(policy, messages):
    result = policy.build_context(messages, None, 12)
    if result.messages:
        idx = [messages.index(m) for m in result.messages]
        assert idx == list(range(idx[0], idx[0] + len(idx)))  # contiguous
        assert idx[-1] == len(messages) - 1  # anchored to newest
        assert result.messages[-1] == messages[-1]


def test_dropped_count(policy, messages):
    result = policy.build_context(messages, None, 12)
    assert result.dropped_messages == len(messages) - len(result.messages)


def test_budget_zero_and_empty_history(policy, messages):
    assert policy.build_context(messages, None, 0).messages == ()
    assert policy.build_context([], None, 100).messages == ()


def test_huge_budget_keeps_all(policy, counter, messages):
    result = policy.build_context(messages, None, 100000)
    assert result.messages == tuple(messages)
    assert result.dropped_messages == 0
    assert result.token_count == counter.count_messages(messages)


def test_message_larger_than_budget_is_atomic(policy, counter, messages):
    newest = messages[-1]
    result = policy.build_context(messages, None, counter.count_message(newest) - 1)
    assert result.messages == ()  # never truncated


def test_query_is_ignored(policy, messages):
    a = policy.build_context(messages, "a question", 12).messages
    b = policy.build_context(messages, None, 12).messages
    assert a == b


def test_reset_and_reuse(policy, messages):
    first = policy.build_context(messages, None, 12).token_count
    policy.reset()
    assert policy.build_context(messages, None, 12).token_count == first


def test_policy_meta_diagnostics(policy, messages):
    result = policy.build_context(messages, None, 12)
    meta = result.policy_meta
    assert meta["selected_messages"] == len(result.messages)
    if result.messages:
        assert meta["newest_turn_id"] == messages[-1].turn_id
