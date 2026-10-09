"""Tests for the Summary policy using the hermetic WordCounter.

These lock in the fix for a real bug found via full-benchmark diagnostics:
when the heuristic summary of dropped turns didn't fit ``summary_budget`` on
its own, the old exact-fit trim loop popped the tail (already empty) and then
popped the summary message itself, silently sending the model an EMPTY
context on every affected turn. ``avg_context_tokens`` was 0.0 across an
entire 1,986-QA LoCoMo run because of this.
"""

from __future__ import annotations

import pytest

from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import Message
from contextbench.policies.summary import SummaryPolicy


@pytest.fixture
def policy(counter):
    return SummaryPolicy(PolicyConfig(name="summary"), counter)


def _long_conversation(n: int) -> list[Message]:
    """``n`` short, distinctly-labelled turns so summary coverage is checkable."""
    return [
        Message(
            role="user" if i % 2 == 0 else "assistant",
            content=f"topic{i} detail",
            turn_id=i,
            speaker=f"Speaker{i % 2}",
            session_id="session_1",
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("budget", [0, 1, 3, 5, 8, 13, 21, 50, 1000])
def test_budget_never_exceeded(policy, counter, messages, budget):
    result = policy.build_context(messages, None, budget)
    assert result.token_count <= budget
    assert result.token_count == counter.count_messages(result.messages)


def test_budget_zero_and_empty_history(policy, messages):
    assert policy.build_context(messages, None, 0).messages == ()
    assert policy.build_context([], None, 100).messages == ()


def test_huge_budget_keeps_all_no_summary_needed(policy, counter, messages):
    result = policy.build_context(messages, None, 100000)
    assert result.messages == tuple(messages)  # nothing dropped -> no summary message
    assert result.dropped_messages == 0


def test_never_silently_empties_context_when_something_could_fit(policy, counter):
    """Regression test for the empty-context bug.

    A budget that can't hold the full history but *can* hold a summary header
    plus at least one bullet must never come back with an empty context.
    """
    history = _long_conversation(60)
    # Budget too small for any tail turn, but large enough for a real summary.
    budget = 40
    result = policy.build_context(history, None, budget)
    assert result.messages != (), "context must not be silently emptied"
    assert result.token_count <= budget


def test_summary_component_respects_its_own_sub_budget(policy, counter):
    history = _long_conversation(80)
    budget = 60
    result = policy.build_context(history, None, budget)
    summary_msgs = [m for m in result.messages if m.turn_id == -1]
    assert len(summary_msgs) <= 1
    if summary_msgs:
        summary_budget = int(budget * policy._cfg.summary_fraction)
        assert counter.count_message(summary_msgs[0]) <= summary_budget


def test_dropped_count_matches(policy, messages):
    result = policy.build_context(messages, None, 12)
    non_summary = [m for m in result.messages if m.turn_id != -1]
    assert result.dropped_messages == len(messages) - len(non_summary)


def test_reset_and_reuse(policy, messages):
    first = policy.build_context(messages, None, 12).token_count
    policy.reset()
    assert policy.build_context(messages, None, 12).token_count == first


def test_query_is_ignored(policy, messages):
    a = policy.build_context(messages, "a question", 12).messages
    b = policy.build_context(messages, None, 12).messages
    assert a == b


def test_summary_covers_full_dropped_span_not_just_the_opening(policy, counter):
    """Stride-sampling regression: a tight summary budget must not always pick
    only the earliest dropped turns — it should spread across the full span so
    later (pre-tail) topics get a chance at inclusion too."""
    history = _long_conversation(100)
    # Budget: room for the tail's last ~17 turns plus a handful (~6) of
    # summary bullets, out of ~83 dropped turns.
    budget = 100
    result = policy.build_context(history, None, budget)
    summary_msgs = [m for m in result.messages if m.turn_id == -1]
    assert summary_msgs, "expected a non-empty summary given this budget"

    text = summary_msgs[0].content
    included_indices = [i for i in range(100) if f"topic{i} " in text]
    assert len(included_indices) >= 2, "expected more than one bullet to fit"
    # The picked indices should span a meaningful fraction of the dropped
    # range, not cluster at the very start (which is what the old
    # oldest-first-only loop would do).
    span = max(included_indices) - min(included_indices)
    assert span > 20, f"summary only covered a narrow early slice: {included_indices}"
