"""Tests for the conversation store and replay planner."""

from __future__ import annotations

import pytest

from contextbench.chatbot.conversation import Conversation, ConversationReplayer
from contextbench.core.types import QAItem
from contextbench.datasets.base import ConversationSample


def test_conversation_append_history_and_reset(messages):
    conv = Conversation()
    assert len(conv) == 0 and conv.last is None
    for m in messages:
        conv.append(m)
    assert conv.history == tuple(messages)
    assert conv.last == messages[-1]
    conv.reset()
    assert len(conv) == 0


def test_history_snapshot_is_immutable(messages):
    conv = Conversation()
    conv.append(messages[0])
    snap = conv.history
    assert isinstance(snap, tuple)
    with pytest.raises(TypeError):
        snap[0] = None  # type: ignore[index]


def _sample(messages):
    qa = (QAItem(question="q?", answer="a"), QAItem(question="r?", answer="b"))
    return ConversationSample(sample_id="s", messages=tuple(messages), qa_items=qa)


def test_replayer_end_timing(messages):
    steps = list(ConversationReplayer("end").iter_steps(_sample(messages)))
    assert len(steps) == len(messages)
    checkpoints = [s for s in steps if s.is_checkpoint]
    assert len(checkpoints) == 1
    assert checkpoints[0].is_last
    assert checkpoints[0].checkpoint_qa == _sample(messages).qa_items
    assert all(not s.is_checkpoint for s in steps[:-1])


def test_replayer_empty_sample():
    empty = ConversationSample(sample_id="e", messages=(), qa_items=())
    assert list(ConversationReplayer("end").iter_steps(empty)) == []


def test_replayer_per_session_asks_after_latest_evidence_session():
    from contextbench.core.types import Message

    sessions = ["session_1"] * 3 + ["session_2"] * 3 + ["session_3"] * 3
    messages = tuple(
        Message(role="user", content=f"turn {i}", turn_id=i, speaker="A",
                session_id=sid, meta={"dia_id": f"D:{i}"})
        for i, sid in enumerate(sessions)
    )
    early = QAItem(question="early?", answer="a", evidence=("D:1",))
    spans = QAItem(question="spans?", answer="b", evidence=("D:0", "D:4"))
    none = QAItem(question="none?", answer="c")
    sample = ConversationSample(sample_id="s", messages=messages,
                                qa_items=(early, spans, none))
    steps = list(ConversationReplayer("per_session").iter_steps(sample))
    asked = {s.turn_index: s.checkpoint_qa for s in steps if s.is_checkpoint}
    assert asked == {2: (early,), 5: (spans,), 8: (none,)}
