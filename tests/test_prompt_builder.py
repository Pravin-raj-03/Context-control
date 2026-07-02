"""Tests for the prompt builder (formatter only; no model needed)."""

from __future__ import annotations

import pytest

from contextbench.chatbot.formatting import TranscriptFormatter
from contextbench.chatbot.prompt_builder import PromptBuilder
from contextbench.core.types import Message


@pytest.fixture
def builder():
    return PromptBuilder(
        formatter=TranscriptFormatter(),
        system_prompt="SYS",
        question_template="Conversation:\n{transcript}\n\nQuestion: {question}\nAnswer:",
    )


def _msgs():
    return [
        Message(role="user", content="hello there", turn_id=0, speaker="Caroline"),
        Message(role="assistant", content="hi back", turn_id=1, speaker="Melanie"),
    ]


def test_structure_and_content(builder):
    prompt = builder.build(_msgs(), "who spoke?")
    assert [m["role"] for m in prompt] == ["system", "user"]
    assert prompt[0]["content"] == "SYS"
    user = prompt[1]["content"]
    assert "Caroline: hello there" in user and "Melanie: hi back" in user
    assert "who spoke?" in user


def test_transcript_matches_formatter(builder):
    msgs = _msgs()
    assert TranscriptFormatter().render(msgs) in builder.build(msgs, "q")[1]["content"]


def test_empty_context_placeholder(builder):
    assert "(no prior conversation)" in builder.build([], "q")[1]["content"]


def test_system_prompt_override(builder):
    assert builder.build(_msgs(), "q", system_prompt="OTHER")[0]["content"] == "OTHER"


def test_template_missing_field_rejected():
    with pytest.raises(ValueError):
        PromptBuilder(TranscriptFormatter(), "s", "no placeholders")
