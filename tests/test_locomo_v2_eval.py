"""Tests for the corrected LoCoMo evaluation: session dates, abstention scoring
of adversarial probes, evidence recall, and selection-only runs."""

from __future__ import annotations

import pytest

from contextbench.chatbot.conversation import ConversationReplayer
from contextbench.chatbot.formatting import TranscriptFormatter, compact_date
from contextbench.chatbot.prompt_builder import PromptBuilder
from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import Message, QAItem
from contextbench.datasets.base import ConversationSample
from contextbench.engine.conversation_runner import ConversationRunner, QAPrediction
from contextbench.evaluation.abstention import is_abstention
from contextbench.evaluation.evaluator import EVALUATOR_REGISTRY
from contextbench.evaluation.evidence import evidence_recall, parse_evidence_ids
from contextbench.policies.sliding_window import SlidingWindowPolicy


def _msg(turn_id: int, dia_id: str, content: str = "hello there", date: str | None = None):
    meta = {"dia_id": dia_id}
    if date is not None:
        meta["session_date"] = date
    return Message(role="user", content=content, turn_id=turn_id, speaker="Ann",
                   session_id="session_1", meta=meta)


# -- session dates -----------------------------------------------------------

def test_compact_date_strips_time_of_day():
    assert compact_date("1:56 pm on 8 May, 2023") == "8 May 2023"
    assert compact_date("10:37 am on 27 June, 2023") == "27 June 2023"
    assert compact_date("2023/05/20 (Sat) 02:21") == "2023/05/20"
    assert compact_date("sometime") == "sometime"


def test_formatter_prefixes_session_date_only_when_enabled():
    message = _msg(0, "D1:1", "I went hiking.", date="1:56 pm on 8 May, 2023")
    assert TranscriptFormatter().format_line(message) == "Ann: I went hiking."
    dated = TranscriptFormatter(include_dates=True)
    assert dated.format_line(message) == "[8 May 2023] Ann: I went hiking."
    # messages without a date (e.g. a synthetic summary) render unprefixed
    assert dated.format_line(_msg(1, "D1:2", "hi")) == "Ann: hi"


# -- abstention ----------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Not mentioned in the conversation.",
    "The conversation does not mention her charity race.",
    "It doesn't say.",
    "There is no information about that.",
    "I don't know.",
    "This isn't specified anywhere.",
    "That cannot be determined from the conversation.",
    "Unknown",
])
def test_abstention_detected(text):
    assert is_abstention(text)


@pytest.mark.parametrize("text", [
    "7 May 2023",
    "Self-care is important.",
    "She painted a sunrise.",
    "She knows how to paint.",
])
def test_answers_are_not_abstentions(text):
    assert not is_abstention(text)


# -- evidence recall -----------------------------------------------------------

def test_parse_evidence_ids_handles_packed_entries():
    assert parse_evidence_ids(["D2:3", "D8:6; D9:17"]) == {"D2:3", "D8:6", "D9:17"}
    assert parse_evidence_ids(["answer_280352e9:4"]) == {"answer_280352e9:4"}


def test_evidence_recall_ignores_unreachable_ids():
    history = [_msg(i, f"D1:{i + 1}") for i in range(4)]
    selected = history[:1]
    assert evidence_recall(["D1:1", "D1:3"], selected, history) == pytest.approx(0.5)
    # D9:9 is not in the history, so only D1:1 counts toward the denominator
    assert evidence_recall(["D1:1", "D9:9"], selected, history) == pytest.approx(1.0)
    assert evidence_recall(["D9:9"], selected, history) is None
    assert evidence_recall([], selected, history) is None


# -- locomo_qa evaluator -------------------------------------------------------

def _pred(pred, gold, category, *, adversarial=False, recall=None):
    return QAPrediction(
        sample_id="s", qa_id="q", budget=256, category=category, question="?",
        gold_answer=gold, prediction=pred, context_tokens=10, context_messages=2,
        dropped_messages=0, prompt_tokens=40, response_tokens=5, latency_s=0.5,
        adversarial=adversarial, evidence_recall=recall,
    )


def test_locomo_evaluator_scores_adversarial_on_abstention_and_excludes_them():
    preds = [
        _pred("Paris", "Paris", 1, recall=1.0),
        _pred("London", "Paris", 4, recall=0.0),
        # falls for the trap: matches the trap answer but must score 0
        _pred("self-care is important", "self-care is important", 5, adversarial=True),
        _pred("Not mentioned in the conversation.", "self-care is important", 5,
              adversarial=True),
    ]
    report = EVALUATOR_REGISTRY.create("locomo_qa").evaluate(preds)
    assert report.overall.n == 2  # answerable only
    assert report.overall.f1 == pytest.approx(0.5)
    assert report.overall.evidence_recall == pytest.approx(0.5)
    assert report.adversarial is not None and report.adversarial.n == 2
    assert report.adversarial.f1 == pytest.approx(0.5)
    assert report.adversarial.abstention_rate == pytest.approx(0.5)
    assert report.by_category[5].exact_match == pytest.approx(0.5)


def test_legacy_evaluator_scores_are_unchanged():
    preds = [_pred("self-care is important", "self-care is important", 5, adversarial=True)]
    report = EVALUATOR_REGISTRY.create("qa_em_f1").evaluate(preds)
    assert report.overall.f1 == pytest.approx(1.0)
    assert report.adversarial is None


# -- selection-only runner -----------------------------------------------------

def test_selection_only_run_records_evidence_recall_without_a_model(counter):
    messages = tuple(_msg(i, f"D1:{i + 1}", "word " * 3) for i in range(6))
    sample = ConversationSample(
        sample_id="s",
        messages=messages,
        qa_items=(
            QAItem(question="q1", answer="a", evidence=("D1:6",), category=4, qa_id="s::q0"),
            QAItem(question="q2", answer="a", evidence=("D1:1",), category=4, qa_id="s::q1"),
        ),
    )
    runner = ConversationRunner(
        policy=SlidingWindowPolicy(PolicyConfig(name="sliding_window"), counter),
        prompt_builder=PromptBuilder(TranscriptFormatter(), "sys", "{transcript} {question}"),
        model=None,
        replayer=ConversationReplayer("end"),
        budget=10,
    )
    first, second = runner.run(sample).predictions
    assert first.prediction == "" and first.latency_s == 0.0
    assert first.evidence_recall == 1.0  # newest turn is in the window
    assert second.evidence_recall == 0.0  # oldest turn was dropped
    assert first.context_turn_ids and first.context_turn_ids[-1] == 5
