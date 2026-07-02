"""Tests for normalization, EM/F1, and the evaluator."""

from __future__ import annotations

import pytest

from contextbench.engine.conversation_runner import QAPrediction
from contextbench.evaluation.evaluator import EVALUATOR_REGISTRY, EvaluationReport
from contextbench.evaluation.exact_match import exact_match_score
from contextbench.evaluation.f1 import f1_score
from contextbench.evaluation.metrics import aggregate
from contextbench.evaluation.normalization import get_tokens, normalize_answer


def test_normalize_answer():
    assert normalize_answer("The Answer, please!") == "answer please"
    assert get_tokens("The QUICK brown.") == ["quick", "brown"]


def test_exact_match():
    assert exact_match_score("Paris", "paris.") == 1.0
    assert exact_match_score("7 May 2023", "7 may 2023") == 1.0
    assert exact_match_score("London", "Paris") == 0.0


@pytest.mark.parametrize("pred,gold,expected", [
    ("quick brown fox", "quick brown fox", 1.0),
    ("cat", "dog", 0.0),
    ("the quick brown fox", "quick brown", 0.8),
    ("", "", 1.0),
    ("something", "", 0.0),
    ("the a an", "paris", 0.0),  # articles-only normalizes to empty
])
def test_f1(pred, gold, expected):
    assert f1_score(pred, gold) == pytest.approx(expected)


def _pred(pred, gold, category):
    return QAPrediction(
        sample_id="s", qa_id="q", budget=256, category=category, question="?",
        gold_answer=gold, prediction=pred, context_tokens=10, context_messages=2,
        dropped_messages=0, prompt_tokens=40, response_tokens=5, latency_s=0.5,
    )


def test_evaluator_overall_and_by_category():
    preds = [
        _pred("7 May 2023", "7 May 2023", 1),
        _pred("London", "Paris", 1),
        _pred("the quick brown fox", "quick brown", 2),
    ]
    report = EVALUATOR_REGISTRY.create("qa_em_f1").evaluate(preds)
    assert isinstance(report, EvaluationReport)
    assert report.overall.n == 3
    assert report.overall.exact_match == pytest.approx(1 / 3)
    assert report.overall.f1 == pytest.approx(1.8 / 3)
    assert report.by_category[1].exact_match == pytest.approx(0.5)
    assert report.by_category[2].f1 == pytest.approx(0.8)
    assert sum(m.n for m in report.by_category.values()) == 3


def test_aggregate_empty():
    assert aggregate([]).n == 0
