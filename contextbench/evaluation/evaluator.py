"""The evaluator: scores raw predictions into per-item scores and aggregates.

Scoring is decoupled from generation on purpose: an :class:`Evaluator` consumes
the :class:`~contextbench.engine.conversation_runner.QAPrediction` records the
runner emits, so a run can be re-scored with a different metric without touching
the model. Selected by name via :data:`EVALUATOR_REGISTRY`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from contextbench.core.registry import Registry
from contextbench.engine.conversation_runner import QAPrediction
from contextbench.evaluation.exact_match import exact_match_score
from contextbench.evaluation.f1 import f1_score
from contextbench.evaluation.metrics import AggregateMetrics, ScoredPrediction, aggregate


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """The full scored result of a run: per-item scores and aggregates."""

    overall: AggregateMetrics
    by_category: dict[int | None, AggregateMetrics]
    scored: tuple[ScoredPrediction, ...]


class Evaluator(ABC):
    """Abstract evaluator scoring predictions into an :class:`EvaluationReport`."""

    @abstractmethod
    def evaluate(self, predictions: Sequence[QAPrediction]) -> EvaluationReport:
        """Score predictions and compute aggregates."""
        raise NotImplementedError


EVALUATOR_REGISTRY: Registry[Evaluator] = Registry("evaluator")
"""Global evaluator registry, keyed by ``experiment.evaluator``."""

register_evaluator = EVALUATOR_REGISTRY.register
"""Convenience alias for :meth:`Registry.register` on the evaluator registry."""


@register_evaluator("qa_em_f1")
class QAEvaluator(Evaluator):
    """Scores each prediction with exact match and token-level F1.

    Reports aggregates overall and broken down by QA category, so per-category
    behaviour (e.g. adversarial vs. single-hop) is visible.
    """

    def score_one(self, prediction: QAPrediction) -> ScoredPrediction:
        """Score a single prediction against its gold answer."""
        return ScoredPrediction(
            prediction=prediction,
            exact_match=exact_match_score(prediction.prediction, prediction.gold_answer),
            f1=f1_score(prediction.prediction, prediction.gold_answer),
        )

    def evaluate(self, predictions: Sequence[QAPrediction]) -> EvaluationReport:
        """Score all predictions and aggregate overall and per category."""
        scored = tuple(self.score_one(p) for p in predictions)

        by_category: dict[int | None, AggregateMetrics] = {}
        for category in {s.prediction.category for s in scored}:
            subset = [s for s in scored if s.prediction.category == category]
            by_category[category] = aggregate(subset)

        return EvaluationReport(
            overall=aggregate(scored),
            by_category=by_category,
            scored=scored,
        )
