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
from contextbench.evaluation.abstention import is_abstention
from contextbench.evaluation.exact_match import exact_match_score
from contextbench.evaluation.f1 import f1_score
from contextbench.evaluation.metrics import AggregateMetrics, ScoredPrediction, aggregate


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """The full scored result of a run: per-item scores and aggregates."""

    overall: AggregateMetrics
    by_category: dict[int | None, AggregateMetrics]
    scored: tuple[ScoredPrediction, ...]
    adversarial: AggregateMetrics | None = None
    """Aggregate over adversarial probes when the evaluator scores them
    separately from :attr:`overall` (``None`` otherwise)."""


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
            abstained=is_abstention(prediction.prediction),
        )

    def evaluate(self, predictions: Sequence[QAPrediction]) -> EvaluationReport:
        """Score all predictions and aggregate overall and per category."""
        scored = tuple(self.score_one(p) for p in predictions)
        return EvaluationReport(
            overall=aggregate(scored),
            by_category=_by_category(scored),
            scored=scored,
        )


@register_evaluator("locomo_qa")
class LoCoMoQAEvaluator(QAEvaluator):
    """LoCoMo scoring: EM/F1 on answerable probes, abstention on adversarial ones.

    An adversarial probe's gold is the trap answer (see
    :mod:`contextbench.evaluation.abstention`), so it scores 1.0 for both EM and
    F1 exactly when the model abstains, matching LoCoMo's own evaluation.
    Adversarial probes are reported in :attr:`EvaluationReport.adversarial` and
    under their own category, and kept out of :attr:`EvaluationReport.overall`
    so the headline F1 is over answerable questions only (the convention in
    the LoCoMo memory literature).
    """

    def score_one(self, prediction: QAPrediction) -> ScoredPrediction:
        if not prediction.adversarial:
            return super().score_one(prediction)
        abstained = is_abstention(prediction.prediction)
        return ScoredPrediction(
            prediction=prediction,
            exact_match=float(abstained),
            f1=float(abstained),
            abstained=abstained,
        )

    def evaluate(self, predictions: Sequence[QAPrediction]) -> EvaluationReport:
        scored = tuple(self.score_one(p) for p in predictions)
        answerable = [s for s in scored if not s.prediction.adversarial]
        adversarial = [s for s in scored if s.prediction.adversarial]
        return EvaluationReport(
            overall=aggregate(answerable),
            by_category=_by_category(scored),
            scored=scored,
            adversarial=aggregate(adversarial) if adversarial else None,
        )


def _by_category(scored: Sequence[ScoredPrediction]) -> dict[int | None, AggregateMetrics]:
    by_category: dict[int | None, AggregateMetrics] = {}
    for category in {s.prediction.category for s in scored}:
        by_category[category] = aggregate([s for s in scored if s.prediction.category == category])
    return by_category
