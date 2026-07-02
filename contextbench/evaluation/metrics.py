"""Scored records and aggregate metrics.

Separates *quality* metrics (exact match, F1) from *system* metrics (context
size, tokens, latency) but reports both together per condition, since the whole
point of the framework is to trade them off across policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from contextbench.engine.conversation_runner import QAPrediction


@dataclass(frozen=True, slots=True)
class ScoredPrediction:
    """A prediction plus its quality scores."""

    prediction: QAPrediction
    exact_match: float
    f1: float


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Mean quality and system metrics over a set of scored predictions."""

    n: int
    exact_match: float
    f1: float
    avg_context_tokens: float
    avg_context_messages: float
    avg_dropped_messages: float
    avg_prompt_tokens: float
    avg_response_tokens: float
    avg_latency_s: float

    @classmethod
    def empty(cls) -> AggregateMetrics:
        """Return a zero-filled aggregate (for empty inputs)."""
        return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def aggregate(scored: Sequence[ScoredPrediction]) -> AggregateMetrics:
    """Average quality and system metrics over ``scored`` predictions."""
    n = len(scored)
    if n == 0:
        return AggregateMetrics.empty()

    def mean(value) -> float:
        return sum(value(s) for s in scored) / n

    return AggregateMetrics(
        n=n,
        exact_match=mean(lambda s: s.exact_match),
        f1=mean(lambda s: s.f1),
        avg_context_tokens=mean(lambda s: s.prediction.context_tokens),
        avg_context_messages=mean(lambda s: s.prediction.context_messages),
        avg_dropped_messages=mean(lambda s: s.prediction.dropped_messages),
        avg_prompt_tokens=mean(lambda s: s.prediction.prompt_tokens),
        avg_response_tokens=mean(lambda s: s.prediction.response_tokens),
        avg_latency_s=mean(lambda s: s.prediction.latency_s),
    )
