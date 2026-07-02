r"""Semantic change accumulator — Module 4.

The controller must not act on the noisy instantaneous :math:`\delta_t`; it acts
on an accumulated quantity :math:`A_t` that represents the semantic evolution
*since the last checkpoint*, and resets when a checkpoint fires.

There are two non-equivalent, defensible definitions — a genuine research fork —
so both are provided behind one interface:

* **Sum / path-length** (default):
  :math:`A_t = A_{t-1} + \delta_t`. Total churn; a conversation that drifts away
  and returns still accumulates a large :math:`A_t`.
* **Displacement**:
  :math:`A_t = d(C_{\text{checkpoint}}, C_t)`. Net movement from the checkpoint;
  returns toward 0 if the conversation comes back.

Complexity: Sum is :math:`O(1)`; Displacement is one metric evaluation,
:math:`O(d)`. Intended future siblings: decayed (EMA) accumulation, or variance-
based change energy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from contextbench.context.semantic_metric import SemanticChangeMetric
from contextbench.context.state import SemanticState
from contextbench.core.registry import Registry


class SemanticChangeAccumulator(ABC):
    """Accumulates semantic change since the last checkpoint, and resets."""

    @abstractmethod
    def update(self, delta: float, current: SemanticState) -> float:
        """Fold in one step and return the new accumulated value ``A_t``.

        Args:
            delta: The instantaneous change ``δ_t`` from the metric.
            current: The current semantic state ``C_t`` (used by displacement-style
                accumulators; ignored by sum-style ones).
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, checkpoint_state: SemanticState | None = None) -> None:
        """Reset the accumulator at a checkpoint.

        Args:
            checkpoint_state: The state that becomes the new reference point for
                displacement-style accumulators. Ignored by sum-style ones.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self) -> float:
        """The current accumulated value ``A_t``."""
        raise NotImplementedError


ACCUMULATOR_REGISTRY: Registry[SemanticChangeAccumulator] = Registry("accumulator")
"""Global accumulator registry, keyed by ``delta.accumulator``."""

register_accumulator = ACCUMULATOR_REGISTRY.register
"""Convenience alias for :meth:`Registry.register` on the accumulator registry."""


@register_accumulator("sum")
class SumAccumulator(SemanticChangeAccumulator):
    r"""Path-length accumulation: :math:`A_t = A_{t-1} + \delta_t`."""

    def __init__(self) -> None:
        self._value = 0.0

    def update(self, delta: float, current: SemanticState) -> float:
        self._value += float(delta)
        return self._value

    def reset(self, checkpoint_state: SemanticState | None = None) -> None:
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value


@register_accumulator("displacement")
class DisplacementAccumulator(SemanticChangeAccumulator):
    r"""Displacement accumulation: :math:`A_t = d(C_{\text{checkpoint}}, C_t)`.

    Needs a metric to measure distance from the checkpoint reference state.
    """

    def __init__(self, metric: SemanticChangeMetric) -> None:
        self._metric = metric
        self._reference: SemanticState | None = None
        self._value = 0.0

    def update(self, delta: float, current: SemanticState) -> float:
        if self._reference is None:
            # First observation acts as the implicit reference (displacement 0).
            self._reference = current
            self._value = 0.0
        else:
            self._value = self._metric.distance(self._reference, current)
        return self._value

    def reset(self, checkpoint_state: SemanticState | None = None) -> None:
        self._reference = checkpoint_state
        self._value = 0.0

    @property
    def value(self) -> float:
        return self._value
