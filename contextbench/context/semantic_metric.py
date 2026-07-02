r"""Semantic change metric interface — Module 2.

Defines the abstraction for measuring the instantaneous semantic change

.. math::  \delta_t = d(C_{t-1}, C_t)

between two successive semantic states. Cosine distance is only *one*
implementation (Module 3); isolating the metric here means the accumulator and
controller never mention cosine, so alternative metrics (learned distances,
Mahalanobis, distributional divergences) plug in without touching them.

A metric must return a non-negative scalar; larger means "more changed".
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from contextbench.context.state import SemanticState
from contextbench.core.registry import Registry


class SemanticChangeMetric(ABC):
    """Abstract distance between two semantic states, selected by name."""

    @abstractmethod
    def distance(self, previous: SemanticState, current: SemanticState) -> float:
        """Return the non-negative semantic change ``δ`` from ``previous`` to ``current``."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """A human-readable metric name."""
        return type(self).__name__


METRIC_REGISTRY: Registry[SemanticChangeMetric] = Registry("semantic change metric")
"""Global metric registry, keyed by ``delta.metric``."""

register_metric = METRIC_REGISTRY.register
"""Convenience alias for :meth:`Registry.register` on the metric registry."""
