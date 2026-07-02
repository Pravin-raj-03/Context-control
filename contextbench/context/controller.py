r"""Context controller — Module 5.

The controller is the decision core of ΔContext. Given only *context signals*
(accumulated change, instantaneous change, current context size, budget, turns
since the last checkpoint) it decides what memory operation should happen. It
knows nothing about transformers, tokenizers, or prompts — it emits an action;
something else carries it out.

Version 1 is a transparent, fully-configurable threshold controller. All
thresholds are injected (nothing hardcoded), so budget-adaptive or learned
controllers are drop-in replacements. The decision is a *pure function* of the
signals, which makes it exhaustively testable.

Decision precedence (V1):

1. ``budget_pressure >= ρ``           -> ``COMPRESS``  (must free space)
2. ``A_t >= τ`` and gap satisfied     -> ``SNAPSHOT``  (semantic shift)
3. ``δ_t < ε``                        -> ``IGNORE``    (negligible turn)
4. otherwise                          -> ``KEEP``

``RETRIEVE`` exists in the action set for forward-compatibility but is never
emitted in V1 (retrieval is out of scope). The precedence itself is a research
variable and is documented so it can be made configurable later.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from contextbench.context.state import SemanticState
from contextbench.core.registry import Registry


class ControlAction(str, Enum):
    """The memory operations the controller can request."""

    KEEP = "keep"
    IGNORE = "ignore"
    SNAPSHOT = "snapshot"
    COMPRESS = "compress"
    RETRIEVE = "retrieve"  # reserved; not emitted in V1


@dataclass(frozen=True, slots=True)
class ControlSignals:
    """The budget/semantic signals the controller decides from."""

    accumulated_change: float
    instantaneous_delta: float
    context_tokens: int
    budget: int
    turns_since_checkpoint: int
    state: SemanticState | None = None

    @property
    def budget_pressure(self) -> float:
        """Fraction of the budget currently used (``inf`` if budget is 0)."""
        if self.budget <= 0:
            return math.inf
        return self.context_tokens / self.budget


@dataclass(frozen=True, slots=True)
class ControlDecision:
    """A controller decision plus the reason and signal snapshot (for logging)."""

    action: ControlAction
    reason: str
    signals: Mapping[str, Any] = field(default_factory=dict, compare=False)


class ContextController(ABC):
    """Abstract context controller, selected by name."""

    @abstractmethod
    def decide(self, signals: ControlSignals) -> ControlDecision:
        """Return the memory operation for the current signals."""
        raise NotImplementedError


CONTROLLER_REGISTRY: Registry[ContextController] = Registry("context controller")
"""Global controller registry, keyed by ``delta.controller``."""

register_controller = CONTROLLER_REGISTRY.register
"""Convenience alias for :meth:`Registry.register` on the controller registry."""


@register_controller("threshold")
class ThresholdController(ContextController):
    """A transparent threshold controller (Version 1)."""

    def __init__(
        self,
        *,
        change_threshold: float,
        ignore_below: float,
        budget_pressure_threshold: float,
        min_snapshot_gap: int = 1,
    ) -> None:
        """Configure the thresholds (all injected — nothing hardcoded).

        Args:
            change_threshold: ``τ`` — snapshot when accumulated change reaches it.
            ignore_below: ``ε`` — treat a turn as negligible below this δ.
            budget_pressure_threshold: ``ρ`` — compress at/above this fraction of
                the budget.
            min_snapshot_gap: Minimum turns between snapshots (light hysteresis to
                stop decision flicker); 1 means no debounce.
        """
        self._change_threshold = change_threshold
        self._ignore_below = ignore_below
        self._budget_pressure_threshold = budget_pressure_threshold
        self._min_snapshot_gap = min_snapshot_gap

    def decide(self, signals: ControlSignals) -> ControlDecision:
        snap = self._signal_snapshot(signals)

        if signals.budget_pressure >= self._budget_pressure_threshold:
            return ControlDecision(
                ControlAction.COMPRESS,
                f"budget pressure {signals.budget_pressure:.3f} >= "
                f"{self._budget_pressure_threshold}",
                snap,
            )

        if (
            signals.accumulated_change >= self._change_threshold
            and signals.turns_since_checkpoint >= self._min_snapshot_gap
        ):
            return ControlDecision(
                ControlAction.SNAPSHOT,
                f"accumulated change {signals.accumulated_change:.3f} >= "
                f"{self._change_threshold}",
                snap,
            )

        if signals.instantaneous_delta < self._ignore_below:
            return ControlDecision(
                ControlAction.IGNORE,
                f"delta {signals.instantaneous_delta:.3f} < {self._ignore_below}",
                snap,
            )

        return ControlDecision(ControlAction.KEEP, "no trigger crossed", snap)

    @staticmethod
    def _signal_snapshot(signals: ControlSignals) -> dict[str, Any]:
        return {
            "accumulated_change": signals.accumulated_change,
            "instantaneous_delta": signals.instantaneous_delta,
            "context_tokens": signals.context_tokens,
            "budget": signals.budget,
            "budget_pressure": signals.budget_pressure,
            "turns_since_checkpoint": signals.turns_since_checkpoint,
        }
