r"""ΔContext configuration — Module 7.

A single validated schema for every ΔContext component, so a future
``DeltaContextPolicy`` reads one typed object and constructs its state builder,
metric, accumulator, controller, and checkpoint manager by name from the
registries. All thresholds are configurable; adaptive/budget-scaled thresholds
are a future extension that only changes the controller, not this schema.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DeltaContextConfig(BaseModel):
    """Validated configuration for the ΔContext infrastructure."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    # -- state construction (Module 1) --
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_device: str | None = None
    window: int = Field(default=3, ge=1)

    # -- semantic change metric (Modules 2-3) --
    metric: str = "cosine"

    # -- accumulator (Module 4) --
    accumulator: str = "sum"

    # -- controller (Module 5) --
    controller: str = "threshold"
    change_threshold: float = Field(default=0.5, ge=0.0)
    ignore_below: float = Field(default=0.02, ge=0.0)
    budget_pressure_threshold: float = Field(default=0.9, gt=0.0)
    min_snapshot_gap: int = Field(default=1, ge=1)

    # -- checkpoint manager (Module 6) --
    checkpoint_manager: str = "in_memory"
