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
    ema_alpha: float = Field(default=0.3, gt=0.0, le=1.0)
    """Decay rate for the ``ema`` accumulator (ignored by ``sum``/``displacement``)."""

    # -- controller (Module 5) --
    # change_threshold/ignore_below default to values calibrated against the
    # real LoCoMo delta_t distribution with the default MiniLM embedder and
    # window=3 (see scripts/calibrate_delta.py): mean delta_t ~0.105,
    # p15 ~0.07. The previous defaults (0.5 class default / 0.2 commonly
    # used) fired a checkpoint on ~40% of all turns -- noise-level churn, not
    # topic boundaries. Re-run the calibration script and override these if
    # you change the embedder, window, or dataset.
    controller: str = "threshold"
    change_threshold: float = Field(default=2.0, ge=0.0)
    ignore_below: float = Field(default=0.07, ge=0.0)
    budget_pressure_threshold: float = Field(default=0.9, gt=0.0)
    min_snapshot_gap: int = Field(default=3, ge=1)

    # -- checkpoint manager (Module 6) --
    checkpoint_manager: str = "in_memory"

    # -- assembly (V1 checkpoint-anchored selection) --
    # 0.5 is a local peak found by sweeping 0.3/0.5/0.65 on a fast dev subset
    # once landmarks became query-ranked (see delta_context.yaml for the
    # numbers): more budget helps query-relevant landmarks up to a point, then
    # starves the recency tail instead.
    landmark_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    """Fraction of the token budget reserved for checkpoint anchor turns
    (landmarks of past topics); the remainder holds the recent tail."""
    query_aware_landmarks: bool = True
    """Rank landmark candidates by relevance to the current question instead of
    pure recency, when a query is available. Falls back to recency automatically
    when it isn't."""
    expand_landmark_search: bool = True
    """When query-aware, rank every observed turn (reusing embeddings already
    cached in observe(), so no extra embedding cost) instead of only the
    checkpoint pool -- matches RetrievalPolicy's recall. Set False for the
    cheaper checkpoint-only search (V1.1 behaviour)."""
    segment_prior_weight: float = Field(default=0.0, ge=0.0)
    """Full-history search only: add this weight times the query's similarity
    to each turn's *topic segment* (the turns between consecutive checkpoints,
    as one centroid) to the turn's own similarity. A turn inside a segment that
    is about the question outranks an equally similar turn in an unrelated
    one. 0 disables it (pure per-turn relevance, V1.2 behaviour); this is the
    one ranking signal plain retrieval has no equivalent for."""

    # -- context-rot diagnostic (independent of the checkpoint trigger) --
    rot_ema_alpha: float = Field(default=0.2, gt=0.0, le=1.0)
    """Decay rate for the never-reset EMA of instantaneous delta used to
    detect context rot (semantic stagnation), tracked separately from
    whichever accumulator drives checkpointing."""
    rot_streak_threshold: int = Field(default=8, ge=1)
    """Consecutive turns with instantaneous delta below ``ignore_below``
    before ``policy_meta['context_rot_detected']`` flips to ``True``."""
