r"""ΔContext configuration — Module 7.

A single validated schema for every ΔContext component, so a future
``DeltaContextPolicy`` reads one typed object and constructs its state builder,
metric, accumulator, controller, and checkpoint manager by name from the
registries. All thresholds are configurable; adaptive/budget-scaled thresholds
are a future extension that only changes the controller, not this schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DeltaContextConfig(BaseModel):
    """Validated configuration for the ΔContext infrastructure."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    # -- state construction (Module 1) --
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_device: str | None = None
    embedding_cache: str | None = None
    """Optional precomputed embedding cache (scripts/precompute_embeddings.py)."""
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
    detector: Literal["accumulator", "lookahead"] = "accumulator"
    """How topic boundaries are found. ``accumulator``: the controller fires a
    checkpoint when the accumulated change crosses ``change_threshold`` (V1/V2).
    ``lookahead`` (V3, delayed commitment): the gap after turn g is scored once
    ``lookahead`` more turns have arrived, as the distance between the open
    topic so far and those next turns; it becomes a boundary if the score is a
    local maximum and clears the threshold. Decisions arrive ``lookahead + 1``
    turns late and are never revised."""
    lookahead: int = Field(default=2, ge=1)
    """Turns of look-ahead for the ``lookahead`` detector (its decision delay is
    one more than this)."""
    left_context: int | None = None
    """``lookahead`` detector: how many of the open topic's most recent turns
    form its left representation (``None``: the whole open topic)."""
    threshold_mode: Literal["absolute", "zscore"] = "absolute"
    """``lookahead`` detector: ``absolute`` compares the gap score with
    ``change_threshold``; ``zscore`` with the running mean plus
    ``change_threshold`` standard deviations of this conversation's gap scores
    so far, so each conversation calibrates its own cut-off."""
    min_segment_turns: int = Field(default=2, ge=1)
    """``lookahead`` detector: smallest topic it will close, in turns."""
    reference: Literal["checkpoint", "segment_mean"] = "checkpoint"
    """What the current window is compared against when deciding a checkpoint.
    ``checkpoint``: the accumulator's own signal (for ``displacement``, the
    windowed state recorded at the last checkpoint). ``segment_mean``: the mean
    embedding of the current topic so far, excluding the current window, i.e.
    online clustering -- a new window far from everything said on the topic
    starts a new one. The latter suits short topics, where a single snapshot
    state is a noisy reference."""
    localize_boundaries: bool = False
    """Place each segment boundary at the sharpest single-turn semantic jump
    among the last ``window`` turns up to the checkpoint, instead of at the
    checkpoint turn itself. The windowed-centroid state only crosses the
    threshold once most of the window holds the new topic, so checkpoints
    fire ``window - 1`` turns late; without this, every new segment's first
    turns are filed under the previous topic. Affects only the segment prior."""
    topic_first_k: int = Field(default=0, ge=0)
    """Full-history search only: treat the history as a store of topic
    segments. Rank segments by query-to-centroid similarity, then rank only the
    turns inside the ``topic_first_k`` best (the rest follow, so the budget is
    always used). 0 disables it; takes precedence over the segment prior."""
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
