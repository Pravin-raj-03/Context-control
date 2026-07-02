"""Tests for Modules 5-7: controller, checkpoint manager, config."""

from __future__ import annotations

import numpy as np
import pytest

from contextbench.context.checkpoint_manager import (CHECKPOINT_REGISTRY, Checkpoint,
                                                     InMemoryCheckpointManager)
from contextbench.context.config import DeltaContextConfig
from contextbench.context.controller import (CONTROLLER_REGISTRY, ControlAction,
                                             ControlSignals, ThresholdController)
from contextbench.context.state import SemanticState


def _state(turn=0):
    return SemanticState(embedding=np.zeros(4, dtype=np.float32), token_count=1,
                         turn_index=turn, timestamp=float(turn))


def _controller(**kw):
    params = dict(change_threshold=0.5, ignore_below=0.02,
                  budget_pressure_threshold=0.9, min_snapshot_gap=1)
    params.update(kw)
    return ThresholdController(**params)


def _signals(**kw):
    params = dict(accumulated_change=0.0, instantaneous_delta=0.1, context_tokens=100,
                  budget=1000, turns_since_checkpoint=5, state=_state())
    params.update(kw)
    return ControlSignals(**params)


# -- controller decisions --

def test_controller_registered():
    assert "threshold" in CONTROLLER_REGISTRY


def test_compress_on_budget_pressure():
    d = _controller().decide(_signals(context_tokens=950, budget=1000))
    assert d.action is ControlAction.COMPRESS


def test_snapshot_on_accumulated_change():
    d = _controller().decide(_signals(accumulated_change=0.6))
    assert d.action is ControlAction.SNAPSHOT


def test_ignore_on_tiny_delta():
    d = _controller().decide(_signals(instantaneous_delta=0.001))
    assert d.action is ControlAction.IGNORE


def test_keep_default():
    assert _controller().decide(_signals()).action is ControlAction.KEEP


def test_budget_pressure_dominates_snapshot():
    # both triggers active; COMPRESS wins by precedence
    d = _controller().decide(_signals(accumulated_change=0.9, context_tokens=999, budget=1000))
    assert d.action is ControlAction.COMPRESS


def test_min_snapshot_gap_debounces():
    d = _controller(min_snapshot_gap=10).decide(
        _signals(accumulated_change=0.9, turns_since_checkpoint=3))
    assert d.action is not ControlAction.SNAPSHOT  # gap not satisfied -> no snapshot


def test_retrieve_never_emitted_in_v1():
    actions = {
        _controller().decide(_signals(**kw)).action
        for kw in (
            dict(context_tokens=999, budget=1000),
            dict(accumulated_change=0.9),
            dict(instantaneous_delta=0.0),
            dict(),
        )
    }
    assert ControlAction.RETRIEVE not in actions


def test_decision_carries_signal_snapshot():
    d = _controller().decide(_signals(accumulated_change=0.6))
    assert "budget_pressure" in d.signals and d.reason


def test_zero_budget_is_infinite_pressure():
    assert _controller().decide(_signals(budget=0)).action is ControlAction.COMPRESS


# -- checkpoint manager --

def _checkpoint(cid, turn):
    return Checkpoint(checkpoint_id=cid, state=_state(turn), turn_index=turn,
                      token_count=10, timestamp=float(turn), accumulated_change=0.5,
                      turn_span=(0, turn))


def test_checkpoint_manager_add_latest_all_reset():
    mgr = CHECKPOINT_REGISTRY.create("in_memory")
    assert isinstance(mgr, InMemoryCheckpointManager)
    assert mgr.latest() is None and len(mgr) == 0
    mgr.add(_checkpoint(0, 5))
    mgr.add(_checkpoint(1, 12))
    assert len(mgr) == 2
    assert mgr.latest().checkpoint_id == 1
    assert tuple(c.checkpoint_id for c in mgr.all()) == (0, 1)
    assert mgr.next_id() == 2
    mgr.reset()
    assert len(mgr) == 0 and mgr.latest() is None


# -- config --

def test_config_defaults_and_validation():
    cfg = DeltaContextConfig()
    assert cfg.window == 3 and cfg.metric == "cosine" and cfg.accumulator == "sum"
    assert cfg.controller == "threshold"


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        DeltaContextConfig(window=0)
    with pytest.raises(ValueError):
        DeltaContextConfig(unknown_field=1)
