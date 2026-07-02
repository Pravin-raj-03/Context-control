"""Tests for core types, registry, and config validation."""

from __future__ import annotations

import dataclasses

import pytest

from contextbench.core.config_schema import BenchmarkConfig
from contextbench.core.registry import Registry
from contextbench.core.types import ContextResult, Message, QAItem, TokenCounter


def _base_config_dict(**overrides):
    d = dict(
        model=dict(name="tinyllama", hf_id="x", context_window=2048, max_new_tokens=64),
        policy=dict(name="sliding_window"),
        dataset=dict(name="locomo", path="data/locomo10.json"),
        experiment=dict(system_prompt="sys"),
        budget=1024,
        seed=42,
    )
    d.update(overrides)
    return d


def test_message_is_frozen_and_hashable():
    m = Message(role="user", content="hi", turn_id=0, speaker="A")
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.content = "x"  # type: ignore[misc]
    assert hash(m) == hash(Message(role="user", content="hi", turn_id=0, speaker="A"))


def test_message_meta_excluded_from_equality():
    a = Message(role="user", content="hi", turn_id=0, meta={"x": 1})
    b = Message(role="user", content="hi", turn_id=0, meta={"y": 2})
    assert a == b


def test_context_result_num_messages():
    m = Message(role="user", content="hi", turn_id=0)
    assert ContextResult(messages=(m, m), token_count=3).num_messages == 2


def test_qa_item_defaults():
    q = QAItem(question="q?", answer="a")
    assert q.evidence == () and q.category is None


def test_registry_register_create_and_errors():
    reg: Registry = Registry("thing")

    @reg.register("Foo")
    class Foo:
        pass

    assert isinstance(reg.create("foo"), Foo)  # case-insensitive
    assert "foo" in reg
    with pytest.raises(ValueError):
        reg.register("foo")(Foo)  # duplicate
    with pytest.raises(KeyError):
        reg.get("missing")


def test_word_counter_satisfies_protocol(counter):
    assert isinstance(counter, TokenCounter)


def test_config_validates():
    cfg = BenchmarkConfig.model_validate(_base_config_dict())
    assert cfg.budget == 1024 and cfg.model.name == "tinyllama"


def test_config_budget_overflow_rejected():
    with pytest.raises(ValueError):
        BenchmarkConfig.model_validate(_base_config_dict(budget=2000))  # 2000+64 >= 2048


def test_config_rejects_unknown_key():
    bad = _base_config_dict()
    bad["dataset"]["pth"] = "typo"
    with pytest.raises(ValueError):
        BenchmarkConfig.model_validate(bad)


def test_policy_config_allows_extra():
    cfg = BenchmarkConfig.model_validate(
        _base_config_dict(policy=dict(name="p", custom_param=7))
    )
    assert cfg.policy.model_extra["custom_param"] == 7
