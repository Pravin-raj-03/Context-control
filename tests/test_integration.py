"""Slow, opt-in integration tests that load the real model.

Run with ``CONTEXTBENCH_SLOW=1 pytest``. Skipped by default so the core suite
stays fast and hermetic.
"""

from __future__ import annotations

import pytest

from contextbench.core.config_schema import BenchmarkConfig

pytestmark = pytest.mark.slow


def _config():
    return BenchmarkConfig.model_validate(dict(
        model=dict(name="tinyllama", hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                   dtype="float32", context_window=2048, max_new_tokens=16),
        policy=dict(name="sliding_window"),
        dataset=dict(name="locomo", path="data/locomo10.json",
                     num_samples=1, max_qa_per_sample=2),
        experiment=dict(system_prompt="Answer concisely."),
        budget=256, seed=42,
    ))


def test_engine_run_and_evaluate():
    from contextbench.engine.benchmark_engine import BenchmarkEngine

    engine = BenchmarkEngine.from_config(_config())
    result = engine.run()
    report = engine.evaluate(result)

    assert result.system.num_predictions == 2
    assert len(result.predictions) == 2
    assert all(p.context_tokens <= 256 for p in result.predictions)
    assert report.overall.n == 2
    assert 0.0 <= report.overall.f1 <= 1.0

    # determinism
    result2 = engine.run()
    assert [p.prediction for p in result2.predictions] == \
           [p.prediction for p in result.predictions]
