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


def test_prefix_cached_backend_matches_plain_generation():
    """Reusing a shared prefix's KV cache and chunked prefill must not change outputs."""
    from contextbench.chatbot.model import HFCausalLM, PrefixCachedHFCausalLM
    from contextbench.chatbot.tokenizer import TokenizerWrapper

    config = _config().model
    model = PrefixCachedHFCausalLM(config, TokenizerWrapper(config.hf_id))
    model._PREFILL_CHUNK = 64  # exercise the chunked path on short prompts

    a = "\n".join(f"Ann: On day {i} I bought {i} postcards." for i in range(30))
    b = "\n".join(f"Bob: Day {i}: pasta with {i} tomatoes." for i in range(30))

    def prompt(transcript, question):
        return [{"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": f"{transcript}\n\nQuestion: {question}"}]

    # shared prefix, a full prefix switch, an identical repeat, then back again
    prompts = [prompt(a, "Postcards on day 12?"), prompt(a, "Postcards on day 3?"),
               prompt(b, "Tomatoes on day 5?"), prompt(b, "Tomatoes on day 5?"),
               prompt(a, "Postcards on day 12?")]
    for p in prompts:
        assert model.generate(p).text == HFCausalLM.generate(model, p).text
