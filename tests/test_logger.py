"""Tests for the experiment logger (no model needed)."""

from __future__ import annotations

import csv
import json

from contextbench.core.config_schema import BenchmarkConfig
from contextbench.engine.benchmark_engine import (BenchmarkResult, ConversationResult,
                                                  SystemMetrics)
from contextbench.engine.conversation_runner import QAPrediction
from contextbench.evaluation.evaluator import EVALUATOR_REGISTRY
from contextbench.runlog.logger import ExperimentLogger


def _config():
    return BenchmarkConfig.model_validate(dict(
        model=dict(name="tinyllama", hf_id="x", context_window=2048, max_new_tokens=64),
        policy=dict(name="sliding_window"),
        dataset=dict(name="locomo", path="data/locomo10.json"),
        experiment=dict(system_prompt="sys"),
        budget=512, seed=42,
    ))


def _prediction(cat):
    return QAPrediction(
        sample_id="s1", qa_id="s1::q0", budget=512, category=cat, question="q?",
        gold_answer="paris", prediction="paris", context_tokens=100, context_messages=4,
        dropped_messages=2, prompt_tokens=140, response_tokens=3, latency_s=0.4,
        policy_meta={"selected_messages": 4},
    )


def test_logger_writes_all_artifacts(tmp_path):
    config = _config()
    preds = (_prediction(1), _prediction(2))
    result = BenchmarkResult(
        config=config,
        conversation_results=(ConversationResult("s1", 10, preds),),
        system=SystemMetrics(total_wall_s=1.0, num_conversations=1, num_predictions=2,
                             peak_gpu_mem_mb=None, cpu_mem_mb=123.0),
    )
    report = EVALUATOR_REGISTRY.create("qa_em_f1").evaluate(result.predictions)

    logger = ExperimentLogger(tmp_path, config)
    logger.save_config()
    paths = logger.save_result(result, report)

    run_dir = logger.run_dir
    for name in ("config.json", "metrics.json", "per_sample.csv", "outputs.jsonl"):
        assert (run_dir / name).is_file()

    # metrics.json well-formed
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall"]["n"] == 2
    assert metrics["overall"]["exact_match"] == 1.0
    assert metrics["config_hash"] == logger.config_hash
    assert set(metrics["by_category"]) == {"1", "2"}

    # per_sample.csv has one row per prediction
    with (run_dir / "per_sample.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["gold_answer"] == "paris"

    # outputs.jsonl carries policy_meta
    lines = (run_dir / "outputs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["policy_meta"]["selected_messages"] == 4
    assert paths["metrics"].name == "metrics.json"


def test_run_dir_name_encodes_condition(tmp_path):
    logger = ExperimentLogger(tmp_path, _config())
    name = logger.run_dir.name
    assert "sliding_window" in name and "tinyllama" in name and "b512" in name
    assert logger.config_hash in name
