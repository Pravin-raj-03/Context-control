"""Experiment logging.

Every run writes a self-contained directory so that a result is fully
reproducible and analysable from its artifacts alone:

* ``config.json``   — the resolved, validated configuration
* ``metrics.json``  — system metrics and quality aggregates (overall + per category)
* ``per_sample.csv``— one row per QA probe (scores + system metrics), for pandas
* ``outputs.jsonl`` — raw per-prediction records incl. policy diagnostics, so a
  run can be re-scored or inspected offline
* ``run.log``       — the captured log stream (opt-in via :meth:`attach_run_log`)

The directory name encodes policy, model, budget, and a config hash, so two
directories are comparable iff their hashes match.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from contextbench.core.config_schema import BenchmarkConfig

if TYPE_CHECKING:
    from contextbench.engine.benchmark_engine import BenchmarkResult
    from contextbench.evaluation.evaluator import EvaluationReport

logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "sample_id", "qa_id", "category", "adversarial", "budget", "exact_match", "f1",
    "abstained", "evidence_recall", "context_tokens", "context_messages", "dropped_messages",
    "prompt_tokens", "response_tokens", "latency_s",
    "question", "gold_answer", "prediction",
]


class ExperimentLogger:
    """Writes a run's config and results into a dedicated directory."""

    def __init__(
        self, output_dir: str | Path, config: BenchmarkConfig, *, run_name: str | None = None
    ) -> None:
        """Create (and make) the run directory.

        Args:
            output_dir: Root directory under which run directories are created.
            config: The resolved configuration (serialized and hashed).
            run_name: Explicit run-directory name; if ``None``, a name encoding
                timestamp, policy, model, budget, and config hash is generated.
        """
        self._config = config
        self._config_hash = self._compute_config_hash(config)
        self._run_dir = Path(output_dir) / (run_name or self._default_run_name())
        self._run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        """The directory this run's artifacts are written to."""
        return self._run_dir

    @property
    def config_hash(self) -> str:
        """The short hash identifying comparable configurations."""
        return self._config_hash

    def attach_run_log(self, level: int = logging.INFO) -> logging.Handler:
        """Attach a file handler capturing logs into ``run.log``.

        Returns the handler so the caller can detach it when the run ends.
        """
        handler = logging.FileHandler(self._run_dir / "run.log", encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)
        return handler

    def save_config(self) -> Path:
        """Write the resolved config to ``config.json``."""
        path = self._run_dir / "config.json"
        payload = self._config.model_dump()
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def save_result(self, result: BenchmarkResult, report: EvaluationReport) -> dict[str, Path]:
        """Write metrics, per-sample rows, and raw outputs.

        Args:
            result: The raw benchmark result (system metrics).
            report: The scored evaluation report (quality metrics + per-item).

        Returns:
            Mapping of artifact name to the path written.
        """
        return {
            "metrics": self._save_metrics(result, report),
            "per_sample": self._save_per_sample(report),
            "outputs": self._save_outputs(report),
        }

    # -- internals -------------------------------------------------------------

    def _save_metrics(self, result: BenchmarkResult, report: EvaluationReport) -> Path:
        path = self._run_dir / "metrics.json"
        payload = {
            "run_dir": self._run_dir.name,
            "config_hash": self._config_hash,
            "policy": self._config.policy.name,
            "model": self._config.model.name,
            "budget": self._config.budget,
            "seed": self._config.seed,
            "system": dataclasses.asdict(result.system),
            "overall": dataclasses.asdict(report.overall),
            "by_category": {
                str(category): dataclasses.asdict(metrics)
                for category, metrics in sorted(
                    report.by_category.items(), key=lambda kv: (kv[0] is None, kv[0])
                )
            },
        }
        if report.adversarial is not None:
            payload["adversarial"] = dataclasses.asdict(report.adversarial)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _save_per_sample(self, report: EvaluationReport) -> Path:
        path = self._run_dir / "per_sample.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            for scored in report.scored:
                writer.writerow(self._row(scored))
        return path

    def _save_outputs(self, report: EvaluationReport) -> Path:
        path = self._run_dir / "outputs.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for scored in report.scored:
                record = self._row(scored)
                record["policy_meta"] = dict(scored.prediction.policy_meta)
                record["context_turn_ids"] = list(scored.prediction.context_turn_ids)
                handle.write(json.dumps(record, default=str) + "\n")
        return path

    @staticmethod
    def _row(scored) -> dict:
        p = scored.prediction
        return {
            "sample_id": p.sample_id,
            "qa_id": p.qa_id,
            "category": p.category,
            "adversarial": p.adversarial,
            "budget": p.budget,
            "exact_match": scored.exact_match,
            "f1": scored.f1,
            "abstained": scored.abstained,
            "evidence_recall": p.evidence_recall,
            "context_tokens": p.context_tokens,
            "context_messages": p.context_messages,
            "dropped_messages": p.dropped_messages,
            "prompt_tokens": p.prompt_tokens,
            "response_tokens": p.response_tokens,
            "latency_s": p.latency_s,
            "question": p.question,
            "gold_answer": p.gold_answer,
            "prediction": p.prediction,
        }

    def _default_run_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model = self._sanitize(self._config.model.name)
        policy = self._sanitize(self._config.policy.name)
        return f"{timestamp}_{policy}_{model}_b{self._config.budget}_{self._config_hash}"

    @staticmethod
    def _sanitize(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z._-]", "-", name)

    @staticmethod
    def _compute_config_hash(config: BenchmarkConfig) -> str:
        payload = json.dumps(config.model_dump(), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
