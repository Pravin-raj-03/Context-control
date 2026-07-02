"""Shared orchestration helpers and the ``contextbench`` console entry point.

Both entry points (``python main.py`` for a quick single-conversation run and
``python benchmark.py`` for Hydra-driven sweeps) build a validated
:class:`BenchmarkConfig` and hand it to :func:`run_experiment`, which wires the
engine, runs, scores, and logs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from contextbench.core.config_schema import BenchmarkConfig
from contextbench.engine.benchmark_engine import BenchmarkEngine, BenchmarkResult
from contextbench.evaluation.evaluator import EvaluationReport
from contextbench.runlog.logger import ExperimentLogger

logger = logging.getLogger(__name__)


def compose_config(
    config_dir: str | Path,
    overrides: Sequence[str],
    *,
    config_name: str = "config",
) -> BenchmarkConfig:
    """Compose and validate a config from a Hydra config directory.

    Uses Hydra's compose API (not the ``@hydra.main`` decorator) so it works from
    any working directory and from an installed console script.

    Args:
        config_dir: Directory containing ``config.yaml`` and its groups.
        overrides: Hydra-style overrides, e.g. ``["budget=512", "model=qwen2.5-3b"]``.
        config_name: Root config file name (without extension).

    Returns:
        A validated :class:`BenchmarkConfig`.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    abs_dir = str(Path(config_dir).resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=abs_dir):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    raw = OmegaConf.to_container(cfg, resolve=True)
    return BenchmarkConfig.model_validate(raw)


def run_experiment(
    config: BenchmarkConfig, *, log: bool = True
) -> tuple[BenchmarkResult, EvaluationReport, Path | None]:
    """Build the engine, run, score, and (optionally) log a full experiment.

    Returns:
        ``(result, report, run_dir)`` where ``run_dir`` is ``None`` if logging is
        disabled.
    """
    experiment_logger: ExperimentLogger | None = None
    handler: logging.Handler | None = None
    if log:
        experiment_logger = ExperimentLogger(config.experiment.output_dir, config)
        handler = experiment_logger.attach_run_log()
        experiment_logger.save_config()

    try:
        engine = BenchmarkEngine.from_config(config)
        result = engine.run()
        report = engine.evaluate(result)
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)

    run_dir: Path | None = None
    if experiment_logger is not None:
        experiment_logger.save_result(result, report)
        run_dir = experiment_logger.run_dir
        if handler is not None:
            handler.close()

    return result, report, run_dir


def print_summary(
    config: BenchmarkConfig,
    result: BenchmarkResult,
    report: EvaluationReport,
    run_dir: Path | None,
    *,
    n_examples: int = 3,
) -> None:
    """Print a concise human-readable summary of a run."""
    o = report.overall
    s = result.system
    print("\n" + "=" * 68)
    print(f"ContextBench  |  policy={config.policy.name}  model={config.model.name}  "
          f"budget={config.budget}")
    print("=" * 68)
    print(f"predictions : {o.n}")
    print(f"exact match : {o.exact_match:.4f}")
    print(f"F1          : {o.f1:.4f}")
    print(f"avg context : {o.avg_context_tokens:.1f} tokens / "
          f"{o.avg_context_messages:.1f} msgs (dropped {o.avg_dropped_messages:.1f})")
    print(f"avg latency : {o.avg_latency_s:.2f}s   wall: {s.total_wall_s:.1f}s")
    gpu = f"{s.peak_gpu_mem_mb:.0f}MB" if s.peak_gpu_mem_mb is not None else "n/a (CPU)"
    print(f"memory      : cpu_rss={s.cpu_mem_mb:.0f}MB  gpu_peak={gpu}")

    if report.by_category:
        print("-" * 68)
        print("by category :")
        for category, metrics in sorted(
            report.by_category.items(), key=lambda kv: (kv[0] is None, kv[0])
        ):
            print(f"    cat {str(category):>4}  n={metrics.n:<4}  "
                  f"EM={metrics.exact_match:.3f}  F1={metrics.f1:.3f}")

    if n_examples:
        print("-" * 68)
        for scored in report.scored[:n_examples]:
            p = scored.prediction
            print(f"  Q: {p.question}")
            print(f"     gold={p.gold_answer!r}  pred={p.prediction!r}  "
                  f"(EM={scored.exact_match:.0f} F1={scored.f1:.2f})")

    if run_dir is not None:
        print("-" * 68)
        print(f"artifacts   : {run_dir}")
    print("=" * 68)


def main(argv: Sequence[str] | None = None) -> None:
    """Console entry point: ``contextbench [overrides...]``."""
    parser = argparse.ArgumentParser(
        prog="contextbench",
        description="Run a ContextBench experiment from a Hydra config directory.",
    )
    parser.add_argument("--config-dir", default="configs", help="Hydra config directory.")
    parser.add_argument("--config-name", default="config", help="Root config file name.")
    parser.add_argument("--no-log", action="store_true", help="Do not write a run directory.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. budget=512.")
    args = parser.parse_args(argv)

    config = compose_config(args.config_dir, args.overrides, config_name=args.config_name)
    logging.basicConfig(
        level=config.experiment.log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    result, report, run_dir = run_experiment(config, log=not args.no_log)
    print_summary(config, result, report, run_dir)


if __name__ == "__main__":
    main()
