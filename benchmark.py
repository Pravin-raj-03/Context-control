"""Hydra-driven benchmark entry point.

Runs a full benchmark (all conversations) and writes a run directory. Supports
Hydra composition and multirun sweeps::

    python benchmark.py                                  # defaults
    python benchmark.py budget=512 model=qwen2.5-3b
    python benchmark.py --multirun budget=512,1024,2048  # sweep budgets

Each invocation validates the composed config, runs the engine, scores, and
logs. For a quick single-conversation smoke run, use ``python main.py``.
"""

from __future__ import annotations

import logging
import os

# Prevent TensorFlow and JAX from loading and taking over GPU memory,
# which causes silent crashes when SentenceTransformers is initialized.
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import hydra
from omegaconf import DictConfig, OmegaConf

from contextbench.cli import print_summary, run_experiment
from contextbench.core.config_schema import BenchmarkConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Validate the composed config and run one experiment."""
    config = BenchmarkConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
    logging.basicConfig(
        level=config.experiment.log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    result, report, run_dir = run_experiment(config, log=True)
    print_summary(config, result, report, run_dir)


if __name__ == "__main__":
    main()
