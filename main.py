"""Single-conversation debug runner.

A fast way to sanity-check the full pipeline on one conversation without editing
config files::

    python main.py                        # 1 conversation, 3 QA, budget 1024
    python main.py --budget 512 --max-qa 5
    python main.py model=qwen2.5-3b       # extra Hydra overrides pass through

For full sweeps and logging, use ``python benchmark.py`` instead.
"""

from __future__ import annotations

import argparse
import logging
import os

# Prevent TensorFlow and JAX from loading and taking over GPU memory,
# which causes silent crashes when SentenceTransformers is initialized.
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from contextbench.cli import compose_config, print_summary, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-conversation debug benchmark.")
    parser.add_argument("--budget", type=int, default=1024, help="Context token budget.")
    parser.add_argument("--num-samples", type=int, default=1, help="Conversations to run.")
    parser.add_argument("--max-qa", type=int, default=3, help="QA probes per conversation.")
    parser.add_argument("--config-dir", default="configs", help="Hydra config directory.")
    parser.add_argument("--no-log", action="store_true", help="Do not write a run directory.")
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides, e.g. model=qwen2.5-3b.")
    args = parser.parse_args()

    overrides = list(args.overrides) + [
        f"budget={args.budget}",
        f"dataset.num_samples={args.num_samples}",
        f"dataset.max_qa_per_sample={args.max_qa}",
    ]
    config = compose_config(args.config_dir, overrides)
    logging.basicConfig(
        level=config.experiment.log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    result, report, run_dir = run_experiment(config, log=not args.no_log)
    print_summary(config, result, report, run_dir)


if __name__ == "__main__":
    main()
