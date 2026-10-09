"""Paired significance test between two benchmark runs.

Every policy answers the same probes under the same model and seed, so two runs
form a paired design. This script matches probes by ``qa_id``, bootstraps the
mean per-probe difference of a metric (10,000 resamples by default), and
reports the 95% CI, a two-sided bootstrap p-value, and wins/losses/ties.

It reads only ``outputs.jsonl``, so it also re-scores runs made under the v1
evaluation: with ``--scoring v2`` (the default) adversarial probes (category 5)
are dropped, matching the ``locomo_qa`` evaluator's headline numbers. That
applies to evidence recall too: an adversarial probe's evidence is the trap,
so recalling it is not a quality signal.
Runs whose outputs predate the ``adversarial`` field fall back to category 5.

Usage:
    python scripts/compare_runs.py RUN_A RUN_B
    python scripts/compare_runs.py RUN_A RUN_B --metric evidence_recall
    python scripts/compare_runs.py RUN_A RUN_B --metric f1 --scoring v1

RUN_A / RUN_B are run directories (containing ``outputs.jsonl``). The reported
difference is A - B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_outputs(run_dir: Path) -> dict[str, dict]:
    """Load a run's per-probe records keyed by ``qa_id``."""
    with (run_dir / "outputs.jsonl").open(encoding="utf-8") as handle:
        return {record["qa_id"]: record for record in map(json.loads, handle)}


def is_adversarial(record: dict) -> bool:
    return bool(record.get("adversarial", record.get("category") == 5))


def paired_scores(
    a: dict[str, dict], b: dict[str, dict], metric: str, scoring: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned per-probe scores for the probes both runs scored."""
    shared = sorted(a.keys() & b.keys())
    xs, ys = [], []
    for qa_id in shared:
        ra, rb = a[qa_id], b[qa_id]
        if scoring == "v2" and is_adversarial(ra):
            continue
        va, vb = ra.get(metric), rb.get(metric)
        if va is None or vb is None or va == "" or vb == "":
            continue
        xs.append(float(va))
        ys.append(float(vb))
    return np.asarray(xs), np.asarray(ys)


def bootstrap(diff: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    """95% bootstrap CI of the mean difference and a two-sided p-value."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    # two-sided: how often the resampled mean lands on the other side of zero
    tail = min((means <= 0).mean(), (means >= 0).mean())
    return float(low), float(high), float(min(1.0, 2 * tail))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--metric", default="f1",
                        choices=["f1", "exact_match", "evidence_recall"])
    parser.add_argument("--scoring", default="v2", choices=["v1", "v2"])
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    a, b = load_outputs(args.run_a), load_outputs(args.run_b)
    xs, ys = paired_scores(a, b, args.metric, args.scoring)
    if len(xs) == 0:
        raise SystemExit(f"No paired probes with a '{args.metric}' value in both runs.")
    diff = xs - ys
    low, high, p = bootstrap(diff, args.resamples, args.seed)

    print(f"A: {args.run_a.name}")
    print(f"B: {args.run_b.name}")
    print(f"metric={args.metric} scoring={args.scoring} paired probes={len(diff)}")
    print(f"mean A={xs.mean():.4f}  mean B={ys.mean():.4f}")
    print(f"A - B = {diff.mean():+.4f}  95% CI [{low:+.4f}, {high:+.4f}]  p={p:.4f}")
    print(f"wins/losses/ties (A vs B): "
          f"{(diff > 0).sum()}/{(diff < 0).sum()}/{(diff == 0).sum()}")


if __name__ == "__main__":
    main()
