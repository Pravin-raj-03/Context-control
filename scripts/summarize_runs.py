"""Tabulate every run under a results directory, one row per (policy setting, budget).

Rows are keyed by the policy's name plus every policy hyper-parameter, so
variants of the same policy stay distinct. Scores are recomputed from each
run's ``outputs.jsonl`` under v2 rules (answerable probes only for F1 and
evidence recall; adversarial probes as abstention rate), split into a dev half
and a test half by conversation. Tune on the dev columns and report the test
columns, so a setting chosen on the data is not also graded on it.

Usage:
    python scripts/summarize_runs.py experiments/v2_selection
    python scripts/summarize_runs.py experiments/v2_llm --metric f1 --dev 5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Hyper-parameters that are not policy choices (device placement, model ids).
_IGNORED_POLICY_KEYS = {"name", "embedder_device", "embedder_model", "embedding_cache"}


def variant_label(policy: dict, varying: set[str]) -> str:
    """Policy name plus the hyper-parameters that differ across its runs."""
    params = [f"{k}={v}" for k, v in sorted(policy.items()) if k in varying]
    return policy["name"] + (f" [{', '.join(params)}]" if params else "")


def varying_keys(policies: list[dict]) -> dict[str, set[str]]:
    """Per policy name, the hyper-parameters whose values differ across runs.

    A key present in only some runs (an override of a default) counts as
    varying, with the missing value shown as absent.
    """
    by_name: dict[str, list[dict]] = defaultdict(list)
    for policy in policies:
        by_name[policy["name"]].append(policy)
    out: dict[str, set[str]] = {}
    for name, group in by_name.items():
        keys = {k for p in group for k in p} - _IGNORED_POLICY_KEYS
        out[name] = {k for k in keys if len({repr(p.get(k)) for p in group}) > 1}
    return out


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(run_dir: Path, dev_ids: set[str], metric: str) -> dict[str, float | None]:
    records = [json.loads(line) for line in
               (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    out: dict[str, float | None] = {}
    for split, keep in (("dev", lambda r: r["sample_id"] in dev_ids),
                        ("test", lambda r: r["sample_id"] not in dev_ids),
                        ("all", lambda r: True)):
        subset = [r for r in records if keep(r)]
        answerable = [r for r in subset if not r.get("adversarial", r["category"] == 5)]
        adversarial = [r for r in subset if r.get("adversarial", r["category"] == 5)]
        out[split] = mean([float(r[metric]) for r in answerable if r.get(metric) is not None])
        out[f"{split}_abstain"] = mean([float(r.get("abstained", False)) for r in adversarial])
    out["context_tokens"] = mean([r["context_tokens"] for r in records])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--metric", default="evidence_recall",
                        choices=["evidence_recall", "f1", "exact_match"])
    parser.add_argument("--dev", type=int, default=5,
                        help="First N conversations (dataset order) form the dev half.")
    args = parser.parse_args()

    runs = sorted(p.parent for p in args.results_dir.glob("*/outputs.jsonl"))
    if not runs:
        raise SystemExit(f"No runs with outputs.jsonl under {args.results_dir}")

    # dataset order of conversations, from the first run
    first = (runs[0] / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
    order = list(dict.fromkeys(json.loads(line)["sample_id"] for line in first))
    dev_ids = set(order[: args.dev])

    configs = {run_dir: json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
               for run_dir in runs}
    varying = varying_keys([c["policy"] for c in configs.values()])

    rows: dict[tuple[str, int], dict] = {}
    for run_dir in runs:
        config = configs[run_dir]
        key = (variant_label(config["policy"], varying[config["policy"]["name"]]),
               config["budget"])
        rows[key] = summarize(run_dir, dev_ids, args.metric)  # latest run wins

    show_abstain = args.metric != "evidence_recall"
    by_budget: dict[int, list] = defaultdict(list)
    for (label, budget), row in rows.items():
        by_budget[budget].append((label, row))

    def fmt(value):
        return "   -  " if value is None else f"{value:.4f}"

    print(f"metric={args.metric} (answerable probes); dev = {sorted(dev_ids)}")
    for budget in sorted(by_budget):
        print(f"\n=== budget {budget} ===")
        header = f"{'policy setting':70s} {'dev':>7s} {'test':>7s} {'all':>7s}"
        if show_abstain:
            header += f" {'adv.abstain':>11s}"
        print(header)
        for label, row in sorted(by_budget[budget], key=lambda lr: -(lr[1]["all"] or 0)):
            line = f"{label[:70]:70s} {fmt(row['dev']):>7s} {fmt(row['test']):>7s} {fmt(row['all']):>7s}"
            if show_abstain:
                line += f" {fmt(row['all_abstain']):>11s}"
            print(line)


if __name__ == "__main__":
    main()
