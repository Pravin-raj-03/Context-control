"""Topic-store and QA results restricted to the LongMemEval-S test half.

Segmentation parameters were chosen on dev histories (``dev_test_split``), while
the topic-store and QA tables report every question. This re-scores the same
run outputs on the test half only, so no reported number shares a history with
tuning, and compares ΔContext with each other method by paired bootstrap.

Usage:
    python scripts/test_half_report.py --out paper/results/test_half_lme.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import numpy as np
import yaml

from contextbench.core.config_schema import DatasetConfig
from scripts.paper_tables import METRICS, answerable, load_runs
from scripts.segmentation_report import bootstrap_diff, dev_test_split

OURS = "ΔContext store (top-3)"
STORES = [OURS, "Session oracle store (top-3)", "TextTiling store (top-3)", "Semantic chunking store (top-3)",
          "Adjacent threshold store (top-3)", "Fixed-size chunks (10 turns) store (top-3)", "RAG, turn-level"]
TOPIC_METRICS = ["hit@1", "hit@3", "topic_recall", "searched"]


def summarize(runs, budget, metrics, test_ids):
    out = {}
    for label in STORES:
        records = runs.get((label, budget))
        if not records:
            continue
        rows = [r for r in answerable(records) if r["qa_id"] in test_ids]
        values = {m: [METRICS[m](r) for r in rows] for m in metrics}
        out[label] = {"n": len(rows), **{m: float(np.mean([v for v in vs if v is not None]))
                                         for m, vs in values.items() if any(v is not None for v in vs)}}
        if label != OURS and (OURS, budget) in runs:
            ours = {r["qa_id"]: r for r in answerable(runs[(OURS, budget)]) if r["qa_id"] in test_ids}
            for m in metrics:
                pairs = [(METRICS[m](ours[r["qa_id"]]), METRICS[m](r)) for r in rows if r["qa_id"] in ours]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if pairs and m != "searched":
                    d, lo, hi, p = bootstrap_diff(np.array([a for a, _ in pairs]), np.array([b for _, b in pairs]))
                    out[label][f"ours_minus_this_{m}"] = {"diff": d, "ci": [lo, hi], "p": p}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=2048)
    args = parser.parse_args()

    from contextbench.datasets.longmemeval import LongMemEvalDataset
    config = DatasetConfig(**yaml.safe_load(Path("configs/dataset/longmemeval_s.yaml").read_text()))
    _, test = dev_test_split(list(LongMemEvalDataset(config)))
    test_ids = {s.sample_id for s in test}

    report = {"protocol": __doc__, "test_histories": len(test_ids),
              "topic_store": summarize(load_runs(Path("experiments/lme_topic")), args.budget, TOPIC_METRICS, test_ids),
              "qa": summarize(load_runs(Path("experiments/lme_llm")), args.budget, ["f1", "em"], test_ids)}
    for part in ("topic_store", "qa"):
        print(f"\n=== {part} (test half, answerable)")
        for label, row in report[part].items():
            cells = "  ".join(f"{k} {v:.3f}" for k, v in row.items() if isinstance(v, float))
            print(f"  {label:44s} n={row['n']}  {cells}")
            for k, v in row.items():
                if k.startswith("ours_minus"):
                    print(f"      {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] p={v['p']:.4f}")
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
