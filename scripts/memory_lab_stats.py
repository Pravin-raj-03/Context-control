"""Paired significance for a memory-lab result file: ours vs every other method.

Reads ``scripts/memory_lab.py --out`` output (per-question evidence recall, same
question order for every method), and for each budget compares the method whose
name starts with ``--ours`` against every other method with a paired bootstrap
(10,000 resamples), Holm-corrected within the budget.

Usage:
    python scripts/memory_lab_stats.py paper/results/memory_lab_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import numpy as np

from scripts.paper_tables import holm
from scripts.segmentation_report import bootstrap_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--ours", default="ours |")
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    results = data["results"]
    for b in data["budgets"]:
        rows = {k.rsplit(" @ ", 1)[0]: v for k, v in results.items() if k.endswith(f" @ {b}")}
        ours = next(k for k in rows if k.startswith(args.ours))
        a = np.array(rows[ours]["per_question"])
        tests, raw = {}, {}
        for name, row in rows.items():
            if name == ours:
                continue
            d, lo, hi, p = bootstrap_diff(a, np.array(row["per_question"]))
            tests[name], raw[name] = (d, lo, hi), p
        adj = holm(raw)
        print(f"\n=== budget {b}, {data['questions']} questions; ours = {ours}: recall {np.mean(a):.3f}")
        for name in sorted(tests, key=lambda n: -rows[n]["recall"]):
            d, lo, hi = tests[name]
            print(f"  vs {name:60s} {rows[name]['recall']:.3f}  diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"p_holm={adj[name]:.4f}")


if __name__ == "__main__":
    main()
