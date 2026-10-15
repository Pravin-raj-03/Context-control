"""Significance tests and dataset statistics for the conversational-memory paper.

* recall: ΔContext-R vs every method in the test lab results, per budget (paired
  bootstrap over questions, Holm within budget);
* QA F1 and EM: ΔContext-R vs every QA run with the same budget (Holm within
  budget and metric);
* ``--units``: memory units per test history and mean Qwen tokens per unit.

Usage:
    python scripts/memory_paper_stats.py [--units]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
os.environ.setdefault("USE_TF", "0")

import numpy as np

from scripts.memory_lab import slug
from scripts.paper_figures_memory import load_lab, load_qa
from scripts.paper_tables import holm
from scripts.segmentation_report import bootstrap_diff

OURS = "ours+user | boost=0.5 | k=3 | reserve=0.25"


def recall_tests(test):
    out = {}
    for b in (2048, 4096):
        rows = {k.rsplit(" @ ", 1)[0]: v for k, v in test.items() if k.endswith(f" @ {b}")}
        a = np.array(rows[OURS]["per_question"])
        diffs, raw = {}, {}
        for name, row in rows.items():
            if name != OURS:
                d, lo, hi, p = bootstrap_diff(a, np.array(row["per_question"]))
                diffs[name], raw[name] = [d, lo, hi], p
        adj = holm(raw)
        out[b] = {n: {"diff": diffs[n], "p_holm": adj[n]} for n in diffs}
    return out


def qa_tests(runs):
    out = {}
    for b in (2048, 4096):
        ours = runs.get(slug(f"{OURS} @ {b}"))
        if ours is None:
            continue
        ours = {r["qa_id"]: r for r in ours}
        for metric in ("f1", "exact_match"):
            diffs, raw = {}, {}
            for name, recs in runs.items():
                if not name.endswith(f"_{b}") or name == slug(f"{OURS} @ {b}"):
                    continue
                other = {r["qa_id"]: r for r in recs}
                ids = sorted(set(other) & set(ours))
                d, lo, hi, p = bootstrap_diff(np.array([ours[i][metric] for i in ids]),
                                              np.array([other[i][metric] for i in ids]))
                diffs[name], raw[name] = [d, lo, hi], p
            adj = holm(raw)
            out[f"{b}:{metric}"] = {n: {"diff": diffs[n], "p_holm": adj[n]} for n in diffs}
    return out


def unit_stats():
    import yaml
    from contextbench.chatbot.formatting import TranscriptFormatter
    from contextbench.context.state import SentenceTransformerEmbedder
    from contextbench.core.config_schema import DatasetConfig
    from contextbench.datasets.longmemeval import LongMemEvalDataset
    from scripts.memory_lab import History, load_compression
    from scripts.segmentation_report import _WordCounter, delta_policy_from_params
    from transformers import AutoTokenizer

    cfg = yaml.safe_load(Path("configs/dataset/longmemeval_s.yaml").read_text())
    cfg["sample_ids_file"] = "paper/results/lme_test_answerable_ids.json"
    samples = list(LongMemEvalDataset(DatasetConfig(**cfg)))
    minilm = SentenceTransformerEmbedder(device="cpu", cache_path="data/cache/longmemeval_s_minilm.npz")
    import contextbench.policies.delta_context as dc
    dc.SentenceTransformerEmbedder = lambda **_: minilm
    policy = delta_policy_from_params(json.loads(Path("paper/results/params_v3_lme.json").read_text()),
                                      _WordCounter())
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    fmt = TranscriptFormatter(include_dates=True)
    acc = {k: {"units": [], "tokens": []} for k in ("turn", "user", "assistant", "exchange", "segment",
                                                    "segsess", "session")}
    for s in samples:
        h = History(s, policy, tok, fmt, {})
        roles = np.array([m.role for m in h.msgs])
        acc["turn"]["units"].append(h.n)
        acc["turn"]["tokens"].extend(h.cost.tolist())
        for r in ("user", "assistant"):
            acc[r]["units"].append(int((roles == r).sum()))
            acc[r]["tokens"].extend(h.cost[roles == r].tolist())
        for kind in ("exchange", "segment", "segsess", "session"):
            acc[kind]["units"].append(len(h.units[kind]))
            acc[kind]["tokens"].extend(int(h.cost[u].sum()) for u in h.units[kind])
    return {k: {"units_per_history": float(np.mean(v["units"])), "tokens_per_unit": float(np.mean(v["tokens"]))}
            for k, v in acc.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--units", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("paper/results/memory_paper_stats.json"))
    args = parser.parse_args()
    test = load_lab("paper/results/memory_lab_test.json", "paper/results/memory_lab_test2.json")
    report = {"recall": recall_tests(test), "qa": qa_tests(load_qa())}
    if args.units:
        report["units"] = unit_stats()
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for b, rows in report["recall"].items():
        print(f"recall @ {b}")
        for n, r in sorted(rows.items(), key=lambda kv: kv[1]["diff"][0]):
            print(f"  {n:60s} {r['diff'][0]:+.3f} [{r['diff'][1]:+.3f}, {r['diff'][2]:+.3f}] p={r['p_holm']:.4f}")
    for k, rows in report["qa"].items():
        print(f"QA {k}")
        for n, r in sorted(rows.items(), key=lambda kv: kv[1]["diff"][0]):
            print(f"  {n:60s} {r['diff'][0]:+.3f} [{r['diff'][1]:+.3f}, {r['diff'][2]:+.3f}] p={r['p_holm']:.4f}")
    if "units" in report:
        for k, v in report["units"].items():
            print(f"  {k:10s} {v['units_per_history']:7.1f} units/history  {v['tokens_per_unit']:7.1f} tokens/unit")


if __name__ == "__main__":
    main()
