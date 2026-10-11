"""Streaming segmentation: what each method does to a live, growing chat store.

In a live chat, turns arrive one at a time and the store must stay segmented
as it grows. Online methods (ΔContext, adjacent threshold, fixed chunks) only
ever append a boundary; offline methods (semantic chunking, TextTiling) must be
re-run on the growing history, and their earlier boundaries can move. For each
LongMemEval test history, every method segments the prefix of the first t
turns for t = step, 2*step, ..., n, and we measure:

* removals -- boundaries placed earlier that later disappear or move, per 100
  turns (0 for an append-only store);
* late cuts -- boundaries later inserted more than ``HORIZON`` turns back into
  history that was already segmented, per 100 turns (re-cutting old topics,
  as opposed to the few-turn look-back any detector needs);
* topics re-indexed per new turn -- segments whose membership changes between
  consecutive prefixes (each needs its centroid recomputed); 1 for an
  append-only store (only the open segment grows), more when old segments are
  re-cut;
* detection delay -- turns between a true session boundary and the first
  prefix in which a boundary within one turn of it appears and then persists;
* final quality -- boundary F1 (+-1) and Pk of the last prefix.

Parameters: the dev-tuned values in ``paper/results/seg_lme.json``.

Usage:
    python scripts/streaming_report.py --out paper/results/streaming_lme.json
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
import yaml

from contextbench.context import segmenters
from contextbench.context.state import SentenceTransformerEmbedder
from contextbench.core.config_schema import DatasetConfig, PolicyConfig
from contextbench.datasets.longmemeval import LongMemEvalDataset
from contextbench.evaluation.segmentation import boundary_prf, pk
from scripts.segmentation_report import _WordCounter, delta_boundaries, dev_test_split

HORIZON = 5


def membership(boundaries: set[int], n: int) -> list[tuple[int, int]]:
    """Segments as (first, last) turn spans over n turns."""
    spans, start = [], 0
    for b in sorted(x for x in boundaries if x < n - 1):
        spans.append((start, b))
        start = b + 1
    spans.append((start, n - 1))
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--histories", type=int, default=250)
    args = parser.parse_args()

    params = json.loads(Path("paper/results/seg_lme.json").read_text())["chosen_parameters"]
    config = DatasetConfig(**yaml.safe_load(Path("configs/dataset/longmemeval_s.yaml").read_text()))
    samples = list(LongMemEvalDataset(config))
    test = dev_test_split(samples)[1][: args.histories]

    embedder = SentenceTransformerEmbedder(device="cpu", cache_path="data/cache/longmemeval_s_minilm.npz")
    import contextbench.policies.delta_context as dc
    dc.SentenceTransformerEmbedder = lambda **_: embedder
    base = yaml.safe_load(Path("configs/policy/delta_context.yaml").read_text())
    policy = dc.DeltaContextPolicy(PolicyConfig(**{**base, "accumulator": "displacement",
                                                   "change_threshold": params["delta_context"],
                                                   "localize_boundaries": True}), _WordCounter())

    methods = {
        "ΔContext (online)": None,
        "Adjacent threshold (online)": lambda e: segmenters.adjacent_threshold_boundaries(e, params["adjacent"]),
        "Fixed-size chunks (online)": lambda e: segmenters.fixed_boundaries(len(e), int(params["fixed"])),
        "Semantic chunking (re-run)": lambda e: segmenters.semantic_chunker_boundaries(e, params["semantic_chunker"]),
        "TextTiling (re-run)": lambda e: segmenters.texttiling_boundaries(e, int(params["texttiling"])),
    }
    stats = {m: {"removed": 0, "late": 0, "turns": 0, "reindexed": 0, "steps_turns": 0,
                 "delays": [], "f1": [], "pk": []} for m in methods}

    for h, sample in enumerate(test):
        messages = list(sample.messages)
        n = len(messages)
        e = embedder.embed_batch([f"{m.speaker or m.role}: {m.content}" for m in messages])
        truth = [i for i in range(n - 1) if messages[i].session_id != messages[i + 1].session_id]
        # ΔContext's online run: boundaries emitted by turn t are exactly those
        # with their checkpoint at or before t, and never change afterwards.
        policy.reset()
        emitted_at = []
        for t, m in enumerate(messages):
            before = len(policy._checkpoints.all())
            policy.observe(m)
            if len(policy._checkpoints.all()) > before:
                start = policy._checkpoints.all()[-1].meta["segment_start"]
                emitted_at.append((start - 1, t))
        prefixes = list(range(args.step, n, args.step)) + [n]
        for name, fn in methods.items():
            history: list[set[int]] = []
            for t in prefixes:
                if fn is None:
                    bounds = {b for b, at in emitted_at if at < t and b < t - 1}
                else:
                    bounds = {b for b in fn(e[:t]) if b < t - 1}
                history.append(bounds)
            s = stats[name]
            for (t0, b0), (t1, b1) in zip(zip(prefixes, history), zip(prefixes[1:], history[1:])):
                s["removed"] += len(b0 - b1)
                s["late"] += len({b for b in b1 - b0 if b < t0 - 1 - HORIZON})
                spans0 = set(membership(b0, t0))
                spans1 = set(membership(b1, t1))
                s["reindexed"] += len([sp for sp in spans1 if sp not in spans0])
                s["steps_turns"] += t1 - t0
            s["turns"] += n
            final = history[-1]
            for b in truth:
                present = [any(abs(x - b) <= 1 for x in bounds) for bounds in history]
                if not present[-1]:
                    continue
                last_absent = max((i for i, ok in enumerate(present) if not ok), default=-1)
                s["delays"].append(prefixes[last_absent + 1] - (b + 1))
            s["f1"].append(boundary_prf(truth, sorted(final), tolerance=1)[2])
            s["pk"].append(pk(truth, sorted(final), n))
        if (h + 1) % 50 == 0:
            print(f"  {h + 1}/{len(test)} histories", flush=True)

    results = {}
    print(f"\n{len(test)} test histories, prefixes every {args.step} turns")
    print(f"{'method':30s} {'removed/100t':>12s} {'late cuts/100t':>14s} {'re-index/turn':>13s} "
          f"{'delay med':>9s} {'F1@1':>6s} {'Pk':>6s}")
    for name, s in stats.items():
        results[name] = {
            "removed_boundaries_per_100_turns": 100 * s["removed"] / s["turns"],
            "late_cuts_per_100_turns": 100 * s["late"] / s["turns"],
            "segments_reindexed_per_new_turn": s["reindexed"] / s["steps_turns"],
            "median_detection_delay_turns": float(np.median(s["delays"])) if s["delays"] else None,
            "mean_detection_delay_turns": float(np.mean(s["delays"])) if s["delays"] else None,
            "final_f1@1": float(np.mean(s["f1"])), "final_pk": float(np.mean(s["pk"])),
        }
        r = results[name]
        print(f"{name:30s} {r['removed_boundaries_per_100_turns']:12.2f} {r['late_cuts_per_100_turns']:14.2f} "
              f"{r['segments_reindexed_per_new_turn']:13.3f} {r['median_detection_delay_turns']!s:>9s} "
              f"{r['final_f1@1']:6.3f} {r['final_pk']:6.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"step": args.step, "histories": len(test), "parameters": params,
                                    "results": results}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
