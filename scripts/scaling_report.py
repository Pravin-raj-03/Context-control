"""How topic-first retrieval scales as the chat store grows.

For a sample of LongMemEval test questions, the question's own history is
appended after ``D`` other questions' histories (distractor topics), growing the
store from ~500 to ~10,000 turns. For each way of storing the history, it
measures, per question:

* topic hit@1 / hit@3 -- is an evidence turn inside the best / top-3 topics;
* evidence rank -- the evidence's best position in the topic-first order;
* comparisons -- query similarities a topic-indexed search computes (segment
  centroids + turns in the top-3 topics), vs. one per turn for a full scan;
* ranking time per query (ms), segment centroids precomputed as a store would.

ΔContext segments online through the real policy's ``observe()``; baselines
use the parameters tuned on the dev half (``paper/results/seg_lme.json``).

Usage:
    python scripts/scaling_report.py --out paper/results/scaling_lme.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
os.environ.setdefault("USE_TF", "0")

import numpy as np
import yaml

from contextbench.context import segmenters
from contextbench.context.state import SentenceTransformerEmbedder
from contextbench.core.config_schema import DatasetConfig, PolicyConfig
from contextbench.core.types import Message
from contextbench.datasets.base import DATASET_REGISTRY
import contextbench.datasets.locomo  # noqa: F401  (registration)
import contextbench.datasets.longmemeval  # noqa: F401  (registration)
from scripts.segmentation_report import _WordCounter, delta_boundaries


def segment_ranking(matrix, boundaries, query):
    """Segments by centroid similarity (centroids precomputed), best first."""
    n = len(matrix)
    seg = np.searchsorted(np.asarray(sorted(boundaries)), np.arange(n), side="left")
    ids = np.unique(seg)
    centroids = np.stack([matrix[seg == i].mean(axis=0) for i in ids])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    start = time.perf_counter()
    order = ids[np.argsort(-(centroids @ query), kind="stable")]
    top3 = order[:3]
    candidates = np.flatnonzero(np.isin(seg, top3))
    ranked = candidates[np.argsort(-(matrix[candidates] @ query), kind="stable")]
    elapsed = time.perf_counter() - start
    return seg, order, ranked, len(ids) + len(candidates), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--distractors", type=int, nargs="+", default=[0, 3, 9, 19])
    parser.add_argument("--dataset", type=Path, default=Path("configs/dataset/longmemeval_s.yaml"))
    parser.add_argument("--cache", default="data/cache/longmemeval_s_minilm.npz")
    parser.add_argument("--params", type=Path, default=Path("paper/results/seg_lme.json"))
    args = parser.parse_args()

    params = json.loads(args.params.read_text())["chosen_parameters"]
    config = DatasetConfig(**yaml.safe_load(args.dataset.read_text()))
    samples = list(DATASET_REGISTRY.create(config.name, config))
    test = samples[len(samples) // 2:]
    rng = np.random.default_rng(0)
    chosen = [test[i] for i in rng.choice(len(test), size=min(args.questions, len(test)), replace=False)
              if test[i].qa_items[0].evidence and not test[i].qa_items[0].meta.get("adversarial")]

    embedder = SentenceTransformerEmbedder(device="cpu", cache_path=args.cache or None)
    import contextbench.policies.delta_context as dc
    dc.SentenceTransformerEmbedder = lambda **_: embedder
    base = yaml.safe_load(Path("configs/policy/delta_context.yaml").read_text())
    policy = dc.DeltaContextPolicy(PolicyConfig(**{**base, "accumulator": "displacement",
                                                   "change_threshold": params["delta_context"],
                                                   "localize_boundaries": True}), _WordCounter())

    results: dict[str, dict[int, dict[str, float]]] = {}
    for d in args.distractors:
        acc: dict[str, list] = {}
        for q_index, sample in enumerate(chosen):
            others = [s for s in samples if s.sample_id != sample.sample_id]
            picks = [others[i] for i in np.random.default_rng(q_index).choice(len(others), d, replace=False)] if d else []
            messages, offset = [], 0
            for part in picks + [sample]:
                for m in part.messages:
                    messages.append(Message(role=m.role, content=m.content, turn_id=len(messages),
                                            speaker=m.speaker, session_id=f"{part.sample_id}/{m.session_id}",
                                            meta=m.meta))
                offset = len(messages) - len(part.messages)
            evidence_ids = set(sample.qa_items[0].evidence)
            evidence = [offset + i for i, m in enumerate(sample.messages) if m.meta["dia_id"] in evidence_ids]
            if not evidence:
                continue
            matrix = embedder.embed_batch([f"{m.speaker or m.role}: {m.content}" for m in messages]).astype(np.float64)
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            query = embedder.embed(sample.qa_items[0].question).astype(np.float64)
            query /= max(np.linalg.norm(query), 1e-12)
            n = len(messages)
            stores = {
                "ΔContext": delta_boundaries(policy, messages),
                "Fixed-size chunks": segmenters.fixed_boundaries(n, int(params["fixed"])),
                "Semantic chunking": segmenters.semantic_chunker_boundaries(matrix, params["semantic_chunker"]),
                "Session oracle": [i for i in range(n - 1) if messages[i].session_id != messages[i + 1].session_id],
            }
            start = time.perf_counter()
            full_order = np.argsort(-(matrix @ query), kind="stable")
            full_time = time.perf_counter() - start
            full_rank = int(min(np.flatnonzero(np.isin(full_order, evidence))))
            acc.setdefault("Full scan (turn-level)", []).append(
                (np.nan, np.nan, full_rank, n, full_time, n))
            for name, bounds in stores.items():
                seg, order, ranked, comparisons, elapsed = segment_ranking(matrix, bounds, query)
                ev_segs = set(seg[evidence].tolist())
                hit1 = float(order[0] in ev_segs)
                hit3 = float(bool(ev_segs & set(order[:3].tolist())))
                hits = np.flatnonzero(np.isin(ranked, evidence))
                rank = int(hits[0]) if len(hits) else n
                acc.setdefault(name, []).append((hit1, hit3, rank, comparisons, elapsed, n))
        results[str(d)] = {
            name: {
                "hit@1": float(np.nanmean([r[0] for r in rows])),
                "hit@3": float(np.nanmean([r[1] for r in rows])),
                "median_evidence_rank": float(np.median([r[2] for r in rows])),
                "comparisons": float(np.mean([r[3] for r in rows])),
                "turns": float(np.mean([r[5] for r in rows])),
                "ms_per_query": 1000 * float(np.mean([r[4] for r in rows])),
                "questions": len(rows),
            } for name, rows in acc.items()
        }
        print(f"\n== store = own history + {d} distractor histories "
              f"(~{results[str(d)]['ΔContext']['turns']:.0f} turns)", flush=True)
        for name, m in results[str(d)].items():
            print(f"  {name:24s} hit@1 {m['hit@1']:.3f}  hit@3 {m['hit@3']:.3f}  "
                  f"evidence rank {m['median_evidence_rank']:6.0f}  comparisons {m['comparisons']:7.0f}  "
                  f"{m['ms_per_query']:.2f} ms", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"parameters": params, "results": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
