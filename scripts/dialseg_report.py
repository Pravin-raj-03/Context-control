"""ΔContext vs. baseline segmenters on dialogue topic-segmentation benchmarks.

SuperDialseg, TIAGE and DialSeg711 (HuggingFace ``Coldog2333/*``) annotate
real dialogues with topic boundaries (``segmentation_label = 1`` on the last
turn of each topic), including gradual topic drift. Every method segments each
dialogue from the same MiniLM utterance embeddings; parameters are tuned on a
validation split by mean boundary F1 at tolerance 1 (the LongMemEval protocol;
tuning by Pk picks near-empty segmentations, which Pk rewards) and reported on
test. ΔContext runs through the real policy, online, turn by turn.

Usage:
    python scripts/dialseg_report.py --out paper/results/dialseg.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
os.environ.setdefault("USE_TF", "0")

import numpy as np
import yaml

from contextbench.context import segmenters
from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import Message
from contextbench.evaluation.segmentation import boundary_prf, pk, window_diff
from scripts.segmentation_report import _WordCounter, bootstrap_diff, delta_boundaries

DATA = Path("data/dialseg")
# (name, validation split used for tuning, test split)
BENCHMARKS = [
    ("super_dialseg", ("super_dialseg", "validation"), ("super_dialseg", "test")),
    ("tiage", ("tiage", "validation"), ("tiage", "test")),
    ("dialseg711", ("super_dialseg", "validation"), ("dialseg711", "test")),
]


def load(dataset: str, split: str) -> list[list[dict]]:
    data = json.loads((DATA / dataset / f"{split}.json").read_text(encoding="utf-8"))["dial_data"]
    return [d["turns"] for d in next(iter(data.values())) if len(d["turns"]) >= 2]


class DictEmbedder:
    """Serves precomputed embeddings to the ΔContext policy by exact text."""

    def __init__(self, table: dict[str, np.ndarray]):
        self.table = table

    @property
    def dim(self):
        return len(next(iter(self.table.values())))

    def embed(self, text):
        return self.table[text]

    def embed_batch(self, texts):
        return np.stack([self.table[t] for t in texts])


def text_of(turn: dict) -> str:
    return f"{turn['role']}: {turn['utterance']}"


def reference(turns: list[dict]) -> list[int]:
    return [i for i, t in enumerate(turns[:-1]) if t.get("segmentation_label") == 1]


def score(ref, hyp, n):
    p0, r0, f0 = boundary_prf(ref, hyp, tolerance=0)
    _, _, f1 = boundary_prf(ref, hyp, tolerance=1)
    return {"pk": pk(ref, hyp, n), "windowdiff": window_diff(ref, hyp, n), "f1@0": f0, "f1@1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    splits = {key: load(*key) for b in BENCHMARKS for key in (b[1], b[2])}
    texts = sorted({text_of(t) for dialogues in splits.values() for d in dialogues for t in d})
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=args.device)
    vectors = model.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    table = {t: v.astype(np.float32) for t, v in zip(texts, vectors)}
    print(f"embedded {len(texts)} utterances", flush=True)

    import contextbench.policies.delta_context as dc
    embedder = DictEmbedder(table)
    dc.SentenceTransformerEmbedder = lambda **_: embedder
    base = yaml.safe_load(Path("configs/policy/delta_context.yaml").read_text(encoding="utf-8"))

    def delta_policy(tau, window, gap, reference="checkpoint"):
        return dc.DeltaContextPolicy(PolicyConfig(**{
            **base, "accumulator": "displacement", "change_threshold": tau, "window": window,
            "min_snapshot_gap": gap, "localize_boundaries": True, "reference": reference}),
            _WordCounter())

    grids = {
        "Fixed-size chunks": [dict(size=n) for n in (2, 3, 4, 5, 6, 8)],
        "Adjacent-turn threshold": [dict(theta=t) for t in np.arange(0.3, 0.96, 0.05).round(2)],
        "Semantic chunking": [dict(p=p) for p in (50, 60, 70, 75, 80, 85, 90)],
        "TextTiling": [dict(block=b) for b in (1, 2, 3, 4)],
        "ΔContext (LongMemEval config; τ tuned)": [dict(tau=t, window=3, gap=3) for t in
                                                     np.arange(0.2, 0.95, 0.05).round(2)],
        "ΔContext (τ, w, g tuned)": [dict(tau=t, window=w, gap=g) for t, w, g in itertools.product(
            np.arange(0.2, 0.95, 0.05).round(2), (1, 2, 3), (1, 2, 3))],
        "ΔContext topic-mean (τ, w, g tuned)": [
            dict(tau=t, window=w, gap=g, reference="segment_mean") for t, w, g in itertools.product(
                np.arange(0.2, 0.95, 0.05).round(2), (1, 2, 3), (1, 2, 3))],
    }

    policies: dict[tuple, object] = {}

    def segment(method, params, turns):
        e = np.stack([table[text_of(t)] for t in turns])
        n = len(turns)
        if method == "Fixed-size chunks":
            return segmenters.fixed_boundaries(n, params["size"])
        if method == "Adjacent-turn threshold":
            return segmenters.adjacent_threshold_boundaries(e, params["theta"])
        if method == "Semantic chunking":
            return segmenters.semantic_chunker_boundaries(e, params["p"])
        if method == "TextTiling":
            return segmenters.texttiling_boundaries(e, block=params["block"])
        key = (params["tau"], params["window"], params["gap"], params.get("reference", "checkpoint"))
        if key not in policies:
            policies[key] = delta_policy(*key)
        messages = [Message(role="user" if t["role"] == "user" else "assistant",
                            content=t["utterance"], turn_id=i, speaker=t["role"])
                    for i, t in enumerate(turns)]
        return delta_boundaries(policies[key], messages)

    def evaluate(method, params, dialogues):
        return [score(reference(d), segment(method, params, d), len(d)) for d in dialogues]

    dev_cache: dict = {}

    def dev_f1(dev_key, method, params):
        kind = "delta" if method.startswith("ΔContext") else method
        key = (dev_key, kind, tuple(sorted(params.items())))
        if key not in dev_cache:
            dev_cache[key] = float(np.mean([s["f1@1"] for s in evaluate(method, params, splits[dev_key])]))
        return dev_cache[key]

    report = {}
    for name, dev_key, test_key in BENCHMARKS:
        dev, test = splits[dev_key], splits[test_key]
        print(f"\n=== {name}: tune on {dev_key[0]}/{dev_key[1]} ({len(dev)}), "
              f"test {test_key[0]}/{test_key[1]} ({len(test)})", flush=True)
        chosen, per_dialogue, means = {}, {}, {}
        for method, grid in grids.items():
            best = max(grid, key=lambda prm: dev_f1(dev_key, method, prm))
            chosen[method] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                              for k, v in best.items()}
            per_dialogue[method] = evaluate(method, best, test)
            means[method] = {k: float(np.mean([s[k] for s in per_dialogue[method]]))
                             for k in ("pk", "windowdiff", "f1@0", "f1@1")}
        print(f"{'method':38s} {'Pk':>6s} {'WD':>6s} {'F1@0':>6s} {'F1@1':>6s}  configs  params")
        for method, m in means.items():
            print(f"{method:38s} {m['pk']:6.3f} {m['windowdiff']:6.3f} {m['f1@0']:6.3f} "
                  f"{m['f1@1']:6.3f}  {len(grids[method]):7d}  {chosen[method]}", flush=True)
        tests = {}
        for ours in ("ΔContext (τ, w, g tuned)", "ΔContext topic-mean (τ, w, g tuned)"):
            for method in grids:
                if method.startswith("ΔContext"):
                    continue
                tests[f"{ours} vs {method}"] = {
                    metric: bootstrap_diff(np.array([s[metric] for s in per_dialogue[ours]]),
                                           np.array([s[metric] for s in per_dialogue[method]]))
                    for metric in ("pk", "f1@1")
                }
        for label, metrics in tests.items():
            cells = "  ".join(f"{k}: {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.4f}"
                              for k, (d, lo, hi, p) in metrics.items())
            print(f"  {label:72s} {cells}")
        report[name] = {"dev": f"{dev_key[0]}/{dev_key[1]}", "test": f"{test_key[0]}/{test_key[1]}",
                        "tuning": "max mean boundary F1 (tolerance 1) on dev",
                        "configs": {m: len(g) for m, g in grids.items()},
                        "dialogues": len(test), "chosen": chosen, "test_results": means,
                        "delta_minus_baseline": tests}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
