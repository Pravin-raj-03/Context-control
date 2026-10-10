"""Topic-segmentation quality of ΔContext vs. baseline segmenters.

The reference segmentation is the dataset's session boundaries (in LongMemEval
each session is a separate topic). Every method has one parameter; each is
tuned on the dev half of the histories by mean boundary F1 (tolerance 1 turn)
and then scored on the test half with Pk, WindowDiff and boundary P/R/F1.
ΔContext is run through the real policy's ``observe()`` (online, turn by turn);
the baselines see the whole history at once except ``adjacent`` and ``fixed``.

Writes a JSON summary next to printing a table, for the paper tables.

Usage:
    python scripts/segmentation_report.py configs/dataset/longmemeval_s.yaml \\
        --cache data/cache/longmemeval_s_minilm.npz --out paper/results/seg_lme.json
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
from contextbench.datasets.base import DATASET_REGISTRY
import contextbench.datasets.locomo  # noqa: F401  (registration)
import contextbench.datasets.longmemeval  # noqa: F401  (registration)
from contextbench.evaluation.segmentation import boundary_prf, pk, window_diff

GRIDS = {
    "fixed": [5, 8, 10, 12, 15, 20],
    "adjacent": [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
    "semantic_chunker": [70, 80, 85, 90, 93, 95, 97],
    "texttiling": [2, 3, 4, 5],
    "delta_context": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
}
LABELS = {
    "fixed": "Fixed-size chunks",
    "adjacent": "Adjacent-turn threshold (online)",
    "semantic_chunker": "Semantic chunking (offline)",
    "texttiling": "TextTiling (offline)",
    "delta_context": "ΔContext (online)",
    "delta_context_unlocalized": "ΔContext w/o localization",
    "delta_context_sum": "ΔContext, path-length detector",
}


class _WordCounter:
    """Token counts do not affect segmentation; avoid the slow real tokenizer."""

    def count_text(self, text):
        return len(text.split())

    def count_message(self, message):
        return len(message.content.split()) + 1

    def count_messages(self, messages):
        return sum(self.count_message(m) for m in messages)


def delta_boundaries(policy, messages) -> list[int]:
    policy.reset()
    for m in messages:
        policy.observe(m)
    return sorted(c.meta["segment_start"] - 1 for c in policy._checkpoints.all()
                  if c.meta["segment_start"] - 1 < len(messages) - 1)


def score(reference, hypothesis, n) -> dict[str, float]:
    p0, r0, f0 = boundary_prf(reference, hypothesis, tolerance=0)
    p1, r1, f1 = boundary_prf(reference, hypothesis, tolerance=1)
    return {"pk": pk(reference, hypothesis, n), "windowdiff": window_diff(reference, hypothesis, n),
            "p@0": p0, "r@0": r0, "f1@0": f0, "p@1": p1, "r@1": r1, "f1@1": f1,
            "segments": len(hypothesis) + 1}


def bootstrap_diff(a: np.ndarray, b: np.ndarray, resamples: int = 10_000) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(0)
    diff = a - b
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return float(diff.mean()), float(lo), float(hi), float(p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset_config", type=Path)
    parser.add_argument("--cache", default=None, help="Precomputed embedding cache (.npz).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.5)
    args = parser.parse_args()

    config = DatasetConfig(**yaml.safe_load(args.dataset_config.read_text(encoding="utf-8")))
    samples = list(DATASET_REGISTRY.create(config.name, config))
    split = int(len(samples) * args.dev_fraction)
    dev, test = samples[:split], samples[split:]

    embedder = SentenceTransformerEmbedder(device="cpu", cache_path=args.cache)
    embeddings = {s.sample_id: embedder.embed_batch([f"{m.speaker or m.role}: {m.content}"
                                                     for m in s.messages]) for s in samples}
    reference = {s.sample_id: [i for i in range(len(s.messages) - 1)
                               if s.messages[i].session_id != s.messages[i + 1].session_id]
                 for s in samples}

    import contextbench.policies.delta_context as dc
    dc.SentenceTransformerEmbedder = lambda **_: embedder  # share the loaded embedder/cache
    base = yaml.safe_load(Path("configs/policy/delta_context.yaml").read_text(encoding="utf-8"))

    def make_delta(threshold, **extra):
        cfg = {**base, "accumulator": "displacement", "change_threshold": threshold,
               "localize_boundaries": True, **extra}
        return dc.DeltaContextPolicy(PolicyConfig(**cfg), _WordCounter())

    def segment(method, param, sample):
        e, n = embeddings[sample.sample_id], len(sample.messages)
        if method == "fixed":
            return segmenters.fixed_boundaries(n, int(param))
        if method == "adjacent":
            return segmenters.adjacent_threshold_boundaries(e, param)
        if method == "semantic_chunker":
            return segmenters.semantic_chunker_boundaries(e, param)
        if method == "texttiling":
            return segmenters.texttiling_boundaries(e, block=int(param))
        return delta_boundaries(param, sample.messages)  # param is a policy

    def evaluate(method, param, subset):
        return [score(reference[s.sample_id], segment(method, param, s), len(s.messages))
                for s in subset]

    chosen, results = {}, {}
    for method, grid in GRIDS.items():
        best = None
        for value in grid:
            param = make_delta(value) if method == "delta_context" else value
            f1 = float(np.mean([r["f1@1"] for r in evaluate(method, param, dev)]))
            print(f"  dev {method:18s} {value!s:>6s}  F1@1={f1:.3f}", flush=True)
            if best is None or f1 > best[1]:
                best = (value, f1)
        chosen[method] = best[0]

    runs = {m: (make_delta(chosen[m]) if m == "delta_context" else chosen[m]) for m in GRIDS}
    runs["delta_context_unlocalized"] = make_delta(chosen["delta_context"], localize_boundaries=False)
    runs["delta_context_sum"] = dc.DeltaContextPolicy(PolicyConfig(**base), _WordCounter())
    per_history = {}
    for method, param in runs.items():
        kind = "delta_context" if method.startswith("delta_context") else method
        per_history[method] = evaluate(kind, param, test)
        results[method] = {k: float(np.mean([r[k] for r in per_history[method]]))
                           for k in per_history[method][0]}

    significance = {}
    ours = per_history["delta_context"]
    for method in runs:
        if method == "delta_context":
            continue
        significance[method] = {
            metric: bootstrap_diff(np.array([r[metric] for r in ours]),
                                   np.array([r[metric] for r in per_history[method]]))
            for metric in ("f1@1", "pk", "windowdiff")
        }

    ref_segments = float(np.mean([len(reference[s.sample_id]) + 1 for s in test]))
    print(f"\nTest histories: {len(test)}; reference segments per history: {ref_segments:.1f}")
    print(f"{'method':36s} {'param':>6s} {'Pk':>6s} {'WD':>6s} {'P@1':>6s} {'R@1':>6s} {'F1@1':>6s} {'F1@0':>6s} {'segs':>6s}")
    for method, m in results.items():
        param = chosen.get(method, chosen["delta_context"] if method == "delta_context_unlocalized" else base["change_threshold"])
        print(f"{LABELS[method]:36s} {param!s:>6s} {m['pk']:6.3f} {m['windowdiff']:6.3f} {m['p@1']:6.3f} "
              f"{m['r@1']:6.3f} {m['f1@1']:6.3f} {m['f1@0']:6.3f} {m['segments']:6.1f}")
    print("\nΔContext minus baseline on test histories (mean, 95% CI, p):")
    for method, metrics in significance.items():
        cells = "  ".join(f"{k}: {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.4f}"
                          for k, (d, lo, hi, p) in metrics.items())
        print(f"  vs {LABELS[method]:34s} {cells}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "dataset": str(args.dataset_config), "dev_histories": len(dev), "test_histories": len(test),
        "reference_segments_per_history": ref_segments, "chosen_parameters": chosen,
        "test_results": results, "delta_minus_baseline": significance,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
