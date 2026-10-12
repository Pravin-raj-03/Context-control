"""Same selection rule for every method: one setting per method, chosen on all dev sets.

``segbench.py`` gives only ΔContext v3 a "universal" setting (best mean dev J over
the LongMemEval, SuperDialseg and TIAGE dev splits). This script applies that rule
to every baseline too, scores each universal setting once on the four test sets,
and compares ΔContext v3 (universal) against each with a paired bootstrap, Holm-
corrected within each dataset. DialSeg711 is unseen by every method.

Usage:
    python scripts/segbench_universal.py --out paper/results/segbench_universal.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
os.environ.setdefault("USE_TF", "0")

import numpy as np
import yaml

from contextbench.context import segmenters
from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import Message
from contextbench.evaluation.segmentation import boundary_prf, pk, window_diff
from scripts.paper_tables import holm
from scripts.segbench import dialogues, grids, longmemeval
from scripts.segmentation_report import _WordCounter, bootstrap_diff, delta_boundaries

OURS = "ΔContext v3 (universal)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bench = json.loads(Path("paper/results/segbench.json").read_text(encoding="utf-8"))
    lme_dev, lme_test = longmemeval(100)
    sd_dev = random.Random(2).sample(dialogues("super_dialseg", "validation"), 500)
    sets = {"LongMemEval-S": (lme_dev, lme_test),
            "SuperDialseg": (sd_dev, dialogues("super_dialseg", "test")),
            "TIAGE": (dialogues("tiage", "validation"), dialogues("tiage", "test")),
            "DialSeg711": ([], dialogues("dialseg711", "test"))}
    tuned_sets = ["LongMemEval-S", "SuperDialseg", "TIAGE"]

    from contextbench.context.state import SentenceTransformerEmbedder, text_key
    cache = SentenceTransformerEmbedder(device="cpu", cache_path="data/cache/longmemeval_s_minilm.npz")
    texts = sorted({f"{r}: {u}" for dev, test in sets.values() for convs in (dev, test)
                    for turns, _ in convs for r, u in turns})
    missing = [t for t in texts if text_key(t) not in cache._cache]
    from sentence_transformers import SentenceTransformer
    fresh = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode(
        missing, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    table = {t: cache._cache[text_key(t)] for t in texts if text_key(t) in cache._cache}
    table.update({t: v.astype(np.float32) for t, v in zip(missing, fresh)})
    del cache

    class TableEmbedder:
        dim = 384

        def embed(self, text):
            return table[text]

        def embed_batch(self, ts):
            return np.stack([table[t] for t in ts])

    import contextbench.policies.delta_context as dc
    embedder = TableEmbedder()
    dc.SentenceTransformerEmbedder = lambda **_: embedder
    base = yaml.safe_load(Path("configs/policy/delta_context.yaml").read_text(encoding="utf-8"))
    policies = {}

    def segment(method, params, turns):
        e = np.stack([table[f"{r}: {u}"] for r, u in turns])
        n = len(turns)
        if method == "Fixed-size chunks":
            return segmenters.fixed_boundaries(n, params["size"])
        if method == "Adjacent-turn threshold":
            return segmenters.adjacent_threshold_boundaries(e, params["theta"])
        if method == "Semantic chunking":
            return segmenters.semantic_chunker_boundaries(e, params["p"])
        if method == "TextTiling":
            return segmenters.texttiling_boundaries(e, block=params["block"], cutoff_std=params["cutoff_std"])
        key = tuple(sorted(params.items()))
        if key not in policies:
            policies[key] = dc.DeltaContextPolicy(PolicyConfig(**{**base, **params}), _WordCounter())
        msgs = [Message(role="user", content=u, turn_id=i, speaker=r) for i, (r, u) in enumerate(turns)]
        return delta_boundaries(policies[key], msgs)

    def scores(method, params, convs):
        out = []
        for turns, ref in convs:
            hyp = segment(method, params, turns)
            n = len(turns)
            f0 = boundary_prf(ref, hyp, tolerance=0)[2]
            p_k = pk(ref, hyp, n)
            out.append({"pk": p_k, "windowdiff": window_diff(ref, hyp, n), "f1@0": f0,
                        "f1@1": boundary_prf(ref, hyp, tolerance=1)[2], "J": ((1 - p_k) + f0) / 2})
        return out

    baselines = {m: g for m, g in grids().items() if not m.startswith("ΔContext v3")}
    universal = {}
    for method, grid in baselines.items():
        start = time.perf_counter()
        dev_j = {tuple(sorted(p.items())): np.mean([np.mean([s["J"] for s in scores(method, p, sets[n][0])])
                                                    for n in tuned_sets]) for p in grid}
        best = max(grid, key=lambda p: dev_j[tuple(sorted(p.items()))])
        universal[method] = best
        print(f"universal {method:28s} mean dev J {dev_j[tuple(sorted(best.items()))]:.3f}  {best}"
              f"  ({time.perf_counter() - start:.0f}s)", flush=True)
    universal[OURS] = bench["universal_delta_v3"]

    report = {"protocol": __doc__, "universal": universal, "datasets": {}}
    for name, (_, test) in sets.items():
        per = {m: scores("ΔContext v3" if m == OURS else m, p, test) for m, p in universal.items()}
        means = {m: {k: float(np.mean([s[k] for s in v])) for k in ("pk", "windowdiff", "f1@0", "f1@1", "J")}
                 for m, v in per.items()}
        tests, raw = {}, {}
        for b in baselines:
            for metric in ("pk", "f1@0", "J"):
                d, lo, hi, p = bootstrap_diff(np.array([s[metric] for s in per[OURS]]),
                                              np.array([s[metric] for s in per[b]]))
                tests[(b, metric)] = (d, lo, hi)
                raw[(b, metric)] = p
        adjusted = holm(raw)
        print(f"\n=== {name} ({len(test)} test conversations)")
        for m, v in sorted(means.items(), key=lambda kv: -kv[1]["J"]):
            print(f"  {m:30s} Pk {v['pk']:.3f}  WD {v['windowdiff']:.3f}  F1@0 {v['f1@0']:.3f}  J {v['J']:.3f}")
        for (b, metric), (d, lo, hi) in tests.items():
            print(f"    ours - {b} [{metric}]: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] p_holm={adjusted[(b, metric)]:.4f}")
        report["datasets"][name] = {
            "test_conversations": len(test), "test": means,
            "tests": {f"{OURS} vs {b} [{mt}]": {"diff": tests[(b, mt)][0], "ci": tests[(b, mt)][1:],
                                                "p_holm": adjusted[(b, mt)]} for (b, mt) in tests}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
