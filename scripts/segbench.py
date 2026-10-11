"""One topic-segmentation benchmark over four datasets, one protocol.

Datasets (reference boundaries in brackets):
* LongMemEval-S -- long assistant chat histories [session boundaries];
* SuperDialseg, TIAGE, DialSeg711 -- annotated dialogues [topic labels].

Protocol, fixed before running:
* every method segments from the same MiniLM utterance embeddings;
* every parameter grid is searched on a dev split only, maximising
  J = ((1 - Pk) + F1_exact) / 2, which guards against the degenerate solutions
  each metric alone rewards (Pk: near-empty segmentations; lenient F1:
  over-segmentation);
* the chosen setting is scored once on the test split;
* DialSeg711 has no dev split and uses the SuperDialseg dev choice;
* "ΔContext v3 (universal)" is one setting, the best mean J over the three dev
  sets, used unchanged on all four test sets;
* paired bootstrap over test conversations, ΔContext vs every baseline, Holm-
  corrected within each dataset.

Dev splits: LongMemEval -- 100 histories of the seeded dev half; SuperDialseg --
500 validation dialogues (seeded); TIAGE -- its validation split.

Usage:
    python scripts/segbench.py --out paper/results/segbench.json
"""

from __future__ import annotations

import argparse
import itertools
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
from contextbench.core.config_schema import DatasetConfig, PolicyConfig
from contextbench.core.types import Message
from contextbench.evaluation.segmentation import boundary_prf, pk, window_diff
from scripts.paper_tables import holm
from scripts.segmentation_report import _WordCounter, bootstrap_diff, delta_boundaries, dev_test_split

DIALSEG = Path("data/dialseg")


# -- data ----------------------------------------------------------------------------

def dialogues(dataset: str, split: str) -> list[list[tuple[str, str]]]:
    data = json.loads((DIALSEG / dataset / f"{split}.json").read_text(encoding="utf-8"))["dial_data"]
    out = []
    for d in next(iter(data.values())):
        turns = d["turns"]
        if len(turns) < 2:
            continue
        out.append(([(t["role"], t["utterance"]) for t in turns],
                    [i for i, t in enumerate(turns[:-1]) if t.get("segmentation_label") == 1]))
    return out


def longmemeval(n_dev: int):
    from contextbench.datasets.longmemeval import LongMemEvalDataset

    config = DatasetConfig(**yaml.safe_load(Path("configs/dataset/longmemeval_s.yaml").read_text()))
    dev, test = dev_test_split(list(LongMemEvalDataset(config)))
    dev = random.Random(1).sample(dev, n_dev)

    def convert(samples):
        out = []
        for s in samples:
            m = s.messages
            out.append(([(x.speaker or x.role, x.content) for x in m],
                        [i for i in range(len(m) - 1) if m[i].session_id != m[i + 1].session_id]))
        return out
    return convert(dev), convert(test)


# -- methods ---------------------------------------------------------------------------

def grids():
    taus = [round(x, 2) for x in np.arange(0.1, 0.85, 0.05)]
    v3 = [dict(detector="lookahead", lookahead=L, left_context=left, threshold_mode="absolute",
               change_threshold=t, min_segment_turns=m)
          for L, left, t, m in itertools.product((1, 2, 3), (None, 3), taus, (1, 2))]
    v3 += [dict(detector="lookahead", lookahead=L, left_context=left, threshold_mode="zscore",
                change_threshold=z, min_segment_turns=m)
           for L, left, z, m in itertools.product((1, 2, 3), (None, 3), (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0), (1, 2))]
    return {
        "Fixed-size chunks": [dict(size=n) for n in (2, 3, 4, 5, 6, 8, 10, 12)],
        "Adjacent-turn threshold": [dict(theta=round(t, 2)) for t in np.arange(0.3, 0.96, 0.05)],
        "Semantic chunking": [dict(p=p) for p in (50, 60, 70, 75, 80, 85, 90, 93, 95)],
        "TextTiling": [dict(block=b, cutoff_std=c) for b, c in itertools.product((1, 2, 3, 4, 5, 6), (0.0, 0.5, 1.0))],
        "ΔContext v2 (displacement)": [
            dict(detector="accumulator", accumulator="displacement", change_threshold=round(t, 2),
                 window=w, min_snapshot_gap=g, localize_boundaries=True)
            for t, w, g in itertools.product(np.arange(0.2, 0.95, 0.05), (1, 2, 3), (1, 2, 3))],
        "ΔContext v3 (delayed commitment)": v3,
    }


ONLINE = {"Fixed-size chunks", "Adjacent-turn threshold", "ΔContext v2 (displacement)",
          "ΔContext v3 (delayed commitment)", "ΔContext v3 (universal)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lme-dev", type=int, default=100)
    parser.add_argument("--superdialseg-dev", type=int, default=500)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick check: no LongMemEval, tiny splits, every 10th grid point.")
    args = parser.parse_args()

    lme_dev, lme_test = ([], []) if args.smoke else longmemeval(args.lme_dev)
    if args.smoke:
        args.superdialseg_dev = 40
    sd_dev = random.Random(2).sample(dialogues("super_dialseg", "validation"), args.superdialseg_dev)
    sets = {
        **({} if args.smoke else {"LongMemEval-S": (lme_dev, lme_test)}),
        "SuperDialseg": (sd_dev, dialogues("super_dialseg", "test")),
        "TIAGE": (dialogues("tiage", "validation"), dialogues("tiage", "test")),
        "DialSeg711": (sd_dev, dialogues("dialseg711", "test")),
    }

    from contextbench.context.state import SentenceTransformerEmbedder, text_key
    if args.smoke:
        sets = {k: (v[0][:40], v[1][:40]) for k, v in sets.items()}
    lme_cache = SentenceTransformerEmbedder(
        device="cpu", cache_path=None if args.smoke else "data/cache/longmemeval_s_minilm.npz")
    texts = sorted({f"{r}: {u}" for dev, test in sets.values() for convs in (dev, test)
                    for turns, _ in convs for r, u in turns})
    from sentence_transformers import SentenceTransformer
    missing = [t for t in texts if text_key(t) not in lme_cache._cache]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    fresh = model.encode(missing, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    table = {t: lme_cache._cache[text_key(t)] for t in texts if text_key(t) in lme_cache._cache}
    table.update({t: v.astype(np.float32) for t, v in zip(missing, fresh)})
    del lme_cache
    print(f"{len(table)} utterance embeddings ({len(missing)} newly encoded)", flush=True)

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

    method_grids = grids()
    if args.smoke:
        method_grids = {m: g[::10] for m, g in method_grids.items()}
    dev_cache: dict = {}

    def dev_J(name, method, params):
        key = (name, method.split(" (")[0] if method.startswith("ΔContext") else method,
               tuple(sorted(params.items())))
        if key not in dev_cache:
            dev_cache[key] = float(np.mean([s["J"] for s in scores(method, params, sets[name][0])]))
        return dev_cache[key]

    chosen = {name: {} for name in sets}
    tuned_sets = [n for n in ("LongMemEval-S", "SuperDialseg", "TIAGE") if n in sets]
    for name in tuned_sets:
        for method, grid in method_grids.items():
            start = time.perf_counter()
            best = max(grid, key=lambda prm: dev_J(name, method, prm))
            chosen[name][method] = best
            print(f"  tuned {name:14s} {method:34s} {len(grid):4d} configs  dev J {dev_J(name, method, best):.3f}"
                  f"  ({time.perf_counter() - start:.0f}s)  {best}", flush=True)
    chosen["DialSeg711"] = dict(chosen["SuperDialseg"])
    universal = max(method_grids["ΔContext v3 (delayed commitment)"],
                    key=lambda prm: np.mean([dev_J(n, "ΔContext v3 (delayed commitment)", prm)
                                             for n in tuned_sets]))
    print(f"universal ΔContext v3: {universal}", flush=True)

    report = {"protocol": __doc__, "configs": {m: len(g) for m, g in method_grids.items()},
              "universal_delta_v3": universal, "datasets": {}}
    for name, (dev, test) in sets.items():
        methods = dict(chosen[name])
        methods["ΔContext v3 (universal)"] = universal
        per = {}
        for method, params in methods.items():
            kind = "ΔContext v3 (delayed commitment)" if method == "ΔContext v3 (universal)" else method
            per[method] = scores(kind, params, test)
        means = {m: {k: float(np.mean([s[k] for s in v])) for k in ("pk", "windowdiff", "f1@0", "f1@1", "J")}
                 for m, v in per.items()}
        tests, raw = {}, {}
        for ours in ("ΔContext v3 (delayed commitment)", "ΔContext v3 (universal)"):
            for base_m in ("Fixed-size chunks", "Adjacent-turn threshold", "Semantic chunking", "TextTiling",
                           "ΔContext v2 (displacement)"):
                for metric in ("pk", "f1@0"):
                    d, lo, hi, p = bootstrap_diff(np.array([s[metric] for s in per[ours]]),
                                                  np.array([s[metric] for s in per[base_m]]))
                    tests[(ours, base_m, metric)] = (d, lo, hi)
                    raw[(ours, base_m, metric)] = p
        adjusted = holm(raw)
        print(f"\n=== {name}: test conversations {len(test)}")
        print(f"{'method':36s} {'online':>6s} {'Pk':>6s} {'WD':>6s} {'F1@0':>6s} {'F1@1':>6s} {'J':>6s}")
        for m, v in sorted(means.items(), key=lambda kv: -kv[1]["J"]):
            print(f"{m:36s} {'yes' if m in ONLINE else 'no':>6s} {v['pk']:6.3f} {v['windowdiff']:6.3f} "
                  f"{v['f1@0']:6.3f} {v['f1@1']:6.3f} {v['J']:6.3f}")
        for (ours, base_m, metric), (d, lo, hi) in tests.items():
            print(f"  {ours} - {base_m} [{metric}]: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                  f"p_holm={adjusted[(ours, base_m, metric)]:.4f}")
        report["datasets"][name] = {
            "test_conversations": len(test), "dev_conversations": len(dev),
            "chosen": {m: p for m, p in methods.items()}, "test": means,
            "tests": {f"{o} vs {b} [{mt}]": {"diff": tests[(o, b, mt)][0], "ci": tests[(o, b, mt)][1:],
                                               "p_holm": adjusted[(o, b, mt)]}
                      for (o, b, mt) in tests},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
