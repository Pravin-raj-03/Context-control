"""Which parts of ΔContext v3 matter? One-component swaps, re-tuned on dev.

Starting from the dev-chosen v3 setting of each dataset (``segbench.json``),
each variant changes one component and re-tunes its threshold on the same dev
split with the same objective (J = ((1 - Pk) + F1_exact) / 2), then is scored
once on test:

* left context -- whole open topic vs. its last 3 turns;
* threshold    -- absolute vs. self-calibrating (z-score);
* look-ahead   -- 1 / 2 / 3 turns;
* detector     -- v2's immediate displacement decision (no look-ahead).

Usage:
    python scripts/v3_ablation.py --out paper/results/v3_ablation.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
os.environ.setdefault("USE_TF", "0")

import numpy as np
import yaml

from contextbench.core.config_schema import PolicyConfig
from contextbench.core.types import Message
from contextbench.evaluation.segmentation import boundary_prf, pk
from scripts.segbench import dialogues, longmemeval
from scripts.segmentation_report import _WordCounter, bootstrap_diff, delta_boundaries

TAUS = [round(x, 2) for x in np.arange(0.1, 0.85, 0.05)]
ZS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bench = json.loads(Path("paper/results/segbench.json").read_text(encoding="utf-8"))
    lme_dev, lme_test = longmemeval(100)
    sd_dev = random.Random(2).sample(dialogues("super_dialseg", "validation"), 500)
    sets = {"LongMemEval-S": (lme_dev, lme_test),
            "SuperDialseg": (sd_dev, dialogues("super_dialseg", "test"))}

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

    def scores(cfg, convs):
        key = tuple(sorted(cfg.items()))
        if key not in policies:
            policies[key] = dc.DeltaContextPolicy(PolicyConfig(**{**base, **cfg}), _WordCounter())
        out = []
        for turns, ref in convs:
            msgs = [Message(role="user", content=u, turn_id=i, speaker=r) for i, (r, u) in enumerate(turns)]
            hyp = delta_boundaries(policies[key], msgs)
            f0 = boundary_prf(ref, hyp, tolerance=0)[2]
            p_k = pk(ref, hyp, len(turns))
            out.append({"pk": p_k, "f1@0": f0, "J": ((1 - p_k) + f0) / 2})
        return out

    def tuned(variants, dev):
        return max(variants, key=lambda c: np.mean([s["J"] for s in scores(c, dev)]))

    report = {}
    for name, (dev, test) in sets.items():
        full = {k: (float(v) if k == "change_threshold" else v)
                for k, v in bench["datasets"][name]["chosen"]["ΔContext v3 (delayed commitment)"].items()}
        mode_grid = TAUS if full["threshold_mode"] == "absolute" else ZS
        swaps = {"full v3 (chosen)": [full]}
        other_left = 3 if full["left_context"] is None else None
        swaps[f"left context = {'last 3 turns' if other_left else 'whole topic'}"] = [
            {**full, "left_context": other_left, "change_threshold": t} for t in mode_grid]
        other_mode = "zscore" if full["threshold_mode"] == "absolute" else "absolute"
        swaps[f"threshold = {other_mode}"] = [
            {**full, "threshold_mode": other_mode, "change_threshold": t}
            for t in (ZS if other_mode == "zscore" else TAUS)]
        for L in (1, 2, 3):
            if L != full["lookahead"]:
                swaps[f"look-ahead = {L}"] = [{**full, "lookahead": L, "change_threshold": t} for t in mode_grid]
        swaps["v2 detector (no look-ahead)"] = [
            {"detector": "accumulator", "accumulator": "displacement", "change_threshold": round(t, 2),
             "window": w, "min_snapshot_gap": g, "localize_boundaries": True}
            for t, w, g in itertools.product(np.arange(0.2, 0.95, 0.05), (1, 2, 3), (1, 2, 3))]
        rows = {}
        full_test = scores(full, test)
        for label, grid in swaps.items():
            cfg = tuned(grid, dev)
            res = full_test if label.startswith("full") else scores(cfg, test)
            row = {k: float(np.mean([s[k] for s in res])) for k in ("pk", "f1@0", "J")}
            if not label.startswith("full"):
                d, lo, hi, p = bootstrap_diff(np.array([s["J"] for s in full_test]),
                                              np.array([s["J"] for s in res]))
                row["full_minus_this_J"] = {"diff": d, "ci": [lo, hi], "p": p}
            row["config"] = cfg
            rows[label] = row
            extra = (f"  full - this: {row['full_minus_this_J']['diff']:+.3f} "
                     f"[{lo:+.3f}, {hi:+.3f}] p={p:.4f}") if "full_minus_this_J" in row else ""
            print(f"{name:14s} {label:34s} Pk {row['pk']:.3f}  F1 {row['f1@0']:.3f}  J {row['J']:.3f}{extra}",
                  flush=True)
        report[name] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
