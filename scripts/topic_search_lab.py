"""Fast offline lab for topic-first search variants on LongMemEval-S.

Simulates the policy's selection (topic ranking -> turn ranking -> greedy
budget fill with skip packing) from precomputed turn embeddings, ΔContext's
online segments and per-turn token counts, so search variants can be compared
in seconds. Final numbers still come from the real harness.

Selection rule, fixed before looking at results: on the dev half, the variant
with the highest evidence recall among those searching at most 25% of a full
scan is chosen; it is then evaluated once on the test half.

Variants:
* ``topk``       -- top-k segments by centroid (current ΔContext store).
* ``gap``        -- every segment scoring within ``delta`` of the best (k adapts).
* ``mass``       -- segments in score order until their turns hold ``c`` x budget
                    tokens (search until there is enough material).
* ``facets``     -- segment key = max over sub-centroids of contiguous runs of
                    ``L`` turns, so a long mixed session keeps several keys.
* ``facets+mass``-- both.

Usage:
    python scripts/topic_search_lab.py --split dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
os.environ.setdefault("USE_TF", "0")

import numpy as np
import yaml

from contextbench.chatbot.formatting import TranscriptFormatter
from contextbench.context.state import SentenceTransformerEmbedder
from contextbench.core.config_schema import DatasetConfig, PolicyConfig
from contextbench.datasets.longmemeval import QUESTION_TYPES, LongMemEvalDataset
from scripts.segmentation_report import _WordCounter, delta_boundaries, dev_test_split

TYPE_NAMES = {v: k for k, v in QUESTION_TYPES.items()}


def normalize(m):
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def segment_ids(n, boundaries):
    return np.searchsorted(np.asarray(sorted(boundaries)), np.arange(n), side="left")


def fill(order, cost, budget):
    used, chosen = 0, []
    for i in order:
        if used + cost[i] <= budget:
            chosen.append(i)
            used += cost[i]
    return chosen


def topic_order(E, q, seg, cost, budget, mode, k=3, delta=0.05, c=2.0, facet_len=None):
    """Turn order and comparisons for one topic-first variant."""
    ids = np.unique(seg)
    members = {s: np.flatnonzero(seg == s) for s in ids}
    keys_per_seg = {}
    for s, idx in members.items():
        if facet_len:
            runs = [idx[i:i + facet_len] for i in range(0, len(idx), facet_len)]
            keys = normalize(np.stack([E[r].mean(0) for r in runs]))
        else:
            keys = normalize(E[idx].mean(0, keepdims=True))
        keys_per_seg[s] = keys
    n_keys = sum(len(v) for v in keys_per_seg.values())
    score = {s: float((keys @ q).max()) for s, keys in keys_per_seg.items()}
    ranked = sorted(ids, key=lambda s: -score[s])
    if mode == "topk":
        chosen = ranked[:k]
    elif mode == "gap":
        best = score[ranked[0]]
        chosen = [s for s in ranked if score[s] >= best - delta][:12] or ranked[:1]
    elif mode == "mass":
        chosen, mass = [], 0
        for s in ranked:
            chosen.append(s)
            mass += int(cost[members[s]].sum())
            if mass >= c * budget:
                break
    else:
        raise ValueError(mode)
    cand = np.concatenate([members[s] for s in chosen])
    sims = E @ q
    first = cand[np.argsort(-sims[cand], kind="stable")]
    rest_mask = np.ones(len(E), bool)
    rest_mask[cand] = False
    rest = np.flatnonzero(rest_mask)
    rest = rest[np.argsort(-sims[rest], kind="stable")]
    return list(first) + list(rest), n_keys + len(cand), set(cand.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--only", default=None, help="Evaluate only this variant label.")
    args = parser.parse_args()

    params = json.loads(Path("paper/results/seg_lme.json").read_text())["chosen_parameters"]
    config = DatasetConfig(**yaml.safe_load(Path("configs/dataset/longmemeval_s.yaml").read_text()))
    samples = list(LongMemEvalDataset(config))
    dev, test = dev_test_split(samples)
    samples = dev if args.split == "dev" else test

    embedder = SentenceTransformerEmbedder(device="cpu", cache_path="data/cache/longmemeval_s_minilm.npz")
    import contextbench.policies.delta_context as dc
    dc.SentenceTransformerEmbedder = lambda **_: embedder
    base = yaml.safe_load(Path("configs/policy/delta_context.yaml").read_text())
    policy = dc.DeltaContextPolicy(PolicyConfig(**{**base, "accumulator": "displacement",
                                                   "change_threshold": params["delta_context"],
                                                   "localize_boundaries": True}), _WordCounter())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    formatter = TranscriptFormatter(include_dates=True)

    variants = {"full scan": None, "topk k=3 (current)": dict(mode="topk", k=3)}
    for k in (4, 5):
        variants[f"topk k={k}"] = dict(mode="topk", k=k)
    for d in (0.02, 0.04, 0.06, 0.08):
        variants[f"gap delta={d}"] = dict(mode="gap", delta=d)
    for c in (1.0, 1.5, 2.0, 3.0):
        variants[f"mass c={c}"] = dict(mode="mass", c=c)
    for L in (3, 5):
        variants[f"facets L={L} k=3"] = dict(mode="topk", k=3, facet_len=L)
        for c in (1.5, 2.0):
            variants[f"facets L={L} + mass c={c}"] = dict(mode="mass", c=c, facet_len=L)
    if args.only:
        variants = {k: v for k, v in variants.items() if k in ("full scan", args.only)}

    acc = defaultdict(lambda: defaultdict(list))
    for h, sample in enumerate(samples):
        qa = sample.qa_items[0]
        if qa.meta["adversarial"] or not qa.evidence:
            continue
        msgs = list(sample.messages)
        n = len(msgs)
        E = normalize(embedder.embed_batch([f"{m.speaker or m.role}: {m.content}" for m in msgs]).astype(np.float64))
        q = normalize(embedder.embed(qa.question).astype(np.float64))
        cost = np.array([len(x) for x in tokenizer([formatter.format_line(m) for m in msgs],
                                                   add_special_tokens=False)["input_ids"]])
        evidence = {i for i, m in enumerate(msgs) if m.meta["dia_id"] in set(qa.evidence)}
        seg = segment_ids(n, delta_boundaries(policy, msgs))
        qtype = TYPE_NAMES[qa.category]
        for name, cfg in variants.items():
            if cfg is None:
                order, comparisons, searched = list(np.argsort(-(E @ q), kind="stable")), n, None
            else:
                order, comparisons, searched = topic_order(E, q, seg, cost, args.budget, **cfg)
            chosen = set(fill(order, cost, args.budget))
            recall = len(evidence & chosen) / len(evidence)
            topic_recall = (len(evidence & chosen & searched) / len(evidence)) if searched is not None else recall
            acc[name]["recall"].append(recall)
            acc[name]["topic_recall"].append(topic_recall)
            acc[name]["searched"].append(comparisons / n)
            acc[name][f"recall:{qtype}"].append(recall)
        if (h + 1) % 50 == 0:
            print(f"  {h + 1}/{len(samples)}", flush=True)

    print(f"\n{args.split}: {len(acc['full scan']['recall'])} answerable questions, budget {args.budget}")
    print(f"{'variant':28s} {'recall':>7s} {'topic rec':>9s} {'searched':>9s} {'know-upd':>8s} {'multi':>7s} {'temporal':>8s}")
    rows = []
    for name, m in acc.items():
        row = (name, np.mean(m["recall"]), np.mean(m["topic_recall"]), np.mean(m["searched"]),
               np.mean(m["recall:knowledge-update"]), np.mean(m["recall:multi-session"]),
               np.mean(m["recall:temporal-reasoning"]))
        rows.append(row)
        print(f"{row[0]:28s} {row[1]:7.3f} {row[2]:9.3f} {100 * row[3]:8.1f}% {row[4]:8.3f} {row[5]:7.3f} {row[6]:8.3f}")
    eligible = [r for r in rows if r[0] != "full scan" and r[3] <= 0.25]
    best = max(eligible, key=lambda r: r[1])
    print(f"\nselected by the pre-set rule (max recall, searched <= 25%): {best[0]}")


if __name__ == "__main__":
    main()
