"""Paper figures, recomputed from run outputs.

1. ``cost_recall`` -- LongMemEval: share of the history searched per question
   vs. topic-only evidence recall, one line per way of storing the history
   (top-1/3/5 topics), plus turn-level search over every turn.
2. ``topic_hits`` -- LongMemEval: topic hit@k (k = 1..5), i.e. how often the
   right topic is among the first k a question returns to.
3. ``segmentation_example`` -- one LongMemEval test history: true session
   boundaries vs. where each segmenter cuts.

Colors are the validated categorical slots in fixed order (ΔContext first);
every series also carries a distinct marker and a direct label, since three
slots sit under 3:1 contrast on the light surface.

Usage:
    python scripts/paper_figures.py --out paper/figures [--example]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.paper_tables import METRICS, answerable, load_runs, mean_metric

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [  # (label prefix in run names, legend label, color, marker)
    ("ΔContext store", "ΔContext (ours)", "#2a78d6", "o"),
    ("Fixed-size chunks store", "Fixed-size chunks", "#eb6834", "s"),
    ("Semantic chunking store", "Semantic chunking", "#1baf7a", "^"),
    ("TextTiling store", "TextTiling", "#eda100", "D"),
    ("Adjacent threshold store", "Adjacent threshold", "#e87ba4", "v"),
    ("Session oracle store", "Session oracle (upper ref.)", "#008300", "P"),
]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
        "ytick.color": TEXT_2, "text.color": TEXT, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })


def mark(ax, xs, ys, color, marker, label, dashed=False):
    """A 2px series line with ringed markers; only ours is direct-labelled
    (the legend and marker shapes identify the rest without collisions)."""
    ours = label.startswith("ΔContext")
    ax.plot(xs, ys, color=color, marker=marker, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, linestyle="--" if dashed else "-", label=label,
            zorder=4 if ours else 3)
    if ours:
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 4), textcoords="offset points",
                    va="bottom", fontsize=8, color=TEXT, fontweight="bold")


def cost_recall(runs, budget, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for prefix, label, color, marker in SERIES:
        points = [(mean_metric(runs[(f"{prefix} (top-{k})", budget)], "searched"),
                   mean_metric(runs[(f"{prefix} (top-{k})", budget)], "topic_recall"))
                  for k in (1, 3, 5) if (f"{prefix} (top-{k})", budget) in runs]
        if points:
            xs, ys = zip(*points)
            mark(ax, [100 * x for x in xs], ys, color, marker, label, dashed="oracle" in label)
    full = runs.get(("RAG, turn-level", budget))
    if full:
        y = mean_metric(full, "evidence_recall")
        ax.plot([100], [y], marker="*", markersize=12, color=TEXT_2, markeredgecolor=SURFACE,
                linestyle="none", zorder=3)
        ax.annotate("Search every turn", (100, y), xytext=(-6, 8), textcoords="offset points",
                    ha="right", fontsize=8, color=TEXT_2)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 50, 100])
    ax.set_xticklabels(["10%", "20%", "30%", "50%", "100%"])
    ax.minorticks_off()
    ax.set_xlabel("Turns searched per question (share of a full scan, log scale)")
    ax.set_ylabel("Evidence recall within searched topics")
    ax.set_xlim(8, 130)
    ax.set_title(f"LongMemEval-S, budget {budget} tokens: top-1/3/5 topics", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"cost_recall.{ext}", dpi=200)
    plt.close(fig)


def topic_hits(runs, budget, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ks = [1, 2, 3, 4, 5]
    for prefix, label, color, marker in SERIES:
        records = runs.get((f"{prefix} (top-5)", budget))
        if not records:
            continue
        ys = []
        for k in ks:
            values = [v for v in (METRICS["hit@1"](r) if k == 1 else _hit(r, k)
                                  for r in answerable(records)) if v is not None]
            ys.append(sum(values) / len(values))
        mark(ax, ks, ys, color, marker, label, dashed="oracle" in label)
    ax.set_xticks(ks)
    ax.set_xlabel("Topics returned (k)")
    ax.set_ylabel("Questions whose evidence topic is in the top k")
    ax.set_xlim(0.8, 5.4)
    ax.set_title(f"LongMemEval-S: returning to the right topic", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"topic_hits.{ext}", dpi=200)
    plt.close(fig)


def _hit(record, k):
    evidence = record.get("evidence_turn_ids") or []
    spans = (record.get("policy_meta") or {}).get("topic_spans")
    if not evidence or not spans:
        return None
    return float(any(a <= e <= b for a, b in spans[:k] for e in evidence))


def segmentation_example(out: Path, params: dict, index: int = 0) -> None:
    """True session boundaries vs. each segmenter on one test history."""
    import os

    import yaml

    os.environ.setdefault("USE_TF", "0")
    from contextbench.context import segmenters
    from contextbench.context.state import SentenceTransformerEmbedder
    from contextbench.core.config_schema import DatasetConfig, PolicyConfig
    from contextbench.datasets.longmemeval import LongMemEvalDataset
    import contextbench.policies.delta_context as dc
    from scripts.segmentation_report import _WordCounter, delta_boundaries

    config = DatasetConfig(**yaml.safe_load(Path("configs/dataset/longmemeval_s.yaml").read_text()))
    samples = list(LongMemEvalDataset(config))
    sample = samples[len(samples) // 2 + index]  # first test-half history
    embedder = SentenceTransformerEmbedder(device="cpu", cache_path="data/cache/longmemeval_s_minilm.npz")
    e = embedder.embed_batch([f"{m.speaker or m.role}: {m.content}" for m in sample.messages])
    dc.SentenceTransformerEmbedder = lambda **_: embedder
    base = yaml.safe_load(Path("configs/policy/delta_context.yaml").read_text())
    policy = dc.DeltaContextPolicy(PolicyConfig(**{**base, "accumulator": "displacement",
                                                   "change_threshold": params["delta_context"],
                                                   "localize_boundaries": True}), _WordCounter())
    n = len(sample.messages)
    rows = [
        ("True sessions", [i for i in range(n - 1)
                           if sample.messages[i].session_id != sample.messages[i + 1].session_id], TEXT),
        ("ΔContext (ours)", delta_boundaries(policy, sample.messages), SERIES[0][2]),
        ("Semantic chunking", segmenters.semantic_chunker_boundaries(e, params["semantic_chunker"]), SERIES[2][2]),
        ("TextTiling", segmenters.texttiling_boundaries(e, int(params["texttiling"])), SERIES[3][2]),
        ("Fixed-size chunks", segmenters.fixed_boundaries(n, int(params["fixed"])), SERIES[1][2]),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    window = min(n, 160)
    for y, (label, bounds, color) in enumerate(reversed(rows)):
        xs = [b + 0.5 for b in bounds if b < window]
        ax.vlines(xs, y - 0.35, y + 0.35, color=color, linewidth=1.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlim(0, window)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Turn (first {window} of {n}, LongMemEval question {sample.sample_id})")
    ax.set_title("Where each method places topic boundaries", loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"segmentation_example.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--example", action="store_true",
                        help="Also draw the segmentation example (loads LongMemEval; ~2.5 GB RAM).")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    runs = load_runs(Path("experiments/lme_topic"))
    cost_recall(runs, args.budget, args.out)
    topic_hits(runs, args.budget, args.out)
    if args.example:
        params = json.loads(Path("paper/results/seg_lme.json").read_text())["chosen_parameters"]
        segmentation_example(args.out, params)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
