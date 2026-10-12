"""Comparison figures for the v3 paper, recomputed from results files and run outputs.

* ``segbench_bars``  -- overall segmentation score J on four datasets, per method;
* ``streaming_bars`` -- boundary revisions per 100 turns in a live chat, and final F1;
* ``qa_bars``        -- LongMemEval QA F1 with 95% bootstrap intervals.

Same validated palette and style as scripts/paper_figures.py.

Usage:
    python scripts/paper_figures_v3.py --out paper/figures
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

from scripts.paper_figures import GRID, SURFACE, TEXT, TEXT_2, style
from scripts.paper_tables import answerable, load_runs

COLORS = {  # validated default categorical order (validate_palette.js: all checks pass)
    "ΔContext v3": "#2a78d6",
    "ΔContext v3 (one setting)": "#2a78d6",  # same entity family; told apart by hatching
    "Fixed-size chunks": "#eb6834",  # same entity colours as scripts/paper_figures.py
    "Semantic chunking": "#1baf7a",
    "TextTiling": "#eda100",
    "Adjacent threshold": "#e87ba4",
}
HATCH = {"ΔContext v3 (one setting)": "////"}


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def segbench_bars(out: Path) -> None:
    """Two panels: (a) each method tuned per dataset; (b) one setting per method for all datasets."""
    tuned = json.loads(Path("paper/results/segbench.json").read_text(encoding="utf-8"))["datasets"]
    universal = json.loads(Path("paper/results/segbench_universal.json").read_text(encoding="utf-8"))["datasets"]
    names = list(tuned)
    methods = [("ΔContext v3", "ΔContext v3 (delayed commitment)", "ΔContext v3 (universal)"),
               ("Fixed-size chunks", "Fixed-size chunks", "Fixed-size chunks"),
               ("Semantic chunking", "Semantic chunking", "Semantic chunking"),
               ("TextTiling", "TextTiling", "TextTiling"),
               ("Adjacent threshold", "Adjacent-turn threshold", "Adjacent-turn threshold")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.15), sharey=True)
    width = 0.16
    x = np.arange(len(names))
    panels = ((axes[0], tuned, 1, "(a) Each method tuned per dataset"),
              (axes[1], universal, 2, "(b) One setting per method for all datasets"))
    for ax, data, col, title in panels:
        for i, entry in enumerate(methods):
            values = [data[n]["test"][entry[col]]["J"] for n in names]
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, values, width * 0.9, color=COLORS[entry[0]],
                   label=entry[0], edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("LongMemEval-S", "LongMemEval") for n in names], fontsize=7.5)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=8)
    axes[0].set_ylabel("J (higher is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=7.5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"segbench_bars.{ext}", dpi=220)
    plt.close(fig)


def streaming_bars(out: Path) -> None:
    res = json.loads(Path("paper/results/streaming_lme.json").read_text(encoding="utf-8"))["results"]
    order = [("ΔContext v3", "ΔContext (online)"), ("Fixed-size chunks", "Fixed-size chunks (online)"),
             ("Semantic chunking", "Semantic chunking (re-run)"), ("TextTiling", "TextTiling (re-run)"),
             ("Adjacent threshold", "Adjacent threshold (online)")]
    labels = [f"{name}\n(final F1 {res[key]['final_f1@1']:.3f})" for name, key in order]
    revisions = [res[k]["removed_boundaries_per_100_turns"] + res[k]["late_cuts_per_100_turns"] for _, k in order]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    y = np.arange(len(order))
    ax.barh(y, revisions, height=0.6, color=[COLORS[n] for n, _ in order], edgecolor=SURFACE, zorder=3)
    for yi, v in zip(y, revisions):
        ax.text(v, yi, f" {v:.1f}", va="center", fontsize=7.5, color=TEXT_2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(revisions) * 1.25)
    ax.set_xlabel("Boundaries revised per 100 new turns", fontsize=8)
    _save(fig, out, "streaming_bars")


def qa_bars(out: Path) -> None:
    runs = load_runs(Path("experiments/lme_llm"))
    order = [("Full scan (turn-level RAG)", "RAG, turn-level", 2048, TEXT_2),
             ("ΔContext v3 topic store", "ΔContext store (top-3)", 2048, COLORS["ΔContext v3"]),
             ("Session-oracle topic store", "Session oracle store (top-3)", 2048, "#008300"),
             ("TextTiling topic store", "TextTiling store (top-3)", 2048, COLORS["TextTiling"]),
             ("Semantic-chunking topic store", "Semantic chunking store (top-3)", 2048, COLORS["Semantic chunking"]),
             ("Fixed-chunk topic store", "Fixed-size chunks (10 turns) store (top-3)", 2048, COLORS["Fixed-size chunks"]),
             ("Sliding window", "Sliding window", 2048, GRID)]
    rng = np.random.default_rng(0)
    labels, means, lows, highs, colors = [], [], [], [], []
    for label, key, budget, color in order:
        records = runs.get((key, budget))
        if not records:
            continue
        f1 = np.array([r["f1"] for r in answerable(records)])
        boots = f1[rng.integers(0, len(f1), size=(5000, len(f1)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        labels.append(label)
        means.append(f1.mean())
        lows.append(f1.mean() - lo)
        highs.append(hi - f1.mean())
        colors.append(color)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = np.arange(len(labels))
    ax.barh(y, means, height=0.6, color=colors, edgecolor=SURFACE, zorder=3,
            xerr=[lows, highs], error_kw=dict(ecolor=TEXT, lw=1, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Token F1 (95% CI)", fontsize=8)
    _save(fig, out, "qa_bars")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    segbench_bars(args.out)
    streaming_bars(args.out)
    qa_bars(args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
