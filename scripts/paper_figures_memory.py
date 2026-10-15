"""Figures and number summary for the conversational-memory paper (SeCom comparison).

Reads the memory-lab results (``paper/results/memory_lab_*.json``) and the QA runs
(``experiments/lme_memory_qa``) and writes:

* ``memory_recall``  -- evidence recall at 2,048 and 4,096 tokens per method (test);
* ``memory_qa``      -- QA token F1 with 95% bootstrap intervals (test);
* ``memory_grid``    -- recall by memory unit x retriever, raw vs LLMLingua-2 keys (dev);
* ``memory_types``   -- recall per LongMemEval question type (test, 2,048 tokens);
* ``paper/results/memory_summary.json`` -- every number the paper quotes.

Usage:
    python scripts/paper_figures_memory.py --out paper/figures
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

from scripts.memory_lab import slug
from scripts.paper_figures import GRID, SURFACE, TEXT, TEXT_2, style

# name in the lab -> (paper label, colour). Colours: validated default order.
OURS, SECOM, TURN, EXCH, CTRL, OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8c8c8c"
METHODS = {
    "SeCom-style | segsess  | comp | bm25": ("SeCom (BM25) [faithful]", SECOM),
    "SeCom-style | segsess  | comp | mpnet": ("SeCom (MPNet)", SECOM),
    "SeCom-style | segsess  | comp | hybrid_minilm": ("SeCom (BM25+MiniLM)", SECOM),
    "SeCom-style | exchange | raw  | hybrid_minilm": ("Exchange-level (BM25+MiniLM)", EXCH),
    "SeCom-style | turn     | raw  | minilm": ("Turn-level dense (MiniLM)", OTHER),
    "SeCom-style | turn     | raw  | hybrid_minilm": ("Turn-level hybrid (BM25+MiniLM)", TURN),
    "ours | segment | topic raw | hybrid_minilm | w=0.25 | partner=False": ("ΔContext topic fusion", OURS),
    "ours+user | boost=0.5 | k=3 | reserve=0.25": ("ΔContext-R (ours)", OURS),
    "control: session grouping | boost=0.5 | k=3 | reserve=0.4": ("Control: true sessions as topics", CTRL),
    "control: fixed10 grouping | boost=0.5 | k=3 | reserve=0.4": ("Control: 10-turn chunks as topics", CTRL),
    "control: global user prior | boost=0.25": ("Control: global user-turn prior", CTRL),
    "control: cost-aware fill | alpha=0.25": ("Control: cost-aware fill", CTRL),
}
TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning"]
TYPE_LABELS = ["SS-user", "SS-assistant", "SS-preference", "Multi-session", "Knowledge-update", "Temporal"]


def load_lab(*paths):
    out = {}
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        out.update(d["results"])
    return out


def load_qa(root=Path("experiments/lme_memory_qa")):
    runs = {}
    for run in sorted(root.glob("*/outputs.jsonl")):
        cfg = json.loads((run.parent / "config.json").read_text(encoding="utf-8"))
        recs = [json.loads(line) for line in run.read_text(encoding="utf-8").splitlines()]
        runs[cfg["policy"]["method"]] = recs
    return runs


def ci(values, rng):
    v = np.asarray(values, dtype=float)
    boots = v[rng.integers(0, len(v), size=(10000, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _save(fig, out, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def fig_recall(test, out):
    rows = [m for m in METHODS if f"{m} @ 2048" in test]
    rows.sort(key=lambda m: test[f"{m} @ 2048"]["recall"])
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    y = np.arange(len(rows))
    for yi, m in zip(y, rows):
        a, b = test[f"{m} @ 2048"]["recall"], test[f"{m} @ 4096"]["recall"]
        color = METHODS[m][1]
        ax.plot([a, b], [yi, yi], color=color, lw=2, zorder=2, alpha=0.6)
        ax.scatter([a], [yi], s=34, facecolor=SURFACE, edgecolor=color, lw=1.6, zorder=3)
        ax.scatter([b], [yi], s=34, color=color, edgecolor=SURFACE, lw=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([METHODS[m][0] for m in rows], fontsize=6.5)
    for lab, m in zip(ax.get_yticklabels(), rows):
        if m.startswith("ours+user"):
            lab.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Evidence recall (test, 237 questions)", fontsize=7.5)
    ax.set_xlim(0.55, 1.0)
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=30, facecolor=SURFACE, edgecolor=TEXT_2, lw=1.4, label="2,048 tokens")
    ax.scatter([], [], s=30, color=TEXT_2, label="4,096 tokens")
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    _save(fig, out, "memory_recall")


def fig_qa(qa_stats, out):
    rows = [m for m in METHODS if slug(f"{m} @ 2048") in qa_stats]
    rows.sort(key=lambda m: qa_stats[slug(f"{m} @ 2048")]["f1"][0])
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    y = np.arange(len(rows))
    for yi, m in zip(y, rows):
        color = METHODS[m][1]
        mean, lo, hi = qa_stats[slug(f"{m} @ 2048")]["f1"]
        ax.errorbar([mean], [yi], xerr=[[mean - lo], [hi - mean]], fmt="o", ms=5, color=color,
                    mfc=SURFACE, mew=1.5, ecolor=color, elinewidth=1.2, capsize=2, zorder=3)
        key4 = slug(f"{m} @ 4096")
        if key4 in qa_stats:
            mean4, lo4, hi4 = qa_stats[key4]["f1"]
            ax.errorbar([mean4], [yi + 0.28], xerr=[[mean4 - lo4], [hi4 - mean4]], fmt="D", ms=4, color=color,
                        ecolor=color, elinewidth=1.0, capsize=2, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([METHODS[m][0] for m in rows], fontsize=6.5)
    for lab, m in zip(ax.get_yticklabels(), rows):
        if m.startswith("ours+user"):
            lab.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("QA token F1, 95% CI (Qwen2.5-3B)", fontsize=7.5)
    ax.errorbar([], [], fmt="o", color=TEXT_2, mfc=SURFACE, label="2,048 tokens")
    ax.errorbar([], [], fmt="D", ms=4, color=TEXT_2, label="4,096 tokens")
    ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    _save(fig, out, "memory_qa")


def fig_grid(dev, out):
    units = [("turn", "Turn"), ("exchange", "Exchange"), ("segment", "ΔContext topic"), ("session", "Session")]
    retr = [("bm25", "BM25"), ("mpnet", "MPNet"), ("minilm", "MiniLM"), ("hybrid", "BM25+\nMPNet"),
            ("hybrid_minilm", "BM25+\nMiniLM")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=True)
    grids = {}
    for ax, key, title in ((axes[0], "raw", "(a) Raw keys"), (axes[1], "comp", "(b) LLMLingua-2 keys (SeCom denoising)")):
        M = np.full((len(units), len(retr)), np.nan)
        for i, (u, _) in enumerate(units):
            for j, (r, _) in enumerate(retr):
                k = "raw" if r == "minilm" else key
                name = f"SeCom-style | {u:8s} | {k:4s} | {r} @ 2048"
                if name in dev:
                    M[i, j] = dev[name]["recall"]
        grids[key] = M.tolist()
        ax.imshow(M, cmap="Blues", vmin=0.40, vmax=0.90, aspect="auto")
        for i in range(len(units)):
            for j in range(len(retr)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > 0.72 else TEXT)
        ax.set_xticks(range(len(retr)))
        ax.set_xticklabels([r[1] for r in retr], fontsize=6.5)
        ax.set_yticks(range(len(units)))
        ax.set_yticklabels([u[1] for u in units], fontsize=7)
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    _save(fig, out, "memory_grid")
    return grids


def fig_types(test, out):
    show = ["SeCom-style | segsess  | comp | bm25", "SeCom-style | turn     | raw  | hybrid_minilm",
            "control: cost-aware fill | alpha=0.25", "ours+user | boost=0.5 | k=3 | reserve=0.25"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    width = 0.2
    x = np.arange(len(TYPES))
    for i, m in enumerate(show):
        vals = [test[f"{m} @ 2048"].get(t, np.nan) for t in TYPES]
        ax.bar(x + (i - 1.5) * width, vals, width * 0.92, color=METHODS[m][1], label=METHODS[m][0],
               edgecolor=SURFACE, linewidth=0.6, zorder=3,
               hatch="////" if m.startswith("control") else None)
    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_LABELS, fontsize=6, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Evidence recall", fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=5.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    _save(fig, out, "memory_types")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    test = load_lab("paper/results/memory_lab_test.json", "paper/results/memory_lab_test2.json")
    dev = load_lab("paper/results/memory_lab_dev.json", "paper/results/memory_lab_dev_controls.json")
    rng = np.random.default_rng(0)
    qa_stats = {}
    for method, recs in load_qa().items():
        qa_stats[method] = {"n": len(recs), "f1": ci([r["f1"] for r in recs], rng),
                            "em": ci([r["exact_match"] for r in recs], rng),
                            "context_tokens": float(np.mean([r["context_tokens"] for r in recs])),
                            "context_messages": float(np.mean([r["context_messages"] for r in recs]))}
    fig_recall(test, args.out)
    fig_qa(qa_stats, args.out)
    grids = fig_grid(dev, args.out)
    fig_types(test, args.out)
    summary = {"test": {m: {b: {k: v for k, v in test[f"{m} @ {b}"].items() if k != "per_question"}
                            for b in (2048, 4096) if f"{m} @ {b}" in test} for m in METHODS},
               "qa": {m: {b: qa_stats[slug(f"{m} @ {b}")] for b in (2048, 4096) if slug(f"{m} @ {b}") in qa_stats}
                      for m in METHODS},
               "dev_grid_2048": grids,
               "labels": {m: v[0] for m, v in METHODS.items()}}
    Path("paper/results/memory_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")
    print("figures and paper/results/memory_summary.json written")


if __name__ == "__main__":
    main()
