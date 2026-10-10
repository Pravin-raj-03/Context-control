"""Build the paper's result tables (LaTeX + Markdown) and figures from run folders.

Every number is recomputed from the runs' ``outputs.jsonl`` (answerable probes
only, unless stated), so tables cannot drift from the data. Comparisons are
ΔContext vs. each baseline on the same probes:

* continuous metrics (F1, EM, evidence recall, topic recall): paired bootstrap,
  10,000 resamples, 95% CI of the mean difference;
* binary metrics (topic hit@k): exact McNemar test on discordant pairs;
* p-values Holm-Bonferroni-corrected within each table.

Usage:
    python scripts/paper_tables.py --out paper
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

SOURCE_LABELS = {
    "fixed": "Fixed-size chunks",
    "session": "Session oracle",
    "adjacent": "Adjacent threshold",
    "semantic_chunker": "Semantic chunking",
    "texttiling": "TextTiling",
}
OURS_PREFIX = "ΔContext"


# -- run loading ----------------------------------------------------------------

def method_label(config: dict) -> str:
    p, budget = config["policy"], config["budget"]
    name, k = p["name"], p.get("topic_first_k", 0)
    if name == "sliding_window":
        return "Full context (upper bound)" if budget >= 16000 else "Sliding window"
    if name == "summary":
        return "Extractive summary"
    if name == "retrieval":
        if k:
            return f"{SOURCE_LABELS[p.get('segment_source', 'fixed')]} store (top-{k})"
        if p.get("segment_prior_weight"):
            return "RAG + chunk prior"
        return "RAG, turn-level" if p.get("retrieval_fraction") == 1.0 else "RAG + recency tail (50/50)"
    if name == "delta_context":
        suffix = []
        if p.get("accumulator", "sum") == "sum" and k:
            suffix.append("path-length detector")
        elif k and not p.get("localize_boundaries"):
            suffix.append("no localization")
        tag = f" ({', '.join(suffix)})" if suffix else ""
        if k:
            return f"{OURS_PREFIX} store (top-{k}){tag}"
        if p.get("segment_prior_weight"):
            return f"{OURS_PREFIX} segment prior"
        return f"{OURS_PREFIX} turn ranking (v1.2)"
    return name


def load_runs(root: Path) -> dict[tuple[str, int], list[dict]]:
    """``(method label, budget) -> per-probe records`` for every run under ``root``.

    When a (label, budget) pair was run more than once, the latest run wins.
    """
    runs: dict[tuple[str, int], list[dict]] = {}
    for run in sorted(p.parent for p in root.glob("*/outputs.jsonl")):
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in
                   (run / "outputs.jsonl").read_text(encoding="utf-8").splitlines()]
        runs[(method_label(config), config["budget"])] = records
    return runs


# -- per-probe metrics ------------------------------------------------------------

def topic_hit(record: dict, k: int) -> float | None:
    evidence = record.get("evidence_turn_ids") or []
    spans = (record.get("policy_meta") or {}).get("topic_spans")
    if not evidence or not spans:
        return None
    return float(any(a <= e <= b for a, b in spans[:k] for e in evidence))


def topic_recall(record: dict) -> float | None:
    evidence = record.get("evidence_turn_ids") or []
    spans = (record.get("policy_meta") or {}).get("topic_spans")
    if not evidence:
        return None
    context = set(record["context_turn_ids"])
    kept = [e for e in evidence if e in context and (not spans or any(a <= e <= b for a, b in spans))]
    return len(kept) / len(evidence)


def searched(record: dict) -> float:
    meta = record.get("policy_meta") or {}
    n_turns = record["context_messages"] + record["dropped_messages"]
    return meta.get("topic_comparisons", n_turns) / max(1, n_turns)


METRICS = {
    "f1": lambda r: r.get("f1"),
    "em": lambda r: r.get("exact_match"),
    "evidence_recall": lambda r: r.get("evidence_recall"),
    "topic_recall": topic_recall,
    "hit@1": lambda r: topic_hit(r, 1),
    "hit@3": lambda r: topic_hit(r, 3),
    "searched": searched,
    "context_tokens": lambda r: r.get("context_tokens"),
}
BINARY = {"hit@1", "hit@3"}


def answerable(records):
    return [r for r in records if not r.get("adversarial", r.get("category") == 5)]


def paired(a_records, b_records, metric):
    fn = METRICS[metric]
    b_by_id = {r["qa_id"]: r for r in answerable(b_records)}
    xs, ys = [], []
    for r in answerable(a_records):
        other = b_by_id.get(r["qa_id"])
        if other is None:
            continue
        x, y = fn(r), fn(other)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return np.asarray(xs), np.asarray(ys)


def mean_metric(records, metric):
    values = [METRICS[metric](r) for r in answerable(records)]
    values = [float(v) for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


def abstention(records):
    adversarial = [r for r in records if r.get("adversarial", r.get("category") == 5)]
    if not adversarial or not any(r.get("prediction") for r in adversarial):
        return None
    return sum(bool(r.get("abstained")) for r in adversarial) / len(adversarial)


# -- statistics ------------------------------------------------------------------------

def bootstrap(x, y, resamples=10_000, seed=0):
    diff = x - y
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    return float(diff.mean()), float(lo), float(hi), float(p)


def mcnemar(x, y):
    """Exact two-sided McNemar test on paired binary outcomes."""
    b = int(np.sum((x == 1) & (y == 0)))
    c = int(np.sum((x == 0) & (y == 1)))
    n = b + c
    if n == 0:
        return float(np.mean(x - y)), 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return float(np.mean(x - y)), min(1.0, 2 * tail)


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family of tests."""
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    m = len(order)
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


# -- table rendering --------------------------------------------------------------------

def fmt(value, pct=False):
    if value is None:
        return "--"
    return f"{100 * value:.1f}" if pct else f"{value:.3f}"


def comparison_table(title, runs, budget, rows, metrics, ours, out_dir, stem, pct_metrics=()):
    """One table: a row per method, one column per metric, plus ΔContext-vs-row tests."""
    present = [m for m in rows if (m, budget) in runs]
    tests, raw_p = {}, {}
    for method in present:
        if method == ours or (ours, budget) not in runs:
            continue
        for metric in metrics:
            x, y = paired(runs[(ours, budget)], runs[(method, budget)], metric)
            if len(x) == 0:
                continue
            if metric in BINARY:
                d, p = mcnemar(x, y)
                tests[(method, metric)] = (d, None, None, p)
            elif metric != "searched" and metric != "context_tokens":
                tests[(method, metric)] = bootstrap(x, y)
            else:
                continue
            raw_p[(method, metric)] = tests[(method, metric)][3]
    adjusted = holm(raw_p) if raw_p else {}

    header = ["Method"] + metrics + (["Abstain"] if "f1" in metrics else [])
    lines_md = [f"### {title} (budget {budget})", "",
                "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines_tex = [r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule",
                 " & ".join(h.replace("@", r"@") for h in header) + r" \\", r"\midrule"]
    for method in present:
        cells = []
        for metric in metrics:
            value = mean_metric(runs[(method, budget)], metric)
            cell = fmt(value, metric in pct_metrics)
            test = tests.get((method, metric))
            if test is not None:
                p = adjusted[(method, metric)]
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                cell += stars
            cells.append(cell)
        if "f1" in metrics:
            cells.append(fmt(abstention(runs[(method, budget)])))
        name = f"**{method}**" if method == ours else method
        lines_md.append("| " + " | ".join([name] + cells) + " |")
        tex_name = method.replace("ΔContext", r"$\Delta$Context").replace("%", r"\%")
        tex_cells = [c.replace("*", r"$^{*}$") for c in cells]
        lines_tex.append(" & ".join([r"\textbf{" + tex_name + "}" if method == ours else tex_name]
                                    + tex_cells) + r" \\")
    lines_tex += [r"\bottomrule", r"\end{tabular}"]

    detail = [f"", f"ΔContext ({ours}) minus each baseline; Holm-adjusted p within this table:"]
    for (method, metric), (d, lo, hi, p) in sorted(tests.items()):
        ci = f" [{lo:+.3f}, {hi:+.3f}]" if lo is not None else " (McNemar)"
        detail.append(f"- vs {method}, {metric}: {d:+.3f}{ci}, p_adj={adjusted[(method, metric)]:.4f}")
    lines_md += detail + [""]
    (out_dir / "tables" / f"{stem}.tex").write_text("\n".join(lines_tex) + "\n", encoding="utf-8")
    return "\n".join(lines_md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("paper"))
    args = parser.parse_args()
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    sections = ["# ContextBench / ΔContext: result tables", "",
                "Answerable probes only; significance markers: * p<0.05, ** p<0.01, *** p<0.001 "
                "(ΔContext vs that row, Holm-adjusted within the table).", ""]

    # One row per way of storing the history; every store searched top-3 topics.
    # Stage-1 topic ranking does not depend on k, so hit@1 and hit@3 both come
    # from the top-3 runs.
    ours_store = f"{OURS_PREFIX} store (top-3)"
    store_rows = [ours_store] + [f"{label} store (top-3)" for label in SOURCE_LABELS.values()]
    ablation_rows = [ours_store, f"{OURS_PREFIX} store (top-3) (no localization)",
                     f"{OURS_PREFIX} store (top-3) (path-length detector)"]
    for name, root, budget in (("LongMemEval-S topic store", "experiments/lme_topic", 2048),
                               ("LoCoMo topic store", "experiments/locomo_topic", 2048)):
        if Path(root).is_dir():
            runs = load_runs(Path(root))
            stem = name.split()[0].lower()
            sections.append(comparison_table(
                name, runs, budget, store_rows + ["RAG, turn-level"],
                ["hit@1", "hit@3", "topic_recall", "evidence_recall", "searched"],
                ours_store, args.out, stem + "_topic", pct_metrics=("searched",)))
            if any((row, budget) in runs for row in ablation_rows[1:]):
                sections.append(comparison_table(
                    name + " ablation", runs, budget, ablation_rows,
                    ["hit@1", "hit@3", "topic_recall"], ours_store, args.out, stem + "_ablation"))

    for name, root, budget in (("LoCoMo end-to-end QA (Qwen2.5-3B)", "experiments/v2_llm", 512),
                               ("LongMemEval-S end-to-end QA (Qwen2.5-3B)", "experiments/lme_llm", 2048)):
        if Path(root).is_dir():
            runs = load_runs(Path(root))
            rows = ["Sliding window", "Extractive summary", "RAG + recency tail (50/50)", "RAG, turn-level",
                    "RAG + chunk prior", f"{OURS_PREFIX} turn ranking (v1.2)"] + store_rows
            ours = f"{OURS_PREFIX} store (top-3)" if (f"{OURS_PREFIX} store (top-3)", budget) in runs \
                else f"{OURS_PREFIX} turn ranking (v1.2)"
            sections.append(comparison_table(
                name, runs, budget, rows, ["f1", "em", "evidence_recall", "context_tokens"],
                ours, args.out, name.split()[0].lower() + "_qa"))
            upper = runs.get(("Full context (upper bound)", 32000))
            if upper:
                sections.append(f"Full-context upper bound: F1 {fmt(mean_metric(upper, 'f1'))}, "
                                f"EM {fmt(mean_metric(upper, 'em'))}, context "
                                f"{mean_metric(upper, 'context_tokens'):.0f} tokens.\n")

    for seg in sorted(Path("paper/results").glob("seg_*.json")) if Path("paper/results").is_dir() else []:
        data = json.loads(seg.read_text(encoding="utf-8"))
        sections += [f"### Segmentation quality: {data['dataset']} "
                     f"(test histories: {data['test_histories']}, parameters tuned on dev)", "",
                     "| Method | Pk | WindowDiff | P@1 | R@1 | F1@1 | F1@0 | Segments |", "|---|---|---|---|---|---|---|---|"]
        for method, m in data["test_results"].items():
            sections.append(f"| {method} | {m['pk']:.3f} | {m['windowdiff']:.3f} | {m['p@1']:.3f} | "
                            f"{m['r@1']:.3f} | {m['f1@1']:.3f} | {m['f1@0']:.3f} | {m['segments']:.1f} |")
        sections.append("")

    (args.out / "RESULTS_TABLES.md").write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))


if __name__ == "__main__":
    main()
