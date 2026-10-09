"""Data-driven calibration for ΔContext's controller thresholds.

Running a full LoCoMo benchmark to grid-search `change_threshold` /
`ignore_below` / `min_snapshot_gap` costs 30-140 minutes *per config* (the LLM
generation dominates). This script instead runs only the cheap half of the
pipeline -- embed -> state -> cosine metric -- over the real LoCoMo
conversations (no LLM, no GPU generation) to answer two questions directly
from data:

1. What does the actual instantaneous-delta distribution look like for this
   embedder/window on this dataset? (percentiles -> sane `ignore_below` /
   `change_threshold` values, instead of guessing).
2. For a handful of candidate (change_threshold, min_snapshot_gap) pairs, how
   many checkpoints would actually fire per conversation, and how far apart
   are they? The diagnosis from the first benchmark sweep was that the
   defaults fire a checkpoint roughly every 2.5 turns (~40% of all turns) --
   this reports the same statistic for each candidate so a threshold can be
   picked that lands checkpoints near genuine topic boundaries (a target of
   roughly one checkpoint per 15-30 turns) instead of on almost every turn.

Usage:
    python scripts/calibrate_delta.py
    python scripts/calibrate_delta.py --window 3 --embedder sentence-transformers/all-MiniLM-L6-v2

Runs in well under a minute on CPU for all 10 LoCoMo conversations.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from contextbench.context.embedding_metric import CosineDistanceMetric
from contextbench.context.state import SemanticState, SentenceTransformerEmbedder
from contextbench.core.config_schema import DatasetConfig
from contextbench.datasets.locomo import LoCoMoDataset


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def collect_deltas(dataset: LoCoMoDataset, embedder: SentenceTransformerEmbedder, window: int):
    """Return (all_deltas, per_conversation_deltas) via the real ΔContext pipeline.

    Reimplements ``WindowedCentroidConstructor``'s windowed-centroid formula
    directly (rather than calling it turn by turn) so the whole conversation's
    turns can be embedded in a single batched ``embed_batch`` call -- one HF
    forward pass instead of hundreds of one-off ones. Distances still go
    through the real :class:`CosineDistanceMetric`, so the numbers this script
    reports are exactly what ``DeltaContextPolicy.observe()`` would compute.
    """
    metric = CosineDistanceMetric()
    all_deltas: list[float] = []
    per_conv: list[list[float]] = []

    for sample in dataset:
        texts = [f"{m.speaker or m.role}: {m.content}" for m in sample.messages]
        deltas: list[float] = []
        if len(texts) >= 2:
            embeddings = embedder.embed_batch(texts)
            buffer: list[np.ndarray] = []
            last_state: SemanticState | None = None
            for i, emb in enumerate(embeddings):
                buffer.append(emb)
                if len(buffer) > window:
                    buffer.pop(0)
                centroid = np.mean(buffer, axis=0).astype(np.float32)
                state = SemanticState(embedding=centroid, token_count=0, turn_index=i,
                                      timestamp=float(i))
                if last_state is not None:
                    deltas.append(metric.distance(last_state, state))
                last_state = state
        all_deltas.extend(deltas)
        per_conv.append(deltas)
        print(f"  {sample.sample_id}: {len(sample.messages)} turns, {len(deltas)} deltas")

    return all_deltas, per_conv


def simulate_checkpoints(per_conv: list[list[float]], change_threshold: float,
                          ignore_below: float, min_snapshot_gap: int) -> dict:
    """Replay the sum-accumulator + threshold-controller decision rule (V1's
    default) over each conversation's delta sequence and count checkpoints,
    without needing the full policy/controller machinery."""
    total_checkpoints = 0
    total_turns = 0
    gaps: list[int] = []
    for deltas in per_conv:
        acc = 0.0
        last_ckpt_turn = -1
        for turn_idx, delta in enumerate(deltas):
            acc += delta
            turns_since = turn_idx - last_ckpt_turn
            if acc >= change_threshold and turns_since >= min_snapshot_gap:
                total_checkpoints += 1
                gaps.append(turns_since)
                acc = 0.0
                last_ckpt_turn = turn_idx
        total_turns += len(deltas)
    avg_gap = statistics.mean(gaps) if gaps else float("nan")
    return {
        "change_threshold": change_threshold,
        "min_snapshot_gap": min_snapshot_gap,
        "ignore_below": ignore_below,
        "total_checkpoints": total_checkpoints,
        "total_turns": total_turns,
        "checkpoint_rate": total_checkpoints / total_turns if total_turns else float("nan"),
        "avg_turns_between_checkpoints": avg_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()

    if not Path(args.data).is_file():
        raise SystemExit(f"LoCoMo data not found at {args.data}")

    print(f"Loading LoCoMo from {args.data} ...")
    dataset = LoCoMoDataset(DatasetConfig(name="locomo", path=args.data))
    print(f"{len(dataset)} conversations.")

    print(f"Loading embedder {args.embedder} (CPU) ...")
    embedder = SentenceTransformerEmbedder(model_name=args.embedder, device="cpu")

    print(f"Computing instantaneous-delta trajectories (window={args.window}) ...")
    all_deltas, per_conv = collect_deltas(dataset, embedder, args.window)

    print("\n=== delta_t distribution (all conversations pooled) ===")
    print(f"n = {len(all_deltas)}")
    for p in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  p{int(p*100):>2} = {_percentile(all_deltas, p):.4f}")
    print(f"  mean = {statistics.mean(all_deltas):.4f}  stdev = {statistics.pstdev(all_deltas):.4f}")

    recommended_ignore_below = _percentile(all_deltas, 0.15)
    print(f"\nRecommended ignore_below (~p15): {recommended_ignore_below:.4f}")

    print("\n=== checkpoint-frequency sweep (sum accumulator) ===")
    print(f"{'threshold':>10} {'gap':>4} {'checkpoints':>12} {'rate':>8} {'avg_turns_between':>18}")
    candidates = []
    for change_threshold in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0):
        for min_snapshot_gap in (1, 5, 10):
            stats = simulate_checkpoints(
                per_conv, change_threshold, recommended_ignore_below, min_snapshot_gap
            )
            candidates.append(stats)
            print(f"{change_threshold:>10.2f} {min_snapshot_gap:>4} "
                  f"{stats['total_checkpoints']:>12} {stats['checkpoint_rate']:>8.3f} "
                  f"{stats['avg_turns_between_checkpoints']:>18.1f}")

    # Recommend the candidate whose avg gap is closest to a 15-30 turn target
    # band (genuine topic boundaries, not noise-level churn), preferring the
    # candidate nearest the middle of that band.
    target = 22.0
    in_band = [c for c in candidates if 15 <= c["avg_turns_between_checkpoints"] <= 30]
    pool = in_band or candidates
    best = min(pool, key=lambda c: abs(c["avg_turns_between_checkpoints"] - target))
    print(f"\nRecommended config: change_threshold={best['change_threshold']}, "
          f"min_snapshot_gap={best['min_snapshot_gap']}, "
          f"ignore_below={recommended_ignore_below:.4f} "
          f"-> ~{best['avg_turns_between_checkpoints']:.1f} turns/checkpoint, "
          f"{best['total_checkpoints']} checkpoints total across {len(dataset)} conversations.")


if __name__ == "__main__":
    main()
