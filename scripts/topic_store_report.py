"""Report how well a history stored as topic segments supports topic switching
and retrieval, for every run under a results directory.

Per run (answerable probes with evidence only):

* ``topic hit@1`` / ``topic hit@k``: an evidence turn lies inside the best
  segment / any of the ``topic_first_k`` selected segments -- i.e. returning
  to the right topic when a question resumes it.
* ``recall``: share of evidence turns in the final context, which also
  includes turns found after the selected topics ran out.
* ``topic recall``: share of evidence turns in the final context *and* inside
  the selected topics -- what a store that searches only those topics keeps.
* ``searched``: query-time similarity comparisons as a share of the history's
  turns (segments + turns inside the selected ones, vs. scanning every turn).
* ``purity``: share of the selected context drawn from the evidence's own
  session(s), a proxy for restoring one topic cleanly instead of fragments of
  many. Sessions are topics in LongMemEval; in LoCoMo they are only days.

Usage:
    python scripts/topic_store_report.py experiments/lme_topic --dataset configs/dataset/longmemeval_s.yaml
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

import yaml

from contextbench.core.config_schema import DatasetConfig
from contextbench.datasets.base import DATASET_REGISTRY
import contextbench.datasets.locomo  # noqa: F401  (registration)
import contextbench.datasets.longmemeval  # noqa: F401  (registration)


def label(policy: dict) -> str:
    k = policy.get("topic_first_k", 0)
    if policy["name"] == "delta_context":
        source = "ΔContext segments"
    elif not k:
        return "turn-level retrieval (no topics)"
    elif policy.get("segment_source") == "session":
        source = "true sessions (oracle)"
    else:
        source = f"fixed {policy.get('segment_turns', 20)}-turn chunks"
    return f"{source}, top-{k}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset config used for the runs (for session ids).")
    args = parser.parse_args()

    config = DatasetConfig(**yaml.safe_load(args.dataset.read_text(encoding="utf-8")))
    sessions = {s.sample_id: [m.session_id for m in s.messages]
                for s in DATASET_REGISTRY.create(config.name, config)}

    rows: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for run in sorted(p.parent for p in args.results_dir.glob("*/outputs.jsonl")):
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        hit1 = hitk = recall = topic_recall = searched = purity = 0.0
        n = 0
        for line in (run / "outputs.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            evidence = r.get("evidence_turn_ids") or []
            if r.get("adversarial") or not evidence:
                continue
            n += 1
            meta = r["policy_meta"]
            spans = meta.get("topic_spans") or []
            inside = [any(a <= e <= b for e in evidence) for a, b in spans]
            hit1 += bool(inside[:1] and inside[0])
            hitk += any(inside)
            recall += r["evidence_recall"] or 0.0
            context_set = set(r["context_turn_ids"])
            in_topics = [e for e in evidence if e in context_set
                         and (not spans or any(a <= e <= b for a, b in spans))]
            topic_recall += len(in_topics) / len(evidence)
            n_turns = r["context_messages"] + r["dropped_messages"]
            searched += meta.get("topic_comparisons", n_turns) / max(1, n_turns)
            sess = sessions[r["sample_id"]]
            evidence_sessions = {sess[e] for e in evidence}
            context = r["context_turn_ids"]
            purity += (sum(sess[t] in evidence_sessions for t in context) / len(context)
                       if context else 0.0)
        if n:
            rows[cfg["budget"]].append((label(cfg["policy"]), dict(
                hit1=hit1 / n, hitk=hitk / n, recall=recall / n, topic_recall=topic_recall / n,
                searched=searched / n, purity=purity / n, n=n)))

    for budget in sorted(rows):
        print(f"\n=== budget {budget} ===")
        print(f"{'store':40s} {'hit@1':>6s} {'hit@k':>6s} {'recall':>7s} "
              f"{'topic recall':>12s} {'searched':>9s} {'purity':>7s}")
        for name, m in sorted(rows[budget], key=lambda x: x[0]):
            print(f"{name:40s} {m['hit1']:6.3f} {m['hitk']:6.3f} {m['recall']:7.3f} "
                  f"{m['topic_recall']:12.3f} {m['searched']:8.1%} {m['purity']:7.3f}")


if __name__ == "__main__":
    main()
