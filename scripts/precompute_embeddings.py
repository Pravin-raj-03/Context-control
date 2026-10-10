"""Precompute turn and question embeddings for a dataset into a reusable cache.

Embedding policies (retrieval, ΔContext) embed every turn they observe, one at
a time, in every run. On LongMemEval that is ~250K turns per run, most of them
repeats, since questions share haystack sessions. This script embeds each
unique text once, in batches, and writes an ``.npz`` that policies load via
their ``embedding_cache`` option. It embeds exactly the strings the policies
embed: ``"<speaker or role>: <content>"`` per turn, plus each question.

Usage:
    python scripts/precompute_embeddings.py configs/dataset/longmemeval_s.yaml \\
        data/cache/longmemeval_s_minilm.npz --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")

import numpy as np
import yaml

from contextbench.context.state import SentenceTransformerEmbedder, text_key
from contextbench.core.config_schema import DatasetConfig
from contextbench.datasets.base import DATASET_REGISTRY
import contextbench.datasets.locomo  # noqa: F401  (registration)
import contextbench.datasets.longmemeval  # noqa: F401  (registration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset_config", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = DatasetConfig(**yaml.safe_load(args.dataset_config.read_text(encoding="utf-8")))
    dataset = DATASET_REGISTRY.create(config.name, config)

    texts: dict[str, str] = {}
    for sample in dataset:
        for m in sample.messages:
            text = f"{m.speaker or m.role}: {m.content}"
            texts.setdefault(text_key(text), text)
        for qa in sample.qa_items:
            texts.setdefault(text_key(qa.question), qa.question)
    keys = list(texts)
    print(f"{len(dataset)} samples -> {len(keys)} unique texts", flush=True)

    embedder = SentenceTransformerEmbedder(args.model, device=args.device)
    start = time.perf_counter()
    # Longest-first batches waste less padding; order is restored by key.
    order = sorted(range(len(keys)), key=lambda i: -len(texts[keys[i]]))
    vectors = np.zeros((len(keys), embedder.dim), dtype=np.float32)
    for begin in range(0, len(order), args.batch_size):
        batch = order[begin:begin + args.batch_size]
        vectors[batch] = embedder.embed_batch([texts[keys[i]] for i in batch])
        if (begin // args.batch_size) % 50 == 0:
            print(f"  {begin + len(batch)}/{len(keys)}", flush=True)
    print(f"embedded in {time.perf_counter() - start:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, model=np.asarray(args.model), keys=np.asarray(keys), vectors=vectors)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
