"""LLMLingua-2 compression of LongMemEval-S turns, as SeCom's denoising step.

SeCom (Pan et al., ICLR 2025) compresses every memory unit with LLMLingua-2
(xlm-roberta-large, MeetingBank) before indexing it, and retrieves on the
compressed text while returning the original. Here each *session* is compressed
as one call -- its turns form the context list, so they share one keep-threshold,
as the turns of a SeCom unit do -- and the result is cached per turn. Turn,
exchange, segment and session keys are then joins of the cached turns, so every
memory granularity uses the same compression.

Settings follow SeCom's code: ``rate`` (fraction of tokens kept; the paper's 75%),
``use_context_level_filter=False`` and ``force_tokens=["\\n", "."]`` (speaker
tags are added after compression, so they need not be forced).

The LongMemEval-S cache was built with ``--fp16 --batch 100`` (2.9x faster than
fp32; on a 100-session check, 97% of compressed turns were identical to fp32
and the rest differed by a few words, mean character similarity 0.9994).

Usage:
    python scripts/compress_memory.py --out data/cache/longmemeval_s_llmlingua2_r075.jsonl --fp16 --batch 100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from contextbench.context.state import text_key

MODEL = "data/models/llmlingua-2-xlm-roberta-large-meetingbank"  # microsoft/llmlingua-2-xlm-roberta-large-meetingbank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/longmemeval/longmemeval_s_cleaned.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=None, help="Compress only this many sessions (benchmark).")
    parser.add_argument("--fp16", action="store_true", help="Half-precision model (faster; labels may differ slightly).")
    parser.add_argument("--batch", type=int, default=50, help="LLMLingua-2 max batch size (chunks of 512 tokens).")
    args = parser.parse_args()

    done = set()
    if args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["k"])
    sessions, seen = [], set()
    for q in json.loads(args.data.read_text(encoding="utf-8")):
        for sid, session in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            if sid in seen:
                continue
            seen.add(sid)
            texts = [t["content"] for t in session if t["content"].strip()]
            if texts and any(text_key(t) not in done for t in texts):
                sessions.append(texts)
    if args.limit:
        sessions = sessions[: args.limit]
    print(f"{len(sessions)} sessions to compress ({len(done)} turns cached)", flush=True)

    import torch
    from llmlingua import PromptCompressor
    compressor = PromptCompressor(model_name=MODEL, use_llmlingua2=True, device_map="cuda",
                                  model_config={"dtype": torch.float16 if args.fp16 else torch.float32},
                                  llmlingua2_config={"max_batch_size": args.batch, "max_force_token": 100})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start, tokens_in, failed = time.perf_counter(), 0, 0
    with args.out.open("a", encoding="utf-8") as f:
        for i, texts in enumerate(sessions):
            # tiktoken (used by LLMLingua for counting) rejects special-token text such as
            # "<|endoftext|>" inside user messages; spacing it out keeps the words.
            safe = [re.sub(r"<\|([a-z_]+)\|>", r"< \1 >", t) for t in texts]
            try:
                result = compressor.compress_prompt(safe, rate=args.rate, use_context_level_filter=False,
                                                    force_tokens=["\n", "."])
                compressed = result["compressed_prompt_list"]
                assert len(compressed) == len(texts)
                tokens_in += result["origin_tokens"]
            except Exception as exc:  # keep the raw text; counted and reported
                failed += 1
                print(f"  session {i}: compression failed ({type(exc).__name__}: {exc}); raw text kept",
                      flush=True)
                compressed = texts
            for t, c in zip(texts, compressed):
                k = text_key(t)
                if k not in done:
                    f.write(json.dumps({"k": k, "c": c}, ensure_ascii=False) + "\n")
                    done.add(k)
            if (i + 1) % 100 == 0 or i + 1 == len(sessions):
                rate = tokens_in / (time.perf_counter() - start)
                left = (len(sessions) - i - 1) * (time.perf_counter() - start) / (i + 1)
                print(f"  {i + 1}/{len(sessions)} sessions, {rate:,.0f} tokens/s, ~{left / 60:.0f} min left",
                      flush=True)
                f.flush()
    print(f"done; {failed} sessions kept raw", flush=True)


if __name__ == "__main__":
    main()
