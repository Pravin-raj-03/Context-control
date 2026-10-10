# Experimental protocol

This is the setup behind every number in `RESULTS.md` and `tables/`. Each
section names the code that implements it, so the paper's Methods and
Experimental Setup can be written from it directly.

## 1. Problem

A conversational model answers questions about a long chat history under a
fixed **context budget** B (tokens of history in the prompt). The history is far
longer than B (LoCoMo: 16K–30K tokens; LongMemEval-S: ~115K tokens per
question), so a *context management policy* decides which past turns the
model sees. Everything except the policy is held fixed: model, tokenizer,
prompt, embedder, dataset and decoding. One shared tokenizer-backed counter
measures every policy's budget, so the tokens a policy is charged for are
exactly the tokens the model reads.

## 2. ΔContext: storing a chat as topic segments

ΔContext (`contextbench/policies/delta_context.py`) stores the chat as it
arrives, online and turn by turn, as a sequence of **topic segments**, and
retrieves topic-first.

**Per turn** (`observe`), with e_t the L2-normalised embedding of turn t
(all-MiniLM-L6-v2):

1. State: C_t = mean(e_{t-w+1..t}), a windowed centroid (w = 3).
2. Displacement from the current topic's reference state:
   A_t = 1 − cos(C_ref, C_t), where C_ref is the state at the last checkpoint.
3. Checkpoint when A_t ≥ τ and at least g turns have passed since the last
   one (g = 3). τ is tuned on dev data (§6).
4. **Boundary localisation.** C_t only crosses τ once most of its window holds
   the new topic, so checkpoints fire w − 1 turns late. The new segment instead
   starts at the turn among the last w with the largest single-turn jump,
   argmax_i (1 − cos(e_{i−1}, e_i)).
5. Each segment's centroid (mean of its turn embeddings) is its index key.

Cost per turn is O(d) (one embedding, one distance); nothing is recomputed.

**Per question** (`build_context`, `topic_first_k = k`):

1. Stage 1: rank segments by cos(q, centroid) and keep the top k.
2. Stage 2: rank only the turns inside those k segments by cos(q, e_t).
3. Fill the budget greedily in that order, skipping turns that do not fit;
   if the chosen topics run out, continue with the remaining turns by
   similarity. Selected turns are rendered chronologically, each prefixed with
   its session date.

Query-time cost is (#segments + #turns in the top-k segments) similarity
computations, instead of one per turn for a full scan.

Ablations: no localisation (segment boundary at the checkpoint turn); the
original path-length detector (A_t = Σ δ since the last checkpoint, τ = 2.0).

## 3. Baselines

All embedding-based methods use the same MiniLM turn embeddings.

| Method | What it keeps | Code |
|---|---|---|
| Sliding window | newest turns that fit | `policies/sliding_window.py` |
| Extractive summary | 30% budget for truncated pointers to dropped turns, rest newest turns | `policies/summary.py` |
| RAG + recency tail (50/50) | half the budget for top-similarity turns, half for newest turns (the original configuration) | `policies/retrieval.py` |
| RAG, turn-level | whole budget for top-similarity turns (full scan) | `policies/retrieval.py`, `retrieval_fraction=1.0, packing=skip` |
| Topic store, fixed-size chunks | same two-stage search over chunks of N turns | `segment_source=fixed` |
| Topic store, adjacent threshold | online cut when 1 − cos(e_{t−1}, e_t) > θ | `segment_source=adjacent` |
| Topic store, semantic chunking | offline cut where the consecutive distance exceeds its p-th percentile in the history (as in LangChain's SemanticChunker) | `segment_source=semantic_chunker` |
| Topic store, TextTiling | offline TextTiling (Hearst 1997) over embeddings: block similarity, depth scores, cutoff mean − σ/2 | `segment_source=texttiling` |
| Topic store, session oracle | dataset session boundaries; an upper reference for segmentation | `segment_source=session` |
| Full context | the whole conversation (LoCoMo only; it fits Qwen's 32K window) | `sliding_window`, budget 32000 |

Segmenters: `contextbench/context/segmenters.py`. Topic-first search is the same
function for every store (`contextbench/policies/ranking.py::rank_topic_first`),
so stores differ only in where they cut.

## 4. Datasets

**LoCoMo** (Maharana et al., 2024): 10 two-person conversations, 19–32
sessions each, 369–689 turns, 1,986 QA probes. Categories: multi-hop (282),
temporal (321), open-domain (96), single-hop (841), adversarial (446).

**LongMemEval-S** (Wu et al., ICLR 2025): 500 questions, each with its own
history of ~48 sessions (median 491 turns, ~115K tokens) on mostly unrelated
topics. 479 questions carry turn-level evidence labels; 30 are abstention
questions. Types: single-session-user/assistant/preference, multi-session,
temporal-reasoning, knowledge-update.

### Evaluation corrections (v2)

The original LoCoMo setup mis-scored 38% of probes for every policy:

- **Adversarial probes** name the wrong speaker on purpose; the field
  `adversarial_answer` is the trap answer, and v1 graded F1 against it. v2
  scores them on abstention (the model says the conversation does not contain
  the answer), as LoCoMo's own evaluation does, and reports them separately.
  Headline QA metrics are over answerable probes only.
- **Session dates** were never shown to the model, so temporal questions were
  unanswerable. v2 prefixes each transcript line with its session date; the
  date tokens count against the budget.
- A short-answer prompt; v1's free-form answers scored EM = 0 on all probes.

Config: `configs/experiment/locomo_v2.yaml`, `longmemeval_v2.yaml` (adds the
date each question is asked, which LongMemEval's temporal questions need).

## 5. Model and budgets

Generator: Qwen2.5-3B-Instruct, bfloat16, greedy decoding, ≤ 64 new tokens.
Budgets are counted with Qwen's tokenizer. Embedder: all-MiniLM-L6-v2 (CPU).
LoCoMo QA at B = 512; LongMemEval QA at B = 2048; selection-only sweeps at
256–4096. Hardware: one RTX 4060 Laptop GPU (8 GB), Windows 11. Long prompts
use a prefix-cached, chunk-prefilled backend (`hf_prefix_cached`) whose outputs
match plain generation exactly on test prompts.

## 6. Metrics

**End-to-end QA** (answerable probes): token F1 and exact match after SQuAD
normalisation; abstention rate on adversarial / abstention probes.

**Context selection** (no model needed): *evidence recall*, the share of a
probe's labelled evidence turns present in the selected context. It scores the
policy itself, independent of how well the model reads.

**Topic store** (`scripts/topic_store_report.py`, `paper_tables.py`):
- *topic hit@k*: an evidence turn lies inside one of the k best topics;
- *topic recall*: evidence turns in the context *and* inside the searched
  topics, i.e. what a store searching only those topics keeps;
- *searched*: query-time comparisons as a share of a full scan;
- *purity*: share of the restored context from the evidence's own session.

**Segmentation** against session boundaries (`contextbench/evaluation/segmentation.py`):
Pk (Beeferman et al., 1999) and WindowDiff (Pevzner & Hearst, 2002) with
k = half the mean reference segment length; boundary precision / recall / F1
with one-to-one matching at tolerance 0 and 1 turn.

## 7. Tuning protocol

Histories are split in dataset order: the first half is **dev**, the second
**test**. Every method's single parameter (ΔContext's τ; chunk size N; θ; p;
TextTiling block size) is chosen on LongMemEval dev by mean boundary F1 at
tolerance 1 (`scripts/segmentation_report.py`; chosen values in
`results/seg_lme.json`). The same values are then used everywhere, including on
LoCoMo, without re-tuning. Reported segmentation numbers are on test
histories; retrieval and QA tables report all questions, with dev/test
breakdowns available from `scripts/summarize_runs.py`.

## 8. Statistics

All comparisons are paired on the same probes. Continuous metrics: paired
bootstrap of the mean difference (10,000 resamples), 95% CI and two-sided p.
Binary metrics (topic hit@k): exact McNemar test. p-values are Holm–Bonferroni
corrected within each table. Probes inside one LoCoMo conversation are not
independent (10 conversations is too few for a cluster bootstrap), so LoCoMo
p-values are likely optimistic; LongMemEval questions have separate histories.

## 9. Reproducing

```bash
python scripts/precompute_embeddings.py configs/dataset/longmemeval_s.yaml data/cache/longmemeval_s_minilm.npz --device cuda
python scripts/segmentation_report.py configs/dataset/longmemeval_s.yaml --cache data/cache/longmemeval_s_minilm.npz --out paper/results/seg_lme.json
python scripts/sweep.py ...            # grids in the sweep commands recorded in RESULTS.md
python scripts/scaling_report.py --out paper/results/scaling_lme.json
python scripts/paper_tables.py --out paper
python scripts/paper_figures.py --out paper/figures --example
```

## 10. Limitations to state

- LongMemEval's official metric is an LLM-judge (GPT-4o); we report F1/EM and
  evidence recall, which favour short exact answers.
- One generator size (3B) and one small embedder; absolute QA scores would
  rise with larger models.
- LoCoMo sessions are days of one continuing chat, not topics, so it is not a
  topic-segmentation benchmark; LongMemEval is.
- The scaling study grows stores with other questions' histories (unrelated
  topics), which favours topic segmentation more than a single user's
  long-lived chat might.
