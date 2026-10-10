# Experimental protocol (LongMemEval-S)

This is the setup behind every number in `RESULTS.md`, `tables/` and
`main.tex`. Each section names the code that implements it. The paper uses
LongMemEval-S only.

## 1. Problem

A conversational model answers questions about a long chat history under a
fixed **context budget** B (tokens of history in the prompt). Each
LongMemEval-S history is ~115K tokens, far longer than B, so a *context
management policy* decides which past turns the model sees. Everything except
the policy is held fixed: model, tokenizer, prompt, embedder, dataset and
decoding. One shared tokenizer-backed counter measures every policy's budget,
so the tokens a policy is charged for are exactly the tokens the model reads.

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
   one (g = 3). τ is tuned on dev histories (§6).
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

Config: `policy=delta_context ++policy.landmark_fraction=1.0
++policy.accumulator=displacement ++policy.change_threshold=0.75
++policy.localize_boundaries=true ++policy.topic_first_k=3`.

## 3. Baselines

All embedding-based methods use the same MiniLM turn embeddings.

| Method | What it keeps | Code |
|---|---|---|
| Sliding window | newest turns that fit | `policies/sliding_window.py` |
| RAG, full scan | whole budget for the turns most similar to the question | `policies/retrieval.py`, `retrieval_fraction=1.0, packing=skip` |
| Topic store, fixed-size chunks | same two-stage search over chunks of N turns | `segment_source=fixed` |
| Topic store, adjacent threshold | online cut when 1 − cos(e_{t−1}, e_t) > θ | `segment_source=adjacent` |
| Topic store, semantic chunking | offline cut where the consecutive distance exceeds its p-th percentile in the history (as in LangChain's SemanticChunker) | `segment_source=semantic_chunker` |
| Topic store, TextTiling | offline TextTiling (Hearst 1997) over embeddings: block similarity, depth scores, cutoff mean − σ/2 | `segment_source=texttiling` |
| Topic store, session oracle | dataset session boundaries; an upper reference for segmentation | `segment_source=session` |
| Evidence sessions only (QA upper bound) | the whole LongMemEval *oracle* history, which keeps only the sessions containing the answer | `dataset=longmemeval_oracle`, `sliding_window`, budget 24000 |

Segmenters: `contextbench/context/segmenters.py`. Topic-first search is the same
function for every store (`contextbench/policies/ranking.py::rank_topic_first`),
so stores differ only in where they cut.

## 4. Data

**LongMemEval-S** (Wu et al., ICLR 2025; `xiaowu0162/longmemeval-cleaned`,
MIT): 500 questions, each with its own history of ~48 sessions (median 491
turns, ~115K tokens) on mostly unrelated topics. 479 questions carry
turn-level evidence labels (`has_answer`); 30 are abstention questions
(`_abs`). Types: single-session-user (70), single-session-assistant (56),
single-session-preference (30), multi-session (133), temporal-reasoning (133),
knowledge-update (78). Loader: `contextbench/datasets/longmemeval.py`.

**LongMemEval oracle**: the same questions with only the evidence sessions
(median ~6K, max ~23K tokens), used for the QA upper bound.

Prompt (`configs/experiment/longmemeval_v2.yaml`): each transcript line starts
with its session date (counted against the budget); the prompt states the date
the question is asked, which temporal questions are relative to; short answers;
"Not mentioned in the conversation" for unanswerable questions.

## 5. Model and budgets

Generator: Qwen2.5-3B-Instruct, bfloat16, greedy decoding, ≤ 64 new tokens.
Budgets are counted with Qwen's tokenizer. Embedder: all-MiniLM-L6-v2 (CPU),
with turn embeddings precomputed once (`scripts/precompute_embeddings.py`).
QA at B = 2048; topic-store and selection sweeps at 512–4096. Hardware: one
RTX 4060 Laptop GPU (8 GB), Windows 11. Generation uses a chunk-prefilled
backend (`hf_prefix_cached`, 512-token chunks) whose outputs match plain
generation exactly on test prompts; plain single-pass prefill of ~2.5K-token
prompts overflowed GPU memory on this hardware.

## 6. Tuning protocol

Histories are split in dataset order: the first 250 are **dev**, the last 250
**test**. Every method's single parameter is chosen on dev histories by mean
boundary F1 at tolerance 1 (`scripts/segmentation_report.py`; chosen values in
`results/seg_lme.json`): ΔContext τ = 0.75, fixed chunk N = 5 (10 is also
reported, as the stronger retrieval baseline), adjacent θ = 0.85, semantic
chunking p = 90, TextTiling block = 5. The same values are used in every
experiment. Segmentation and scaling numbers are on test histories; topic-store
and QA tables report all questions.

## 7. Metrics

**Segmentation** against session boundaries (`contextbench/evaluation/segmentation.py`):
Pk (Beeferman et al., 1999) and WindowDiff (Pevzner & Hearst, 2002) with
k = half the mean reference segment length; boundary precision / recall / F1
with one-to-one matching at tolerance 0 and 1 turn.

**Topic store** (`scripts/paper_tables.py`, `scripts/topic_store_report.py`):
- *topic hit@k*: an evidence turn lies inside one of the k best topics;
- *topic recall*: evidence turns in the context *and* inside the searched
  topics, i.e. what a store searching only those topics keeps;
- *searched*: query-time comparisons as a share of a full scan.

**Context selection**: *evidence recall*, the share of a question's labelled
evidence turns present in the selected context.

**End-to-end QA** (answerable questions): token F1 and exact match after SQuAD
normalisation; abstention rate on the 30 abstention questions.

**Scaling** (`scripts/scaling_report.py`): each test question's history placed
after D ∈ {0, 3, 9, 19, 39} other questions' histories; topic hit@1/@3,
evidence rank, comparisons per query.

## 8. Statistics

All comparisons are paired on the same questions. Continuous metrics: paired
bootstrap of the mean difference (10,000 resamples), 95% CI and two-sided p.
Binary metrics (topic hit@k): exact McNemar test. p-values are Holm–Bonferroni
corrected within each table (over all 50 tests in the scaling table). Each
LongMemEval question has its own history, so questions are independent.

## 9. Reproducing

```bash
python scripts/precompute_embeddings.py configs/dataset/longmemeval_s.yaml data/cache/longmemeval_s_minilm.npz --device cuda
python scripts/segmentation_report.py configs/dataset/longmemeval_s.yaml --cache data/cache/longmemeval_s_minilm.npz --out paper/results/seg_lme.json
# topic stores (selection only), e.g. ΔContext top-3 at budget 2048:
python scripts/sweep.py --jobs 1 --budgets 2048 --common model=qwen2.5-3b dataset=longmemeval_s experiment.generate=false experiment.output_dir=experiments/lme_topic --variant "policy=delta_context ++policy.landmark_fraction=1.0 ++policy.accumulator=displacement ++policy.change_threshold=0.75 ++policy.localize_boundaries=true ++policy.embedding_cache=data/cache/longmemeval_s_minilm.npz ++policy.topic_first_k=3"
# QA: same policies with ++model.backend=hf_prefix_cached experiment=longmemeval_v2 experiment.output_dir=experiments/lme_llm
python scripts/scaling_report.py --out paper/results/scaling_lme.json --questions 250
python scripts/paper_tables.py --out paper
python scripts/paper_figures.py --out paper/figures --example
```

## 10. Limitations to state

- One benchmark. LoCoMo was run but left out: its sessions are consecutive
  days of one conversation with no topic structure, and there the topic store
  gave no gain over a full scan (F1 0.220 vs 0.259, budget 512).
- LongMemEval's official metric is an LLM judge (GPT-4o); we report F1/EM and
  evidence recall, which favour short exact answers.
- One generator size (3B) and one small embedder; absolute QA scores would
  rise with larger models.
- The scaling study grows stores with other questions' histories (unrelated
  topics), which favours topic segmentation more than a single user's
  long-lived chat might.
