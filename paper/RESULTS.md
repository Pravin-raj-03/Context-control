# Results: ΔContext as topic-structured conversational memory

Setup, baselines, metrics and statistics: [PROTOCOL.md](PROTOCOL.md). Every
number below is regenerated from run outputs by `scripts/paper_tables.py`
(full tables with CIs: [RESULTS_TABLES.md](RESULTS_TABLES.md), LaTeX in
`tables/`) and `scripts/paper_figures.py` (`figures/`). Unless stated,
p-values are Holm-corrected within their table; "n.s." means not significant
after correction.

## Suggested framing

**ΔContext: online semantic-change segmentation for topic-structured
conversational memory.** A chat is stored, as it arrives, as topic segments
found from the rate of semantic change; questions are answered by searching
topics first and then the turns inside the best few. The evidence supports
four contributions and argues against a fifth that a reader might expect:

1. **A corrected LoCoMo evaluation.** The common setup mis-scores 38% of
   probes for every policy (§1).
2. **The best topic segmenter tested, and it runs online** (§2), with two
   design choices that each matter (§3).
3. **Topic-first retrieval over ΔContext segments** finds the right topic and
   keeps ~93% of full-scan evidence recall while searching ~17% of the turns
   (§4), and **tracks an oracle segmentation as the store grows to ~20K
   turns** while fixed chunks and semantic chunking fall behind (§5).
4. **What does not work** (§7): ΔContext does not beat scanning every turn on
   recall, and it is not significantly better than every semantic segmenter
   at retrieval on a single history.
5. *Not supported:* that ΔContext improves end-to-end answer quality over
   turn-level retrieval (§6, pending final runs).

## 1. The LoCoMo evaluation was mis-scoring 38% of probes

| Problem in the common setup | Probes affected | Fix (`experiment=locomo_v2`) |
|---|---|---|
| Adversarial probes graded against `adversarial_answer`, the trap answer (the other speaker's fact) | 446 (22%) | score abstention, report separately, headline over answerable probes |
| Session dates never shown, so "when" questions are unanswerable | 321 (16%) | date prefix on each line, counted against the budget |
| Free-form answers; exact match 0.0 on every probe | all | short-answer prompt |

With TinyLlama under the old setup, F1 was 0.06–0.12 for every policy, and
retrieval and ΔContext answered 1,214 of 1,986 probes word-for-word
identically. Under v2 with Qwen2.5-3B, turn-level retrieval reaches F1 0.259
on answerable probes (§6).

## 2. Segmentation quality (LongMemEval-S, 250 test histories)

Reference: session boundaries (47.7 segments per history). Each method's
parameter tuned on the 250 dev histories by boundary F1 (`results/seg_lme.json`).

| Method | Online | Pk ↓ | WindowDiff ↓ | F1 (±1 turn) | F1 (exact) |
|---|---|---|---|---|---|
| **ΔContext** (τ = 0.75) | yes | **0.099** | **0.130** | **0.840** | **0.828** |
| Adjacent-turn threshold (θ = 0.85) | yes | 0.125 | 0.179 | 0.778 | 0.771 |
| Semantic chunking (p = 90) | no | 0.140 | 0.185 | 0.769 | 0.759 |
| TextTiling (block = 5) | no | 0.212 | 0.257 | 0.774 | 0.768 |
| Fixed-size chunks (N = 5) | yes | 0.534 | 0.572 | 0.375 | 0.127 |

ΔContext is better than every baseline on every metric (paired bootstrap over
histories, all p < 0.0001; e.g. F1 +0.062 [+0.054, +0.070] vs the next best),
while deciding boundaries turn by turn, without seeing the rest of the history
the two offline methods use. Figure: `figures/segmentation_example.pdf` (the
first test history, not selected).

On LoCoMo every method is near chance (Pk ≈ 0.5, `results/seg_locomo.json`):
its sessions are days of one continuing conversation, not topics. LoCoMo is
not a segmentation benchmark and belongs in an appendix.

## 3. Ablations: both design changes are necessary

| ΔContext variant | Seg. F1 (±1) | Topic hit@1 | Topic hit@3 | Topic recall |
|---|---|---|---|---|
| **Full** (displacement detector + boundary localisation) | **0.840** | **0.749** | **0.917** | **0.716** |
| No localisation | 0.170 | 0.426 | 0.732 | 0.516 |
| Original path-length detector | 0.208 | 0.640 | 0.847 | 0.609 |

All differences p < 0.001. Without localisation, checkpoints fire w − 1 = 2
turns after the switch, so exact-boundary F1 falls to 0.035 and each new
segment opens with turns of the old topic.

## 4. Topic store on one history (LongMemEval-S, budget 2048, top-3 topics)

| How the history is stored | hit@1 | hit@3 | Topic recall | Turns searched |
|---|---|---|---|---|
| **ΔContext segments** | **0.749** | **0.917** | **0.716** | 17.0% |
| Session oracle (dataset metadata) | 0.785 | 0.926 | 0.733 | 16.5% |
| Adjacent threshold | 0.728 | 0.900 | 0.699 | 19.1% |
| Semantic chunking | 0.721 | 0.904 | 0.704 | 18.4% |
| TextTiling | 0.711 | 0.894 | 0.696 | 19.1% |
| Fixed chunks, 10 turns | 0.543*** | 0.813*** | 0.601*** | 16.2% |
| Fixed chunks, 5 turns | 0.457*** | 0.736*** | 0.534*** | 23.1% |
| No topics: search every turn | – | – | 0.774*** | 100% |

- Semantic segmentation of any kind beats fixed chunks by 11–29 points (p < 0.001).
- ΔContext is numerically the best non-oracle store at every k (1–5) and
  every cost point (`figures/topic_hits.pdf`, `figures/cost_recall.pdf`), but
  on a single history its lead over semantic chunking, TextTiling and the
  adjacent threshold is 0–4 points and **n.s.** after correction (all 27
  per-k differences ≥ 0, two exact ties; only vs TextTiling is the mean
  hit@1–5 gap significant, +0.029, Holm p = 0.016).
- Searching only the top-3 ΔContext topics keeps 0.716 of evidence turns vs
  0.774 for a full scan (−0.058, p < 0.001) at 17% of the comparisons.

## 5. Scaling the store (LongMemEval-S, ~238 test questions)

The question's own history is placed after D other questions' histories.

| Turns in store | ΔContext hit@1 / hit@3 | Semantic chunking | TextTiling | Fixed chunks | Session oracle | ΔContext comparisons vs full scan |
|---|---|---|---|---|---|---|
| 492 | 0.782 / 0.924 | 0.727 / 0.908 | 0.731 / 0.903 | 0.496 / 0.748 | 0.815 / 0.933 | 83 / 492 |
| 1,969 | 0.685 / 0.828 | 0.588 / 0.790 | 0.664 / 0.836 | 0.424 / 0.697 | 0.706 / 0.861 | 237 / 1,969 |
| 4,912 | 0.576 / 0.739 | 0.479 / 0.689 | 0.546 / 0.744 | 0.345 / 0.563 | 0.588 / 0.765 | 542 / 4,912 |
| 9,862 | 0.492 / 0.647 | 0.408 / 0.584 | 0.479 / 0.647 | 0.273 / 0.504 | 0.525 / 0.660 | 1,056 / 9,862 |
| 19,725 | 0.420 / 0.597 | 0.345 / 0.513 | 0.399 / 0.605 | 0.290 / 0.475 | 0.441 / 0.618 | 2,080 / 19,725 |

- ΔContext stays within 1–3.3 points of the session oracle at every size
  (none significant after correction; the largest, hit@3 at ~2K turns, has
  uncorrected p = 0.04) while searching ~10.6% of the store.
- Semantic chunking falls progressively behind (hit@1 −0.08 to −0.10 from 4×
  up; exact McNemar p = 0.001–0.008 per size before correction; n.s. after
  Holm over all 50 tests in the table). Fixed chunks lose significantly
  everywhere.
- TextTiling matches ΔContext's hit rates but is offline and its segments
  need 30–38% more comparisons from ~2K turns up (1,448 vs 1,056 at ~9.9K).
- Figure: `figures/scaling.pdf`.

## 6. End-to-end QA (Qwen2.5-3B-Instruct) — partial; final runs pending

LoCoMo, budget 512, answerable probes (n = 1,540):

| Policy | F1 | EM | Evidence recall | Abstains on adversarial |
|---|---|---|---|---|
| RAG + chunk prior | 0.267 | – | – | 0.614 |
| RAG, turn-level (full scan) | 0.259 | 0.081 | 0.475 | 0.646 |
| ΔContext v1.2 turn ranking (50/50 split) | 0.228 | – | – | 0.673 |
| RAG + recency tail (50/50, original config) | 0.223 | 0.073 | 0.374 | 0.688 |
| Sliding window | 0.026 | 0.006 | 0.011 | 0.975 |
| Extractive summary | 0.026 | – | – | 0.971 |

Pending: ΔContext / fixed-chunk / semantic-chunking topic stores on LoCoMo
(B = 512); all LongMemEval QA runs (B = 2048, `experiment=longmemeval_v2`); the
LoCoMo full-context upper bound. `RESULTS_TABLES.md` will carry the complete
tables with significance once they finish.

## 7. Negative results the paper should state

- **Full scan wins on recall.** Turn-level search over every turn keeps more
  evidence than any topic store at equal budget (LongMemEval 0.774 vs 0.716).
  The topic store's case is cost and structure, not recall.
- **Single-history retrieval does not separate good segmenters.** ΔContext's
  segmentation advantage (§2) shows up in retrieval only as small, mostly
  non-significant gains until the store grows (§5).
- **ΔContext as a turn-ranking prior did not help.** On LoCoMo a segment
  prior added +2–3 points of evidence recall, but fixed chunks gave the same
  gain; on LongMemEval no prior beat plain retrieval.
- **Topic detection is the wrong signal for budget splitting.** For questions
  asked mid-conversation, whether the best match is recent separates "about
  now" from "about earlier" questions far better (AUC 0.85) than the
  question's similarity to the current topic (AUC 0.63).
- **LoCoMo is not a topic benchmark**, and ΔContext's LongMemEval-tuned
  threshold makes very large segments there (top-3 topics cover ~50% of a
  conversation), which confounds LoCoMo topic-store comparisons.

## 8. Figures

| File | Shows |
|---|---|
| `figures/segmentation_example.pdf` | boundaries of each method vs true sessions on one test history |
| `figures/cost_recall.pdf` | turns searched vs evidence recall, top-1/3/5 topics, plus full scan |
| `figures/topic_hits.pdf` | topic hit@k, k = 1–5 |
| `figures/scaling.pdf` | topic hit@3 as the store grows to ~20K turns |
