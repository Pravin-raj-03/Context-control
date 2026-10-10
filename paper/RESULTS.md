# Results: ΔContext as topic-structured conversational memory (LongMemEval-S)

Setup, baselines, metrics and statistics: [PROTOCOL.md](PROTOCOL.md). Every
number below is regenerated from run outputs by `scripts/paper_tables.py`
(full tables with CIs: [RESULTS_TABLES.md](RESULTS_TABLES.md), LaTeX in
`tables/`) and `scripts/paper_figures.py` (`figures/`). Unless stated,
p-values are Holm-corrected within their table; "n.s." means not significant
after correction. The paper (`main.tex`) uses LongMemEval-S only.

## Framing

**ΔContext: online semantic-change segmentation for topic-structured
conversational memory.** A chat is stored, as it arrives, as topic segments
found from the rate of semantic change; questions are answered by searching
topics first and then the turns inside the best few.

Supported claims:

1. **The best topic segmenter tested, and it runs online** (§1), with two
   design choices that each matter (§2).
2. **Topic-first retrieval over ΔContext segments** finds the right topic and
   keeps ~93% of full-scan evidence recall while searching ~17% of the turns
   (§3), and **tracks an oracle segmentation as the store grows to ~20K
   turns** while fixed chunks and semantic chunking fall behind (§4).
3. **Where it does not help** (§6): it does not beat scanning every turn on
   recall, and on a single history it is not significantly better than every
   semantic segmenter at retrieval.

4. **End-to-end QA parity with a full scan** at ~17% of the search (§5).

## 1. Segmentation quality (250 test histories)

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
histories, all p < 0.0001; F1 +0.062 [+0.054, +0.070] vs the next best), while
deciding boundaries turn by turn. Figure: `figures/segmentation_example.pdf`
(the first test history, not selected).

## 2. Ablations: both design changes are necessary

| ΔContext variant | Seg. F1 (±1) | Topic hit@1 | Topic hit@3 | Topic recall |
|---|---|---|---|---|
| **Full** (displacement detector + boundary localisation) | **0.840** | **0.749** | **0.917** | **0.716** |
| No localisation | 0.170 | 0.426 | 0.732 | 0.516 |
| Path-length detector | 0.208 | 0.640 | 0.847 | 0.609 |

All differences p < 0.001.

## 3. Topic store on one history (budget 2048, top-3 topics, 470 questions)

| How the history is stored | hit@1 | hit@3 | Topic recall | Turns searched |
|---|---|---|---|---|
| **ΔContext segments** | **0.749** | **0.917** | **0.716** | 17.0% |
| Session oracle | 0.785 | 0.926 | 0.733 | 16.5% |
| Adjacent threshold | 0.728 | 0.900 | 0.699 | 19.1% |
| Semantic chunking | 0.721 | 0.904 | 0.704 | 18.4% |
| TextTiling | 0.711 | 0.894 | 0.696 | 19.1% |
| Fixed chunks, 10 turns | 0.543*** | 0.813*** | 0.601*** | 16.2% |
| Fixed chunks, 5 turns | 0.457*** | 0.736*** | 0.534*** | 23.1% |
| No topics: search every turn | – | – | 0.774*** | 100% |

- Semantic segmentation of any kind beats fixed chunks by 11–29 points (p < 0.001).
- ΔContext is numerically the best non-oracle store at every k (1–5) and every
  cost point (`figures/topic_hits.pdf`, `figures/cost_recall.pdf`); all 27
  per-k differences against semantic chunking, TextTiling and the adjacent
  threshold are ≥ 0 (two ties), but only the TextTiling gap (mean hit@1–5,
  +0.029, Holm p = 0.016) is significant.
- Searching only the top-3 ΔContext topics keeps 0.716 of evidence turns vs
  0.774 for a full scan (−0.058, p < 0.001) at 17% of the comparisons.

## 4. Scaling the store (~238 test questions)

| Turns in store | ΔContext hit@1 / hit@3 | Semantic chunking | TextTiling | Fixed chunks | Session oracle | ΔContext comparisons vs full scan |
|---|---|---|---|---|---|---|
| 492 | 0.782 / 0.924 | 0.727 / 0.908 | 0.731 / 0.903 | 0.496 / 0.748 | 0.815 / 0.933 | 83 / 492 |
| 1,969 | 0.685 / 0.828 | 0.588 / 0.790 | 0.664 / 0.836 | 0.424 / 0.697 | 0.706 / 0.861 | 237 / 1,969 |
| 4,912 | 0.576 / 0.739 | 0.479 / 0.689 | 0.546 / 0.744 | 0.345 / 0.563 | 0.588 / 0.765 | 542 / 4,912 |
| 9,862 | 0.492 / 0.647 | 0.408 / 0.584 | 0.479 / 0.647 | 0.273 / 0.504 | 0.525 / 0.660 | 1,056 / 9,862 |
| 19,725 | 0.420 / 0.597 | 0.345 / 0.513 | 0.399 / 0.605 | 0.290 / 0.475 | 0.441 / 0.618 | 2,080 / 19,725 |

- ΔContext stays within 1–3.3 points of the session oracle at every size
  (none significant after correction; the largest, hit@3 at ~2K turns, has
  uncorrected p = 0.04) while searching 11–17% of the store (10.5% at ~20K turns).
- Semantic chunking falls progressively behind (hit@1 −0.08 to −0.10 from ~2K
  turns up; exact McNemar p = 0.001–0.008 per size before correction; n.s.
  after Holm over all 50 tests). Fixed chunks lose significantly everywhere.
- TextTiling matches ΔContext's hit rates but is offline and needs 30–38% more
  comparisons from ~2K turns up.

## 5. End-to-end QA (Qwen2.5-3B-Instruct, budget 2048, 470 answerable questions)

| Policy | F1 | EM | Evidence recall | Abstains (30 abstention Qs) |
|---|---|---|---|---|
| RAG, full scan | 0.311 | 0.217 | 0.774*** | 0.933 |
| Session-oracle topic store | 0.298 | 0.202 | 0.734 | 1.000 |
| **ΔContext topic store** | **0.295** | **0.209** | **0.717** | **0.967** |
| Semantic-chunking topic store | 0.284 | 0.194 | 0.717 | 0.933 |
| Fixed 10-turn topic store | 0.263 | 0.187 | 0.601*** | 1.000 |
| Sliding window | 0.019*** | 0.006*** | 0.009*** | 1.000 |
| Evidence sessions only (~6K tokens, outside the budget) | 0.279 | 0.202 | 1.000 | 0.933 |

- ΔContext vs full scan: F1 −0.016 [−0.036, +0.005], n.s.; EM −0.009 n.s.;
  evidence recall −0.057 (p < 0.001). Same answer quality at 17% of the search.
- ΔContext vs semantic chunking +0.011 F1, vs session oracle −0.003 (both n.s.);
  vs fixed 10-turn chunks +0.032 [+0.008, +0.058] (n.s. after Holm).
- The evidence-sessions-only context is not an upper bound for a 3B model:
  ~6K tokens of context score below 2K tokens of retrieved turns.

## 6. What did not help

- **Full scan wins on recall.** Searching every turn keeps more evidence than
  any topic store at equal budget (0.774 vs 0.716). The topic store's case is
  cost and structure, not recall.
- **On a single history, better segmentation barely helps retrieval.** The
  segmentation lead (§1) becomes small, mostly non-significant retrieval gains
  until the store grows (§4).
- **Segments as a ranking prior.** Adding a weighted segment similarity to
  turn ranking never improved on plain retrieval (0.774 without a prior,
  0.740–0.774 across priors, including oracle sessions).

## Outside the paper's scope

LoCoMo was dropped from the paper (its sessions are consecutive days of one
conversation, with no topic structure). The paper's Limitations keep one
sentence on it: the topic store gave no gain over a full scan there (F1 0.220
vs 0.259, budget 512). Those runs remain under `experiments/` and can be
tabulated with `python scripts/paper_tables.py --include-locomo`.

## Figures

| File | Shows |
|---|---|
| `figures/segmentation_example.pdf` | boundaries of each method vs true sessions on one test history |
| `figures/cost_recall.pdf` | turns searched vs evidence recall, top-1/3/5 topics, plus full scan |
| `figures/topic_hits.pdf` | topic hit@k, k = 1–5 |
| `figures/scaling.pdf` | topic hit@3 as the store grows to ~20K turns |
