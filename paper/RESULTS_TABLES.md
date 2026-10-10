# ContextBench / ΔContext: result tables

Answerable probes only; significance markers: * p<0.05, ** p<0.01, *** p<0.001 (ΔContext vs that row, Holm-adjusted within the table).

### LongMemEval-S topic store (budget 2048)

| Method | hit@1 | hit@3 | topic_recall | evidence_recall | searched |
|---|---|---|---|---|---|
| **ΔContext store (top-3)** | 0.749 | 0.917 | 0.716 | 0.717 | 17.0 |
| Session oracle store (top-3) | 0.785 | 0.926 | 0.733 | 0.734 | 16.5 |
| Adjacent threshold store (top-3) | 0.728 | 0.900 | 0.699 | 0.708 | 19.1 |
| Semantic chunking store (top-3) | 0.721 | 0.904 | 0.704 | 0.717 | 18.4 |
| TextTiling store (top-3) | 0.711 | 0.894 | 0.696 | 0.711 | 19.1 |
| Fixed-size chunks (5 turns) store (top-3) | 0.457*** | 0.736*** | 0.534*** | 0.581*** | 23.1 |
| Fixed-size chunks (10 turns) store (top-3) | 0.543*** | 0.813*** | 0.601*** | 0.601*** | 16.2 |
| RAG, turn-level | -- | -- | 0.774*** | 0.774*** | 100.0 |

ΔContext (ΔContext store (top-3)) minus each baseline; Holm-adjusted p within this table:
- vs Adjacent threshold store (top-3), evidence_recall: +0.010 [-0.012, +0.031], p_adj=1.0000
- vs Adjacent threshold store (top-3), hit@1: +0.021 (McNemar), p_adj=1.0000
- vs Adjacent threshold store (top-3), hit@3: +0.017 (McNemar), p_adj=1.0000
- vs Adjacent threshold store (top-3), topic_recall: +0.017 [-0.003, +0.038], p_adj=0.9580
- vs Fixed-size chunks (10 turns) store (top-3), evidence_recall: +0.116 [+0.086, +0.147], p_adj=0.0000
- vs Fixed-size chunks (10 turns) store (top-3), hit@1: +0.206 (McNemar), p_adj=0.0000
- vs Fixed-size chunks (10 turns) store (top-3), hit@3: +0.104 (McNemar), p_adj=0.0000
- vs Fixed-size chunks (10 turns) store (top-3), topic_recall: +0.115 [+0.085, +0.146], p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), evidence_recall: +0.136 [+0.101, +0.173], p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), hit@1: +0.291 (McNemar), p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), hit@3: +0.181 (McNemar), p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), topic_recall: +0.183 [+0.149, +0.218], p_adj=0.0000
- vs RAG, turn-level, evidence_recall: -0.057 [-0.078, -0.036], p_adj=0.0000
- vs RAG, turn-level, topic_recall: -0.058 [-0.079, -0.037], p_adj=0.0000
- vs Semantic chunking store (top-3), evidence_recall: +0.000 [-0.021, +0.021], p_adj=1.0000
- vs Semantic chunking store (top-3), hit@1: +0.028 (McNemar), p_adj=1.0000
- vs Semantic chunking store (top-3), hit@3: +0.013 (McNemar), p_adj=1.0000
- vs Semantic chunking store (top-3), topic_recall: +0.012 [-0.009, +0.033], p_adj=1.0000
- vs Session oracle store (top-3), evidence_recall: -0.017 [-0.031, -0.002], p_adj=0.3120
- vs Session oracle store (top-3), hit@1: -0.036 (McNemar), p_adj=0.1221
- vs Session oracle store (top-3), hit@3: -0.009 (McNemar), p_adj=1.0000
- vs Session oracle store (top-3), topic_recall: -0.017 [-0.031, -0.002], p_adj=0.3120
- vs TextTiling store (top-3), evidence_recall: +0.007 [-0.015, +0.029], p_adj=1.0000
- vs TextTiling store (top-3), hit@1: +0.038 (McNemar), p_adj=0.6425
- vs TextTiling store (top-3), hit@3: +0.023 (McNemar), p_adj=0.5627
- vs TextTiling store (top-3), topic_recall: +0.020 [-0.002, +0.043], p_adj=0.7744

### LongMemEval-S topic store ablation (budget 2048)

| Method | hit@1 | hit@3 | topic_recall |
|---|---|---|---|
| **ΔContext store (top-3)** | 0.749 | 0.917 | 0.716 |
| ΔContext store (top-3) (no localization) | 0.426*** | 0.732*** | 0.516*** |
| ΔContext store (top-3) (path-length detector) | 0.640*** | 0.847*** | 0.609*** |

ΔContext (ΔContext store (top-3)) minus each baseline; Holm-adjusted p within this table:
- vs ΔContext store (top-3) (no localization), hit@1: +0.323 (McNemar), p_adj=0.0000
- vs ΔContext store (top-3) (no localization), hit@3: +0.185 (McNemar), p_adj=0.0000
- vs ΔContext store (top-3) (no localization), topic_recall: +0.200 [+0.165, +0.236], p_adj=0.0000
- vs ΔContext store (top-3) (path-length detector), hit@1: +0.109 (McNemar), p_adj=0.0000
- vs ΔContext store (top-3) (path-length detector), hit@3: +0.070 (McNemar), p_adj=0.0000
- vs ΔContext store (top-3) (path-length detector), topic_recall: +0.108 [+0.077, +0.141], p_adj=0.0000

### LongMemEval-S end-to-end QA (Qwen2.5-3B) (budget 2048)

| Method | f1 | em | evidence_recall | context_tokens | Abstain |
|---|---|---|---|---|---|
| Sliding window | 0.019*** | 0.006*** | 0.009*** | 1809 | 1.000 |
| RAG, turn-level | 0.311 | 0.217 | 0.774*** | 2037 | 0.933 |
| **ΔContext store (top-3)** | 0.295 | 0.209 | 0.717 | 2038 | 0.967 |
| Session oracle store (top-3) | 0.298 | 0.202 | 0.734 | 2037 | 1.000 |
| Semantic chunking store (top-3) | 0.284 | 0.194 | 0.717 | 2037 | 0.933 |
| Fixed-size chunks (10 turns) store (top-3) | 0.263 | 0.187 | 0.601*** | 2039 | 1.000 |
| Evidence sessions only (oracle) | 0.279 | 0.202 | 1.000 | 5987 | 0.933 |

ΔContext (ΔContext store (top-3)) minus each baseline; Holm-adjusted p within this table:
- vs Fixed-size chunks (10 turns) store (top-3), em: +0.021 [-0.004, +0.047], p_adj=0.9744
- vs Fixed-size chunks (10 turns) store (top-3), evidence_recall: +0.116 [+0.086, +0.147], p_adj=0.0000
- vs Fixed-size chunks (10 turns) store (top-3), f1: +0.032 [+0.008, +0.058], p_adj=0.0940
- vs RAG, turn-level, em: -0.009 [-0.028, +0.011], p_adj=1.0000
- vs RAG, turn-level, evidence_recall: -0.057 [-0.078, -0.036], p_adj=0.0000
- vs RAG, turn-level, f1: -0.016 [-0.036, +0.005], p_adj=0.9744
- vs Semantic chunking store (top-3), em: +0.015 [-0.006, +0.036], p_adj=1.0000
- vs Semantic chunking store (top-3), evidence_recall: +0.000 [-0.021, +0.021], p_adj=1.0000
- vs Semantic chunking store (top-3), f1: +0.011 [-0.011, +0.033], p_adj=1.0000
- vs Session oracle store (top-3), em: +0.006 [-0.009, +0.021], p_adj=1.0000
- vs Session oracle store (top-3), evidence_recall: -0.017 [-0.031, -0.002], p_adj=0.1872
- vs Session oracle store (top-3), f1: -0.003 [-0.018, +0.012], p_adj=1.0000
- vs Sliding window, em: +0.202 [+0.166, +0.238], p_adj=0.0000
- vs Sliding window, evidence_recall: +0.709 [+0.674, +0.743], p_adj=0.0000
- vs Sliding window, f1: +0.276 [+0.239, +0.313], p_adj=0.0000

### Segmentation quality: configs\dataset\longmemeval_s.yaml (test histories: 250, parameters tuned on dev)

| Method | Pk | WindowDiff | P@1 | R@1 | F1@1 | F1@0 | Segments |
|---|---|---|---|---|---|---|---|
| fixed | 0.534 | 0.572 | 0.277 | 0.581 | 0.375 | 0.127 | 98.9 |
| adjacent | 0.125 | 0.179 | 0.722 | 0.854 | 0.778 | 0.771 | 57.4 |
| semantic_chunker | 0.140 | 0.185 | 0.748 | 0.792 | 0.769 | 0.759 | 50.5 |
| texttiling | 0.212 | 0.257 | 0.646 | 0.969 | 0.774 | 0.768 | 71.5 |
| delta_context | 0.099 | 0.130 | 0.813 | 0.870 | 0.840 | 0.828 | 51.1 |
| delta_context_unlocalized | 0.444 | 0.477 | 0.164 | 0.177 | 0.170 | 0.035 | 51.0 |
| delta_context_sum | 0.458 | 0.467 | 0.292 | 0.161 | 0.208 | 0.054 | 26.8 |

### Scaling: LongMemEval-S store grown with distractor histories

Topic hit@1 / hit@3 (comparisons per question). Stars: exact McNemar on hit@3 vs ΔContext, Holm-adjusted over the whole table.

| Turns | ΔContext | Fixed-size chunks | Semantic chunking | TextTiling | Adjacent threshold | Session oracle | Full scan |
|---|---|---|---|---|---|---|---|
| 492 | 0.782 / 0.924 (83) | 0.496 / 0.748*** (114) | 0.727 / 0.908 (91) | 0.731 / 0.903 (93) | 0.748 / 0.891 (93) | 0.815 / 0.933 (81) | (492) |
| 1969 | 0.685 / 0.828 (237) | 0.424 / 0.697*** (409) | 0.588 / 0.790 (238) | 0.664 / 0.836 (308) | 0.634 / 0.807 (264) | 0.706 / 0.861 (224) | (1969) |
| 4912 | 0.576 / 0.739 (542) | 0.345 / 0.563*** (998) | 0.479 / 0.689 (530) | 0.546 / 0.744 (735) | 0.521 / 0.714 (610) | 0.588 / 0.765 (509) | (4912) |
| 9862 | 0.492 / 0.647 (1056) | 0.273 / 0.504*** (1988) | 0.408 / 0.584 (1024) | 0.479 / 0.647 (1448) | 0.445 / 0.609 (1194) | 0.525 / 0.660 (988) | (9862) |
| 19725 | 0.420 / 0.597 (2080) | 0.290 / 0.475* (3960) | 0.345 / 0.513 (2008) | 0.399 / 0.605 (2879) | 0.387 / 0.567 (2360) | 0.441 / 0.618 (1943) | (19725) |
