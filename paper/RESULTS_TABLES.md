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

### LoCoMo topic store (budget 2048)

| Method | hit@1 | hit@3 | topic_recall | evidence_recall | searched |
|---|---|---|---|---|---|
| **ΔContext store (top-3)** | 0.345 | 0.715 | 0.523 | 0.533 | 49.5 |
| Session oracle store (top-3) | 0.320 | 0.552*** | 0.483* | 0.487** | 16.6 |
| Adjacent threshold store (top-3) | 0.423*** | 0.831*** | 0.600*** | 0.600*** | 69.1 |
| Semantic chunking store (top-3) | 0.231*** | 0.412*** | 0.351*** | 0.633*** | 15.4 |
| TextTiling store (top-3) | 0.325 | 0.517*** | 0.460** | 0.715*** | 23.8 |
| Fixed-size chunks (5 turns) store (top-3) | 0.275*** | 0.482*** | 0.421*** | 0.720*** | 22.6 |
| Fixed-size chunks (10 turns) store (top-3) | 0.304* | 0.507*** | 0.447*** | 0.684*** | 15.2 |
| RAG, turn-level | -- | -- | 0.687*** | 0.687*** | 100.0 |

ΔContext (ΔContext store (top-3)) minus each baseline; Holm-adjusted p within this table:
- vs Adjacent threshold store (top-3), evidence_recall: -0.067 [-0.088, -0.045], p_adj=0.0000
- vs Adjacent threshold store (top-3), hit@1: -0.079 (McNemar), p_adj=0.0000
- vs Adjacent threshold store (top-3), hit@3: -0.116 (McNemar), p_adj=0.0000
- vs Adjacent threshold store (top-3), topic_recall: -0.077 [-0.099, -0.056], p_adj=0.0000
- vs Fixed-size chunks (10 turns) store (top-3), evidence_recall: -0.150 [-0.174, -0.127], p_adj=0.0000
- vs Fixed-size chunks (10 turns) store (top-3), hit@1: +0.041 (McNemar), p_adj=0.0232
- vs Fixed-size chunks (10 turns) store (top-3), hit@3: +0.207 (McNemar), p_adj=0.0000
- vs Fixed-size chunks (10 turns) store (top-3), topic_recall: +0.076 [+0.048, +0.103], p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), evidence_recall: -0.186 [-0.209, -0.164], p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), hit@1: +0.070 (McNemar), p_adj=0.0001
- vs Fixed-size chunks (5 turns) store (top-3), hit@3: +0.233 (McNemar), p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), topic_recall: +0.102 [+0.073, +0.130], p_adj=0.0000
- vs RAG, turn-level, evidence_recall: -0.154 [-0.176, -0.132], p_adj=0.0000
- vs RAG, turn-level, topic_recall: -0.164 [-0.186, -0.142], p_adj=0.0000
- vs Semantic chunking store (top-3), evidence_recall: -0.100 [-0.124, -0.076], p_adj=0.0000
- vs Semantic chunking store (top-3), hit@1: +0.114 (McNemar), p_adj=0.0000
- vs Semantic chunking store (top-3), hit@3: +0.302 (McNemar), p_adj=0.0000
- vs Semantic chunking store (top-3), topic_recall: +0.172 [+0.144, +0.199], p_adj=0.0000
- vs Session oracle store (top-3), evidence_recall: +0.047 [+0.019, +0.073], p_adj=0.0080
- vs Session oracle store (top-3), hit@1: +0.025 (McNemar), p_adj=0.1769
- vs Session oracle store (top-3), hit@3: +0.162 (McNemar), p_adj=0.0000
- vs Session oracle store (top-3), topic_recall: +0.040 [+0.012, +0.067], p_adj=0.0232
- vs TextTiling store (top-3), evidence_recall: -0.181 [-0.205, -0.158], p_adj=0.0000
- vs TextTiling store (top-3), hit@1: +0.020 (McNemar), p_adj=0.2179
- vs TextTiling store (top-3), hit@3: +0.198 (McNemar), p_adj=0.0000
- vs TextTiling store (top-3), topic_recall: +0.062 [+0.034, +0.090], p_adj=0.0012

### LoCoMo topic store ablation (budget 2048)

| Method | hit@1 | hit@3 | topic_recall |
|---|---|---|---|
| **ΔContext store (top-3)** | 0.345 | 0.715 | 0.523 |
| ΔContext store (top-3) (path-length detector) | 0.283*** | 0.498*** | 0.436*** |

ΔContext (ΔContext store (top-3)) minus each baseline; Holm-adjusted p within this table:
- vs ΔContext store (top-3) (path-length detector), hit@1: +0.061 (McNemar), p_adj=0.0000
- vs ΔContext store (top-3) (path-length detector), hit@3: +0.217 (McNemar), p_adj=0.0000
- vs ΔContext store (top-3) (path-length detector), topic_recall: +0.086 [+0.058, +0.114], p_adj=0.0000

### LoCoMo end-to-end QA (Qwen2.5-3B) (budget 512)

| Method | f1 | em | evidence_recall | context_tokens | Abstain |
|---|---|---|---|---|---|
| Sliding window | 0.026*** | 0.006*** | 0.011*** | 483 | 0.975 |
| Extractive summary | 0.026*** | 0.006*** | 0.004*** | 467 | 0.971 |
| RAG + recency tail (50/50) | 0.223 | 0.073 | 0.374 | 484 | 0.688 |
| RAG, turn-level | 0.259*** | 0.081 | 0.475*** | 507 | 0.646 |
| RAG + chunk prior | 0.267*** | 0.086 | 0.494*** | 507 | 0.614 |
| ΔContext turn ranking (v1.2) | 0.228 | 0.075 | 0.375 | 486 | 0.673 |
| **ΔContext store (top-3)** | 0.220 | 0.071 | 0.382 | 506 | 0.675 |
| Semantic chunking store (top-3) | 0.202 | 0.067 | 0.356 | 507 | 0.677 |
| Fixed-size chunks (5 turns) store (top-3) | 0.224 | 0.080 | 0.412 | 506 | 0.621 |

ΔContext (ΔContext store (top-3)) minus each baseline; Holm-adjusted p within this table:
- vs Extractive summary, em: +0.064 [+0.052, +0.078], p_adj=0.0000
- vs Extractive summary, evidence_recall: +0.378 [+0.354, +0.401], p_adj=0.0000
- vs Extractive summary, f1: +0.194 [+0.178, +0.210], p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), em: -0.009 [-0.021, +0.003], p_adj=1.0000
- vs Fixed-size chunks (5 turns) store (top-3), evidence_recall: -0.030 [-0.057, -0.004], p_adj=0.3094
- vs Fixed-size chunks (5 turns) store (top-3), f1: -0.004 [-0.020, +0.011], p_adj=1.0000
- vs RAG + chunk prior, em: -0.015 [-0.026, -0.004], p_adj=0.1008
- vs RAG + chunk prior, evidence_recall: -0.112 [-0.133, -0.092], p_adj=0.0000
- vs RAG + chunk prior, f1: -0.047 [-0.061, -0.033], p_adj=0.0000
- vs RAG + recency tail (50/50), em: -0.003 [-0.014, +0.009], p_adj=1.0000
- vs RAG + recency tail (50/50), evidence_recall: +0.008 [-0.015, +0.031], p_adj=1.0000
- vs RAG + recency tail (50/50), f1: -0.003 [-0.018, +0.012], p_adj=1.0000
- vs RAG, turn-level, em: -0.010 [-0.021, +0.000], p_adj=0.5420
- vs RAG, turn-level, evidence_recall: -0.093 [-0.113, -0.074], p_adj=0.0000
- vs RAG, turn-level, f1: -0.039 [-0.052, -0.026], p_adj=0.0000
- vs Semantic chunking store (top-3), em: +0.004 [-0.008, +0.016], p_adj=1.0000
- vs Semantic chunking store (top-3), evidence_recall: +0.026 [+0.003, +0.049], p_adj=0.3278
- vs Semantic chunking store (top-3), f1: +0.018 [+0.002, +0.034], p_adj=0.3096
- vs Sliding window, em: +0.064 [+0.052, +0.078], p_adj=0.0000
- vs Sliding window, evidence_recall: +0.371 [+0.347, +0.394], p_adj=0.0000
- vs Sliding window, f1: +0.194 [+0.178, +0.210], p_adj=0.0000
- vs ΔContext turn ranking (v1.2), em: -0.004 [-0.016, +0.008], p_adj=1.0000
- vs ΔContext turn ranking (v1.2), evidence_recall: +0.007 [-0.016, +0.030], p_adj=1.0000
- vs ΔContext turn ranking (v1.2), f1: -0.008 [-0.023, +0.007], p_adj=1.0000

### LongMemEval-S end-to-end QA (Qwen2.5-3B) (budget 2048)

| Method | f1 | em | evidence_recall | context_tokens | Abstain |
|---|---|---|---|---|---|

ΔContext (ΔContext turn ranking (v1.2)) minus each baseline; Holm-adjusted p within this table:

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

### Segmentation quality: configs\dataset\locomo.yaml (test histories: 5, parameters tuned on dev)

| Method | Pk | WindowDiff | P@1 | R@1 | F1@1 | F1@0 | Segments |
|---|---|---|---|---|---|---|---|
| fixed | 0.520 | 0.981 | 0.145 | 0.642 | 0.237 | 0.073 | 125.2 |
| adjacent | 0.514 | 0.879 | 0.101 | 0.654 | 0.171 | 0.147 | 190.6 |
| semantic_chunker | 0.516 | 0.902 | 0.102 | 0.687 | 0.177 | 0.154 | 188.2 |
| texttiling | 0.516 | 0.923 | 0.175 | 0.800 | 0.287 | 0.208 | 130.8 |
| delta_context | 0.502 | 0.599 | 0.154 | 0.261 | 0.191 | 0.167 | 48.2 |
| delta_context_unlocalized | 0.511 | 0.605 | 0.118 | 0.193 | 0.144 | 0.053 | 48.2 |
| delta_context_sum | 0.485 | 0.485 | 0.157 | 0.175 | 0.164 | 0.026 | 33.4 |

### Scaling: LongMemEval-S store grown with distractor histories

Topic hit@1 / hit@3 (comparisons per question). Stars: exact McNemar on hit@3 vs ΔContext, Holm-adjusted over the whole table.

| Turns | ΔContext | Fixed-size chunks | Semantic chunking | TextTiling | Adjacent threshold | Session oracle | Full scan |
|---|---|---|---|---|---|---|---|
| 492 | 0.782 / 0.924 (83) | 0.496 / 0.748*** (114) | 0.727 / 0.908 (91) | 0.731 / 0.903 (93) | 0.748 / 0.891 (93) | 0.815 / 0.933 (81) | (492) |
| 1969 | 0.685 / 0.828 (237) | 0.424 / 0.697*** (409) | 0.588 / 0.790 (238) | 0.664 / 0.836 (308) | 0.634 / 0.807 (264) | 0.706 / 0.861 (224) | (1969) |
| 4912 | 0.576 / 0.739 (542) | 0.345 / 0.563*** (998) | 0.479 / 0.689 (530) | 0.546 / 0.744 (735) | 0.521 / 0.714 (610) | 0.588 / 0.765 (509) | (4912) |
| 9862 | 0.492 / 0.647 (1056) | 0.273 / 0.504*** (1988) | 0.408 / 0.584 (1024) | 0.479 / 0.647 (1448) | 0.445 / 0.609 (1194) | 0.525 / 0.660 (988) | (9862) |
| 19725 | 0.420 / 0.597 (2080) | 0.290 / 0.475* (3960) | 0.345 / 0.513 (2008) | 0.399 / 0.605 (2879) | 0.387 / 0.567 (2360) | 0.441 / 0.618 (1943) | (19725) |
