# ContextBench / ΔContext: result tables

Answerable probes only; significance markers: * p<0.05, ** p<0.01, *** p<0.001 (ΔContext vs that row, Holm-adjusted within the table).

### LongMemEval-S topic store (budget 2048)

| Method | hit@1 | hit@3 | topic_recall | evidence_recall | searched |
|---|---|---|---|---|---|
| **ΔContext store (top-3)** | 0.770 | 0.930 | 0.732 | 0.733 | 17.1 |
| Session oracle store (top-3) | 0.785 | 0.926 | 0.733 | 0.734 | 16.5 |
| Adjacent threshold store (top-3) | 0.736 | 0.902 | 0.711 | 0.725 | 20.1 |
| Semantic chunking store (top-3) | 0.721 | 0.904 | 0.704 | 0.717 | 18.4 |
| TextTiling store (top-3) | 0.777 | 0.921 | 0.729 | 0.732 | 16.9 |
| Fixed-size chunks (5 turns) store (top-3) | 0.457*** | 0.736*** | 0.534*** | 0.581*** | 23.1 |
| Fixed-size chunks (10 turns) store (top-3) | 0.543*** | 0.813*** | 0.601*** | 0.601*** | 16.2 |
| RAG, turn-level | -- | -- | 0.774*** | 0.774*** | 100.0 |

ΔContext (ΔContext store (top-3)) minus each baseline; Holm-adjusted p within this table:
- vs Adjacent threshold store (top-3), evidence_recall: +0.008 [-0.009, +0.025], p_adj=1.0000
- vs Adjacent threshold store (top-3), hit@1: +0.034 (McNemar), p_adj=0.3224
- vs Adjacent threshold store (top-3), hit@3: +0.028 (McNemar), p_adj=0.1080
- vs Adjacent threshold store (top-3), topic_recall: +0.021 [+0.003, +0.040], p_adj=0.2664
- vs Fixed-size chunks (10 turns) store (top-3), evidence_recall: +0.132 [+0.101, +0.163], p_adj=0.0000
- vs Fixed-size chunks (10 turns) store (top-3), hit@1: +0.228 (McNemar), p_adj=0.0000
- vs Fixed-size chunks (10 turns) store (top-3), hit@3: +0.117 (McNemar), p_adj=0.0000
- vs Fixed-size chunks (10 turns) store (top-3), topic_recall: +0.131 [+0.100, +0.162], p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), evidence_recall: +0.152 [+0.116, +0.188], p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), hit@1: +0.313 (McNemar), p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), hit@3: +0.194 (McNemar), p_adj=0.0000
- vs Fixed-size chunks (5 turns) store (top-3), topic_recall: +0.198 [+0.163, +0.233], p_adj=0.0000
- vs RAG, turn-level, evidence_recall: -0.041 [-0.060, -0.023], p_adj=0.0000
- vs RAG, turn-level, topic_recall: -0.042 [-0.062, -0.024], p_adj=0.0000
- vs Semantic chunking store (top-3), evidence_recall: +0.016 [-0.003, +0.035], p_adj=1.0000
- vs Semantic chunking store (top-3), hit@1: +0.049 (McNemar), p_adj=0.0512
- vs Semantic chunking store (top-3), hit@3: +0.026 (McNemar), p_adj=0.2197
- vs Semantic chunking store (top-3), topic_recall: +0.028 [+0.008, +0.049], p_adj=0.1400
- vs Session oracle store (top-3), evidence_recall: -0.001 [-0.013, +0.010], p_adj=1.0000
- vs Session oracle store (top-3), hit@1: -0.015 (McNemar), p_adj=1.0000
- vs Session oracle store (top-3), hit@3: +0.004 (McNemar), p_adj=1.0000
- vs Session oracle store (top-3), topic_recall: -0.001 [-0.013, +0.010], p_adj=1.0000
- vs TextTiling store (top-3), evidence_recall: +0.001 [-0.011, +0.015], p_adj=1.0000
- vs TextTiling store (top-3), hit@1: -0.006 (McNemar), p_adj=1.0000
- vs TextTiling store (top-3), hit@3: +0.009 (McNemar), p_adj=1.0000
- vs TextTiling store (top-3), topic_recall: +0.002 [-0.010, +0.016], p_adj=1.0000

### LongMemEval-S topic store ablation (budget 2048)

| Method | hit@1 | hit@3 | topic_recall |
|---|---|---|---|
| **ΔContext store (top-3)** | 0.770 | 0.930 | 0.732 |
| ΔContext v2 store (top-3) | 0.749 | 0.917 | 0.716 |
| ΔContext v2 store (top-3) (no localization) | 0.426*** | 0.732*** | 0.516*** |
| ΔContext v2 store (top-3) (path-length detector) | 0.640*** | 0.847*** | 0.609*** |

ΔContext (ΔContext store (top-3)) minus each baseline; Holm-adjusted p within this table:
- vs ΔContext v2 store (top-3), hit@1: +0.021 (McNemar), p_adj=0.1433
- vs ΔContext v2 store (top-3), hit@3: +0.013 (McNemar), p_adj=0.1406
- vs ΔContext v2 store (top-3), topic_recall: +0.016 [+0.002, +0.029], p_adj=0.0714
- vs ΔContext v2 store (top-3) (no localization), hit@1: +0.345 (McNemar), p_adj=0.0000
- vs ΔContext v2 store (top-3) (no localization), hit@3: +0.198 (McNemar), p_adj=0.0000
- vs ΔContext v2 store (top-3) (no localization), topic_recall: +0.216 [+0.179, +0.253], p_adj=0.0000
- vs ΔContext v2 store (top-3) (path-length detector), hit@1: +0.130 (McNemar), p_adj=0.0000
- vs ΔContext v2 store (top-3) (path-length detector), hit@3: +0.083 (McNemar), p_adj=0.0000
- vs ΔContext v2 store (top-3) (path-length detector), topic_recall: +0.123 [+0.091, +0.156], p_adj=0.0000

### LongMemEval-S end-to-end QA (Qwen2.5-3B) (budget 2048)

| Method | f1 | em | evidence_recall | context_tokens | Abstain |
|---|---|---|---|---|---|
| Sliding window | 0.019*** | 0.006*** | 0.009*** | 1809 | 1.000 |
| RAG, turn-level | 0.311 | 0.217 | 0.774*** | 2037 | 0.933 |
| **ΔContext store (top-3)** | 0.302 | 0.206 | 0.733 | 2038 | 1.000 |
| Session oracle store (top-3) | 0.298 | 0.202 | 0.734 | 2037 | 1.000 |
| Semantic chunking store (top-3) | 0.284 | 0.194 | 0.717 | 2037 | 0.933 |
| TextTiling store (top-3) | 0.296 | 0.200 | 0.732 | 2037 | 0.967 |
| Fixed-size chunks (10 turns) store (top-3) | 0.263* | 0.187 | 0.601*** | 2039 | 1.000 |
| Evidence sessions only (oracle) | 0.279 | 0.202 | 1.000 | 5987 | 0.933 |

ΔContext (ΔContext store (top-3)) minus each baseline; Holm-adjusted p within this table:
- vs Fixed-size chunks (10 turns) store (top-3), em: +0.019 [-0.004, +0.043], p_adj=1.0000
- vs Fixed-size chunks (10 turns) store (top-3), evidence_recall: +0.132 [+0.101, +0.163], p_adj=0.0000
- vs Fixed-size chunks (10 turns) store (top-3), f1: +0.039 [+0.016, +0.064], p_adj=0.0182
- vs RAG, turn-level, em: -0.011 [-0.030, +0.009], p_adj=1.0000
- vs RAG, turn-level, evidence_recall: -0.041 [-0.060, -0.023], p_adj=0.0000
- vs RAG, turn-level, f1: -0.009 [-0.028, +0.011], p_adj=1.0000
- vs Semantic chunking store (top-3), em: +0.013 [-0.004, +0.030], p_adj=1.0000
- vs Semantic chunking store (top-3), evidence_recall: +0.016 [-0.003, +0.035], p_adj=1.0000
- vs Semantic chunking store (top-3), f1: +0.018 [-0.001, +0.038], p_adj=0.7848
- vs Session oracle store (top-3), em: +0.004 [-0.006, +0.015], p_adj=1.0000
- vs Session oracle store (top-3), evidence_recall: -0.001 [-0.013, +0.010], p_adj=1.0000
- vs Session oracle store (top-3), f1: +0.004 [-0.007, +0.016], p_adj=1.0000
- vs Sliding window, em: +0.200 [+0.164, +0.236], p_adj=0.0000
- vs Sliding window, evidence_recall: +0.724 [+0.689, +0.759], p_adj=0.0000
- vs Sliding window, f1: +0.283 [+0.246, +0.321], p_adj=0.0000
- vs TextTiling store (top-3), em: +0.006 [-0.002, +0.015], p_adj=1.0000
- vs TextTiling store (top-3), evidence_recall: +0.001 [-0.011, +0.015], p_adj=1.0000
- vs TextTiling store (top-3), f1: +0.006 [-0.005, +0.017], p_adj=1.0000

### Segmentation quality: configs\dataset\longmemeval_s.yaml (test histories: 250, parameters tuned on dev)

| Method | Pk | WindowDiff | P@1 | R@1 | F1@1 | F1@0 | Segments |
|---|---|---|---|---|---|---|---|
| fixed | 0.532 | 0.572 | 0.276 | 0.579 | 0.373 | 0.126 | 98.9 |
| adjacent | 0.125 | 0.180 | 0.714 | 0.858 | 0.775 | 0.768 | 58.1 |
| semantic_chunker | 0.141 | 0.185 | 0.745 | 0.791 | 0.766 | 0.756 | 50.5 |
| texttiling | 0.215 | 0.261 | 0.643 | 0.969 | 0.771 | 0.765 | 71.7 |
| delta_context | 0.102 | 0.133 | 0.809 | 0.870 | 0.838 | 0.825 | 51.2 |
| delta_context_unlocalized | 0.443 | 0.476 | 0.165 | 0.178 | 0.171 | 0.035 | 51.2 |
| delta_context_sum | 0.460 | 0.469 | 0.293 | 0.162 | 0.208 | 0.052 | 26.9 |

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

### Topic segmentation on four datasets (dev-tuned, one-shot test)

Cells: Pk / exact-boundary F1 (J). Tuning objective J = ((1-Pk)+F1)/2.

| Method | Online | LongMemEval-S | SuperDialseg | TIAGE | DialSeg711 |
|---|---|---|---|---|---|
| ΔContext v3 (delayed commitment) | yes | 0.058 / 0.914 (0.928) | 0.413 / 0.502 (0.544) | 0.427 / 0.365 (0.469) | 0.490 / 0.497 (0.503) |
| ΔContext v3 (universal) | yes | 0.094 / 0.848 (0.877) | 0.408 / 0.248 (0.420) | 0.388 / 0.290 (0.451) | 0.283 / 0.554 (0.636) |
| TextTiling | no | 0.069 / 0.894 (0.913) | 0.385 / 0.493 (0.554) | 0.442 / 0.374 (0.466) | 0.401 / 0.560 (0.579) |
| Semantic chunking | no | 0.141 / 0.756 (0.808) | 0.498 / 0.399 (0.450) | 0.441 / 0.397 (0.478) | 0.464 / 0.448 (0.492) |
| Adjacent-turn threshold | yes | 0.122 / 0.754 (0.816) | 0.511 / 0.408 (0.449) | 0.471 / 0.399 (0.464) | 0.499 / 0.417 (0.459) |
| Fixed-size chunks | yes | 0.532 / 0.170 (0.319) | 0.495 / 0.455 (0.480) | 0.544 / 0.260 (0.358) | 0.576 / 0.457 (0.441) |
| ΔContext v2 (displacement) | yes | 0.102 / 0.825 (0.861) | 0.459 / 0.437 (0.489) | 0.454 / 0.385 (0.466) | 0.522 / 0.455 (0.466) |

### Scaling: LongMemEval-S store grown with distractor histories

Topic hit@1 / hit@3 (comparisons per question). Stars: exact McNemar on hit@3 vs ΔContext, Holm-adjusted over the whole table.

| Turns | ΔContext | Fixed-size chunks | Semantic chunking | TextTiling | Adjacent threshold | Session oracle | Full scan |
|---|---|---|---|---|---|---|---|
| 492 | 0.776 / 0.924 (84) | 0.586 / 0.844** (79) | 0.717 / 0.916 (90) | 0.781 / 0.920 (83) | 0.738 / 0.903 (99) | 0.793 / 0.924 (81) | (492) |
| 1971 | 0.595 / 0.827 (227) | 0.439 / 0.700*** (227) | 0.549 / 0.789 (239) | 0.578 / 0.823 (242) | 0.561 / 0.814 (308) | 0.582 / 0.844 (225) | (1971) |
| 4917 | 0.489 / 0.692 (512) | 0.354 / 0.532*** (522) | 0.426 / 0.654 (532) | 0.511 / 0.696 (558) | 0.451 / 0.684 (721) | 0.489 / 0.713 (510) | (4917) |
| 9883 | 0.422 / 0.612 (990) | 0.350 / 0.498** (1019) | 0.333 / 0.532 (1025) | 0.439 / 0.620 (1089) | 0.401 / 0.578 (1424) | 0.451 / 0.616 (988) | (9883) |
| 19732 | 0.283 / 0.481 (1949) | 0.194 / 0.384 (2004) | 0.249 / 0.426 (2010) | 0.316 / 0.489 (2148) | 0.283 / 0.473 (2810) | 0.304 / 0.477 (1945) | (19732) |
