# Ablation report

_Generated 2026-10-15 13:39 UTC by `agvrag ablate` — every number below is computed from `results/<exp_id>/` (split: **dev**). Do not edit by hand._

Baseline for significance tests: **vector_only**; metric: **ndcg@5**; paired bootstrap, 2000 resamples. Brackets are 95% bootstrap CIs of the mean.

## Overall retrieval quality

| System | n | NDCG@5 | Recall@5 | MRR | Hit@5 | Doc-recall@5 | Δ NDCG@5 vs baseline (p) | p50 ms | p95 ms | cost units |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25_only | 34 | 0.521 [0.39, 0.66] | 0.618 [0.47, 0.76] | 0.546 [0.41, 0.69] | 0.706 [0.56, 0.85] | 0.897 [0.81, 0.97] | +0.160 (p=0.011) | 14 | 25 | 0.30 |
| vector_only | 34 | 0.360 [0.23, 0.50] | 0.485 [0.34, 0.63] | 0.352 [0.23, 0.49] | 0.529 [0.38, 0.71] | 0.853 [0.75, 0.94] | +0.000 (p=1.000) | 54 | 63 | 1.00 |
| hybrid | 34 | 0.489 [0.37, 0.62] | 0.618 [0.47, 0.76] | 0.473 [0.35, 0.62] | 0.706 [0.56, 0.85] | 0.868 [0.76, 0.96] | +0.128 (p=0.000) | 74 | 96 | 1.30 |
| hybrid_rerank | 34 | 0.584 [0.46, 0.70] | 0.750 [0.60, 0.88] | 0.554 [0.43, 0.68] | 0.765 [0.62, 0.88] | 0.926 [0.82, 1.00] | +0.224 (p=0.000) | 5186 | 6713 | 2.30 |
| graph_only | 34 | 0.375 [0.22, 0.52] | 0.426 [0.26, 0.59] | 0.393 [0.25, 0.55] | 0.441 [0.26, 0.62] | 0.765 [0.62, 0.90] | +0.014 (p=0.854) | 118 | 258 | 0.59 |
| graph_vector | 34 | 0.460 [0.34, 0.58] | 0.706 [0.56, 0.85] | 0.429 [0.30, 0.55] | 0.735 [0.59, 0.88] | 0.853 [0.74, 0.96] | +0.099 (p=0.111) | 181 | 421 | 1.95 |
| adaptive | 34 | 0.442 [0.32, 0.58] | 0.603 [0.46, 0.76] | 0.441 [0.31, 0.59] | 0.647 [0.50, 0.79] | 0.882 [0.78, 0.97] | +0.081 (p=0.094) | 191 | 12781 | 1.68 |
| adaptive_tuned | 34 | 0.538 [0.41, 0.67] | 0.721 [0.57, 0.85] | 0.531 [0.39, 0.67] | 0.765 [0.62, 0.91] | 0.853 [0.74, 0.96] | +0.177 (p=0.000) | 228 | 10938 | 2.06 |
| adaptive_budget | 34 | 0.481 [0.35, 0.63] | 0.632 [0.47, 0.79] | 0.480 [0.34, 0.63] | 0.676 [0.53, 0.82] | 0.853 [0.74, 0.96] | +0.121 (p=0.009) | 240 | 11780 | 2.35 |

## NDCG@5 per query type

| System | definition | relationship | comparison | multi_hop | experimental |
|---|---|---|---|---|---|
| bm25_only | 0.423 (n=8) | 0.580 (n=7) | 0.570* (n=6) | 0.534* (n=6) | 0.519 (n=7) |
| vector_only | 0.463 (n=8) | 0.490 (n=7) | 0.184 (n=6) | 0.065 (n=6) | 0.519 (n=7) |
| hybrid | 0.570* (n=8) | 0.643 (n=7) | 0.356* (n=6) | 0.298* (n=6) | 0.519 (n=7) |
| hybrid_rerank | 0.704* (n=8) | 0.521 (n=7) | 0.547* (n=6) | 0.541* (n=6) | 0.580 (n=7) |
| graph_only | 0.375 (n=8) | 0.429 (n=7) | 0.508 (n=6) | 0.282 (n=6) | 0.286 (n=7) |
| graph_vector | 0.415 (n=8) | 0.695* (n=7) | 0.371 (n=6) | 0.448* (n=6) | 0.362 (n=7) |
| adaptive | 0.463 (n=8) | 0.429 (n=7) | 0.371 (n=6) | 0.430* (n=6) | 0.500 (n=7) |
| adaptive_tuned | 0.570* (n=8) | 0.695* (n=7) | 0.371 (n=6) | 0.430* (n=6) | 0.580 (n=7) |
| adaptive_budget | 0.570* (n=8) | 0.643 (n=7) | 0.371 (n=6) | 0.264* (n=6) | 0.500 (n=7) |

`*` = paired-bootstrap p < 0.05 vs the baseline on that query type. Per-type n is small; treat unstarred differences as noise.

## Graph path metrics (queries with expected paths)

| System | path entity recall | path relation recall | path hit |
|---|---|---|---|
| bm25_only | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| vector_only | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| hybrid | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| hybrid_rerank | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| graph_only | 0.847 [0.65, 1.00] | 0.889 [0.76, 1.00] | 0.750 [0.50, 1.00] |
| graph_vector | 0.806 [0.62, 0.96] | 0.847 [0.71, 0.97] | 0.667 [0.42, 0.92] |
| adaptive | 0.792 [0.54, 1.00] | 0.847 [0.65, 1.00] | 0.750 [0.50, 1.00] |
| adaptive_tuned | 0.750 [0.54, 0.96] | 0.806 [0.60, 0.97] | 0.667 [0.42, 0.92] |
| adaptive_budget | 0.208 [0.00, 0.42] | 0.264 [0.04, 0.51] | 0.167 [0.00, 0.42] |

## Strategy usage

| System | strategies chosen |
|---|---|
| bm25_only | {'bm25': 34} |
| vector_only | {'vector': 34} |
| hybrid | {'hybrid': 34} |
| hybrid_rerank | {'hybrid': 34} |
| graph_only | {'graph': 34} |
| graph_vector | {'graph_vector': 34} |
| adaptive | {'vector': 13, 'hybrid': 4, 'graph': 7, 'graph_vector': 6, 'multi_hop': 4} |
| adaptive_tuned | {'hybrid': 17, 'graph_vector': 13, 'multi_hop': 4} |
| adaptive_budget | {'hybrid': 19, 'graph_vector': 6, 'multi_hop': 4, 'vector': 5} |

## Failure categories (hit@5 = 0)

| System | categories |
|---|---|
| bm25_only | {'right_doc_wrong_passage': 9, 'vocabulary_mismatch': 2, 'relevant_doc_not_retrieved': 1, 'multi_hop_chain': 1} |
| vector_only | {'right_doc_wrong_passage': 14, 'multi_hop_chain': 5, 'vocabulary_mismatch': 2, 'relevant_doc_not_retrieved': 2} |
| hybrid | {'right_doc_wrong_passage': 8, 'relevant_doc_not_retrieved': 2, 'vocabulary_mismatch': 2, 'multi_hop_chain': 2} |
| hybrid_rerank | {'right_doc_wrong_passage': 6, 'vocabulary_mismatch': 2, 'relevant_doc_not_retrieved': 2, 'multi_hop_chain': 2} |
| graph_only | {'right_doc_wrong_passage': 12, 'relevant_doc_not_retrieved': 4, 'multi_hop_chain': 4, 'entity_not_linked': 3, 'empty_result': 3, 'vocabulary_mismatch': 2} |
| graph_vector | {'right_doc_wrong_passage': 6, 'relevant_doc_not_retrieved': 3, 'multi_hop_chain': 2, 'entity_not_linked': 2, 'vocabulary_mismatch': 1} |
| adaptive | {'right_doc_wrong_passage': 10, 'vocabulary_mismatch': 2, 'relevant_doc_not_retrieved': 2, 'multi_hop_chain': 2} |
| adaptive_tuned | {'right_doc_wrong_passage': 5, 'relevant_doc_not_retrieved': 3, 'multi_hop_chain': 2, 'vocabulary_mismatch': 1} |
| adaptive_budget | {'right_doc_wrong_passage': 8, 'relevant_doc_not_retrieved': 3, 'multi_hop_chain': 3, 'vocabulary_mismatch': 2} |

## Experiment ids

- bm25_only: `ablation_bm25_only_dev_20261015T133203767`
- vector_only: `ablation_vector_only_dev_20261015T133205281`
- hybrid: `ablation_hybrid_dev_20261015T133208419`
- hybrid_rerank: `ablation_hybrid_rerank_dev_20261015T133214974`
- graph_only: `ablation_graph_only_dev_20261015T133545131`
- graph_vector: `ablation_graph_vector_dev_20261015T133551147`
- adaptive: `ablation_adaptive_dev_20261015T133600520`
- adaptive_tuned: `ablation_adaptive_tuned_dev_20261015T133722040`
- adaptive_budget: `ablation_adaptive_budget_dev_20261015T133837248`

![Quality vs cost](pareto.png)

