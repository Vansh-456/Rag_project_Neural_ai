# Ablation report

_Generated 2026-10-15 13:49 UTC by `agvrag ablate` — every number below is computed from `results/<exp_id>/` (split: **test**). Do not edit by hand._

Baseline for significance tests: **vector_only**; metric: **ndcg@5**; paired bootstrap, 2000 resamples. Brackets are 95% bootstrap CIs of the mean.

## Overall retrieval quality

| System | n | NDCG@5 | Recall@5 | MRR | Hit@5 | Doc-recall@5 | Δ NDCG@5 vs baseline (p) | p50 ms | p95 ms | cost units |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25_only | 33 | 0.678 [0.55, 0.80] | 0.795 [0.65, 0.92] | 0.669 [0.54, 0.80] | 0.818 [0.67, 0.94] | 0.901 [0.80, 0.98] | +0.057 (p=0.349) | 11 | 30 | 0.30 |
| vector_only | 33 | 0.621 [0.49, 0.75] | 0.725 [0.58, 0.86] | 0.608 [0.46, 0.74] | 0.758 [0.61, 0.88] | 0.876 [0.76, 0.97] | +0.000 (p=1.000) | 52 | 64 | 1.00 |
| hybrid | 33 | 0.683 [0.55, 0.82] | 0.773 [0.64, 0.91] | 0.675 [0.54, 0.81] | 0.788 [0.64, 0.91] | 0.939 [0.86, 1.00] | +0.062 (p=0.060) | 62 | 79 | 1.30 |
| hybrid_rerank | 33 | 0.661 [0.53, 0.78] | 0.785 [0.66, 0.90] | 0.660 [0.52, 0.79] | 0.849 [0.73, 0.97] | 0.907 [0.81, 0.98] | +0.040 (p=0.518) | 5014 | 5434 | 2.30 |
| graph_only | 33 | 0.522 [0.38, 0.67] | 0.621 [0.47, 0.77] | 0.518 [0.38, 0.66] | 0.667 [0.48, 0.82] | 0.803 [0.67, 0.91] | -0.099 (p=0.365) | 118 | 281 | 0.62 |
| graph_vector | 33 | 0.766 [0.65, 0.88] | 0.851 [0.73, 0.95] | 0.773 [0.65, 0.88] | 0.909 [0.82, 1.00] | 0.957 [0.89, 1.00] | +0.144 (p=0.003) | 186 | 407 | 1.94 |
| adaptive | 33 | 0.684 [0.56, 0.81] | 0.780 [0.65, 0.90] | 0.681 [0.55, 0.82] | 0.818 [0.70, 0.94] | 0.917 [0.81, 1.00] | +0.062 (p=0.133) | 150 | 9946 | 1.70 |
| adaptive_tuned | 33 | 0.700 [0.58, 0.83] | 0.821 [0.70, 0.93] | 0.685 [0.57, 0.82] | 0.879 [0.76, 0.97] | 0.927 [0.83, 1.00] | +0.079 (p=0.100) | 175 | 9907 | 2.08 |
| adaptive_budget | 33 | 0.694 [0.57, 0.82] | 0.811 [0.67, 0.93] | 0.671 [0.55, 0.81] | 0.849 [0.73, 0.97] | 0.917 [0.81, 1.00] | +0.072 (p=0.155) | 200 | 9918 | 2.57 |

## NDCG@5 per query type

| System | definition | relationship | comparison | multi_hop | experimental |
|---|---|---|---|---|---|
| bm25_only | 0.752 (n=8) | 0.580 (n=7) | 0.482 (n=6) | 0.481 (n=5) | 1.000 (n=7) |
| vector_only | 0.625 (n=8) | 0.565 (n=7) | 0.583 (n=6) | 0.611 (n=5) | 0.714 (n=7) |
| hybrid | 0.752 (n=8) | 0.664 (n=7) | 0.605 (n=6) | 0.648 (n=5) | 0.714 (n=7) |
| hybrid_rerank | 0.627 (n=8) | 0.599 (n=7) | 0.710 (n=6) | 0.392 (n=5) | 0.912 (n=7) |
| graph_only | 0.562 (n=8) | 0.723* (n=7) | 0.440 (n=6) | 0.305 (n=5) | 0.500 (n=7) |
| graph_vector | 0.812 (n=8) | 0.752* (n=7) | 0.699 (n=6) | 0.712 (n=5) | 0.823 (n=7) |
| adaptive | 0.625 (n=8) | 0.723* (n=7) | 0.699 (n=6) | 0.475 (n=5) | 0.847 (n=7) |
| adaptive_tuned | 0.752 (n=8) | 0.752* (n=7) | 0.699 (n=6) | 0.349* (n=5) | 0.841 (n=7) |
| adaptive_budget | 0.766 (n=8) | 0.717 (n=7) | 0.699 (n=6) | 0.323* (n=5) | 0.847 (n=7) |

`*` = paired-bootstrap p < 0.05 vs the baseline on that query type. Per-type n is small; treat unstarred differences as noise.

## Graph path metrics (queries with expected paths)

| System | path entity recall | path relation recall | path hit |
|---|---|---|---|
| bm25_only | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| vector_only | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| hybrid | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| hybrid_rerank | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] |
| graph_only | 0.972 [0.92, 1.00] | 0.861 [0.72, 1.00] | 0.917 [0.75, 1.00] |
| graph_vector | 0.972 [0.92, 1.00] | 0.861 [0.72, 1.00] | 0.917 [0.75, 1.00] |
| adaptive | 0.889 [0.69, 1.00] | 0.819 [0.61, 1.00] | 0.833 [0.58, 1.00] |
| adaptive_tuned | 0.889 [0.69, 1.00] | 0.819 [0.61, 1.00] | 0.833 [0.58, 1.00] |
| adaptive_budget | 0.389 [0.14, 0.67] | 0.319 [0.08, 0.57] | 0.333 [0.08, 0.58] |

## Strategy usage

| System | strategies chosen |
|---|---|
| bm25_only | {'bm25': 33} |
| vector_only | {'vector': 33} |
| hybrid | {'hybrid': 33} |
| hybrid_rerank | {'hybrid': 33} |
| graph_only | {'graph': 33} |
| graph_vector | {'graph_vector': 33} |
| adaptive | {'vector': 12, 'hybrid': 4, 'graph': 7, 'graph_vector': 6, 'multi_hop': 4} |
| adaptive_tuned | {'hybrid': 16, 'graph_vector': 13, 'multi_hop': 4} |
| adaptive_budget | {'multi_hop': 6, 'hybrid': 16, 'graph_vector': 6, 'vector': 5} |

## Failure categories (hit@5 = 0)

| System | categories |
|---|---|
| bm25_only | {'right_doc_wrong_passage': 4, 'relevant_doc_not_retrieved': 2, 'multi_hop_chain': 1} |
| vector_only | {'right_doc_wrong_passage': 5, 'relevant_doc_not_retrieved': 3} |
| hybrid | {'right_doc_wrong_passage': 6, 'relevant_doc_not_retrieved': 1, 'multi_hop_chain': 1} |
| hybrid_rerank | {'right_doc_wrong_passage': 4, 'relevant_doc_not_retrieved': 1, 'multi_hop_chain': 1} |
| graph_only | {'right_doc_wrong_passage': 6, 'relevant_doc_not_retrieved': 4, 'multi_hop_chain': 2, 'entity_not_linked': 1, 'empty_result': 1} |
| graph_vector | {'right_doc_wrong_passage': 3} |
| adaptive | {'right_doc_wrong_passage': 4, 'relevant_doc_not_retrieved': 2, 'multi_hop_chain': 1} |
| adaptive_tuned | {'right_doc_wrong_passage': 3, 'relevant_doc_not_retrieved': 1, 'multi_hop_chain': 1} |
| adaptive_budget | {'right_doc_wrong_passage': 3, 'relevant_doc_not_retrieved': 2, 'multi_hop_chain': 2} |

## Router vs oracle

Router `adaptive_tuned` accuracy (chosen strategy achieves the oracle's best utility): **0.06** (n=33).

Per type: {'definition': 0.0, 'relationship': 0.1429, 'comparison': 0.1667, 'multi_hop': 0.0, 'experimental': 0.0}

Mean ndcg@5 by strategy: {'vector': 0.6214, 'bm25': 0.6782, 'hybrid': 0.683, 'graph': 0.5221, 'graph_vector': 0.7659, 'multi_hop': 0.6665, 'adaptive_tuned': 0.7001, 'oracle': 0.8846}

Mean cost units by strategy: {'vector': 1.0, 'bm25': 0.3, 'hybrid': 1.3, 'graph': 0.6212, 'graph_vector': 1.9364, 'multi_hop': 2.9394, 'adaptive_tuned': 2.0818}

Adaptive vs always-hybrid: {'diff': 0.0171631929993718, 'ci_low': -0.06689101590561904, 'ci_high': 0.10554567921628888, 'p_value': 0.696, 'n': 33, 'cost_saved_units': -0.7818}


## Experiment ids

- bm25_only: `ablation_bm25_only_test_20261015T134102653`
- vector_only: `ablation_vector_only_test_20261015T134104331`
- hybrid: `ablation_hybrid_test_20261015T134106728`
- hybrid_rerank: `ablation_hybrid_rerank_test_20261015T134112460`
- graph_only: `ablation_graph_only_test_20261015T134358574`
- graph_vector: `ablation_graph_vector_test_20261015T134403589`
- adaptive: `ablation_adaptive_test_20261015T134411128`
- adaptive_tuned: `ablation_adaptive_tuned_test_20261015T134508762`
- adaptive_budget: `ablation_adaptive_budget_test_20261015T134606100`

![Quality vs cost](pareto.png)

