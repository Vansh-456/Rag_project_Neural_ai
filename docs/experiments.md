# Experiments log

All numbers come from code that was run; each links to its result directory or file in `results/`.
Splits: dev = 38 queries (34 answerable), test = 33 answerable (frozen; run once with the final
configuration). Intervals are 95% percentile-bootstrap CIs over queries (1000 resamples);
significance = paired bootstrap (2000 resamples).

## E1 — Corpus acquisition (Phase 0/2)
* 79/79 arXiv papers downloaded; every expected title verified against arXiv metadata (min
  similarity 0.877; 10 ids flagged "verify" were all correct). — `data/raw/download_report.json`

## E2 — Parsing (Phase 2) — `results/parse_report_*.json`, `results/parse_comparison.json`

| Parser | papers | sections identified | mean sections | mean refs | garbled score |
|---|---|---|---|---|---|
| layout-aware (default) | 79 | **100%** | 24 (median) | 59.2 | 0.014 |
| layout-aware, same 10 papers | 10 | 100% | 19.9 | 38.9 | 0.008 |
| plain text baseline, same 10 | 10 | 100% | 11.2 | 37.2 | 0.008 |

Before two fixes (centred abstract on two-column pages; split "1 | Introduction" heading runs) the
layout parser identified sections in only 86% of papers (ADR-005).

## E3 — Chunking statistics (Phase 3) — `results/chunking_sweep.json`

| Strategy | size | overlap | chunks | mean tokens | % crossing sections | % tiny (<32 tok) |
|---|---|---|---|---|---|---|
| fixed | 256 | 25 | 4,676 | 253 | 38.5 | 0.0 |
| recursive | 256 | 25 | 5,104 | 216 | 22.4 | 0.5 |
| semantic* | 256 | – | 11,651 | 92 | 10.1 | 0.8 |
| structure_aware | 256 | 25 | 6,308 | 174 | **0.0** | 2.8 |
| fixed | 512 | 51 | 2,360 | 503 | 58.7 | 0.0 |
| structure_aware | 512 | 51 | 3,651 | 305 | 0.0 | 2.6 |

Fixed windows cross section boundaries in 22–79% of chunks depending on size; structure-aware never
does, at the cost of more small chunks (short sections). *The semantic chunker ignores overlap and
its size cap rarely binds (topic-shift splits dominate); statistics used the hashing embedder.
Full 48-row grid in the JSON.

## E4 — Retrieval sweep (Phase 5) — `results/retrieval_sweep_*.json` (dense-only, dev)

<!-- SWEEP-TABLE -->

## E5 — Graph construction (Phase 8) — `data/index/<index>/graph_stats.json`, `results/graph_edge_audit*.json`

| | v1 | v2 (after ADR-016) |
|---|---|---|
| nodes / edges | 682 / 2,964 | 680 / 2,684 |
| % edges traceable (chunk or reference/metadata origin) | 100% | 100% |
| % edges grounded to a chunk | 77.0% | 74.6% |
| audit precision (50 random edges) | 0.74 | **0.78** |
| INTRODUCES correct (full review) | 77/94 | 77/77 |

## E6 — Baseline and development comparison (dev) — `results/ablation_report_dev.md`

| System | NDCG@5 | Recall@5 | MRR | p50 ms |
|---|---|---|---|---|
| vector_only | 0.360 [0.23, 0.50] | 0.485 | 0.352 | 54 |
| bm25_only | 0.521 [0.39, 0.66] | 0.618 | 0.546 | 14 |
| hybrid | 0.489 [0.37, 0.62] | 0.618 | 0.473 | 74 |
| hybrid_rerank | **0.584** [0.46, 0.70] | 0.750 | 0.554 | 5,186 |
| graph_only | 0.375 | 0.426 | 0.393 | 118 |
| graph_vector | 0.460 | 0.706 | 0.429 | 181 |
| adaptive (spec rules) | 0.442 | 0.603 | 0.441 | 191 |
| adaptive_tuned (dev-fit) | 0.538 | 0.721 | 0.531 | 228 |
| adaptive_budget | 0.481 | 0.632 | 0.480 | 240 |

The vector-only baseline missed the Phase-0 targets (Recall@5 0.49 vs 0.60; NDCG@5 0.36 vs 0.45) —
recorded as a miss. Dev results motivated hybrid retrieval and the router re-fit (ADR-015).

## E7 — Final study (frozen test) — `docs/ablation_report.md`

See the report for CIs, per-type significance, path metrics, failure categories and the Pareto plot.
Headlines:

* **graph_vector is best**: NDCG@5 0.766 [0.65, 0.88], +0.144 vs vector-only (p = 0.003); hit@5 0.91,
  doc-recall@5 0.96; relationship NDCG@5 0.752 (p < 0.05); path entity recall 0.97.
* **hybrid** 0.683 (+0.062, p = 0.06) at 62 ms; **bm25_only** 0.678 at 11 ms.
* **Reranking did not transfer**: hybrid_rerank 0.661 (dev best 0.584) at ~5 s/query.
* **Adaptive routing**: spec rules 0.684, dev-tuned 0.700, budget 0.694 — none significantly better
  than always-hybrid (tuned vs hybrid: +0.017, CI [−0.07, 0.11], p = 0.70) and all cost more. The
  central hypothesis is **not supported** on this data.
* **Router vs oracle**: oracle NDCG@5 0.885 (per-query best strategy). Router accuracy 0.64 judged on
  quality alone, 0.06 judged on utility (cost term breaks the many ties toward BM25) —
  `results/router_accuracy_quality_only_test.json`.

## E8 — Generation + verification (Phase 13/14) — `results/p14_adaptive_verified_*`

Extractive generator (no LLM), lexical verifier, `adaptive_verified` pipeline:

| Metric | test | dev |
|---|---|---|
| citation validity (all ids exist, every claim cited) | 0.88 | 0.91 |
| claim citation coverage | 0.96 | – |
| citation precision (cited chunks satisfy a label) | 0.48 | – |
| lexical support rate | 0.94 | 0.90 |
| abstention on unanswerable (4 queries, all in dev) | – | **1/4** |

LLM generation, LLM-judge calibration and NLI verification are implemented and unit-tested but were
not run (no API budget).

## E9 — Serving (Phase 17) — `results/load_test.json`
60 requests at concurrency 4 against the live API (graph_vector, CPU, while a re-embedding job ran):
4.2 req/s, p50 1.2 s, p95 2.8 s; 12 requests correctly rejected with 429 by the 60/min rate limit.
Health, query, compare, SSE streaming, graph neighbourhood and chunk endpoints verified through the
Next.js proxy.

## Validity notes
* Labels: draft v1, single annotator (see `docs/labeling_guideline.md`).
* A sleep-corrupted run and an accidental pre-fix test run are archived in `results/superseded/`.
