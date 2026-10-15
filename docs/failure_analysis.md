# Failure analysis

Sources: `results/ablation_*_test_*/failures.json` (final test split), `results/ablation_report_dev.md`,
`results/graph_edge_audit*.json`, `results/p14_adaptive_verified_*`. Failure categories are assigned
automatically by `evaluation/runner.py::categorize_failure` for queries with hit@5 = 0.

## 1. Retrieval failure taxonomy (test, hit@5 = 0)

| Category | vector_only | bm25_only | hybrid | graph_only | graph_vector | adaptive_tuned |
|---|---|---|---|---|---|---|
| right paper, wrong passage | 5 | 4 | 6 | 6 | 3 | 3 |
| relevant paper not retrieved | 3 | 2 | 1 | 4 | 0 | 1 |
| multi-hop chain | 0 | 1 | 1 | 2 | 0 | 1 |
| entity not linked / empty | – | – | – | 2 | 0 | 0 |

**Dominant mode: right paper, wrong passage.** The retriever lands in the correct paper but not on the
labelled passage. Inspection shows two sub-causes:
1. *Unlabelled relevant passages* (labels are strict anchors, written before any run). E.g. for
   "What results show DPR outperforms BM25 on top-20 accuracy?" the top chunk is DPR's introduction
   ("outperforms BM25 by a large margin (65.2% vs. 42.9% in Top-5 accuracy)") — relevant, but the
   label anchors the results-section phrase "9%-19%". Document-level recall (0.88–0.96) is far above
   passage-level recall, consistent with this.
2. *Genuine passage-ranking errors* within long papers (related-work paragraphs that mention the query
   terms outrank the defining paragraph).

## 2. Where each retriever fails

* **Dense (bge-small) on names and acronyms.** On dev, dense-only scored 0.18 NDCG@5 on comparisons and
  0.07 on multi-hop questions; BM25 0.57 / 0.53. Questions are dense with method names ("ColBERTv2",
  "RAG-Token", "kNN-LM") that a small embedding model blurs. This is the measured motivation for
  hybrid retrieval and for re-fitting the router (ADR-015).
* **Dev and test disagree.** On test, dense-only is much stronger (0.62 vs 0.36 on dev) and
  reranking stops helping (hybrid_rerank 0.661 < hybrid 0.683). With ~33 queries per split,
  6–8 per type, a few queries swing a type's mean by 0.1–0.2. **Any per-type conclusion that is not
  starred in the ablation report should be treated as noise.**
* **Graph-only** fails when the answer is a passage rather than a relation (definition 0.56,
  experimental 0.50) and when the query entity is not in the gazetteer (`entity_not_linked`). It wins
  where the question *is* a relation: relationship NDCG@5 0.72 (p < 0.05 vs vector) with path entity
  recall 0.97.
* **Graph + vector** fixes both: it keeps the text retrievers' recall and adds graph-grounded chunks.
  Best example: *c04 "How do REALM and RAG differ?"* — every text-only system scores 0.00, graph_vector
  0.76, because the graph links both entities to their introducing papers and grounds those edges in
  the papers' own chunks.
* **Multi-hop strategy (graph → doc-restricted dense)** is the weakest graph strategy on test
  (multi_hop questions: 0.35 when chosen by the router vs 0.71 for graph_vector). Example m06 "Which
  retrieval-augmented models that use a DPR retriever pair it with a BART generator?": 0.00 vs 1.00.
  Cause: hop decay favours direct edges (DPR→Natural Questions) over the 2-hop chain, and the
  document filter then restricts dense search to the wrong papers. Hypothesis for the next dev
  cycle: route multi_hop intents to graph_vector, or score paths by query-relation coverage rather
  than per-hop decay. (Observed on test — must be validated on a fresh dev split before adoption.)

## 3. Router failures

* With the spec table (`adaptive`), 12/33 test queries went to dense-only retrieval (definitions),
  the weakest retriever on dev. The dev re-fit (`adaptive_tuned`) moved them to hybrid.
* Router accuracy against the per-query oracle is **0.64 by quality** but **0.06 by utility**
  (NDCG@5 − 0.01·cost): on average 3.6 strategies tie for the best NDCG, and the cost term awards
  every tie to the cheapest (BM25). The oracle's ceiling (0.885 NDCG@5) is far above every fixed
  system (best 0.766): per-query selection has headroom, but intent rules don't capture it.
* `adaptive_budget` (start cheap, escalate on low confidence) rarely reaches graph strategies
  because hybrid's dense cosine usually clears the confidence threshold — path recall drops to 0.39.
  The dense cosine is a poor confidence signal for relational questions.

## 4. Graph construction failures (audits)

| Audit | Precision | Main error modes |
|---|---|---|
| v1, 50 random edges | 0.74 | "we propose X" pattern attached any nearby mention (0/17 INTRODUCES correct); cue relations linked every later mention ("LoRA IMPROVES DeBERTa") |
| v2 after ADR-016 fixes | 0.78 | frequency-based USES counts mentions of baselines as usage (0.55); APPLIED_TO from abstracts too loose (1/3); a negated "without using fine-tuning" became USES |

Reliable edge types: CITES (23/23 + 12/12), INTRODUCES (77/77 reviewed), IMPROVES after the fix (4/4).

## 5. Generation / verification failures

* **Abstention:** the extractive generator refused only 1 of 4 unanswerable questions ("boiling point
  of tungsten" was answered with a loosely matching sentence). Term-coverage thresholds cannot tell
  topical overlap from answerability.
* **Lexical verification cannot catch irrelevance:** a quoted sentence is always lexically "supported"
  by the passage it quotes (support rate 0.94 on test), so the verify→re-retrieve loop rarely
  triggers with the extractive generator. NLI or LLM-judge verification against the *question* (not
  just the cited passage) is needed; both are implemented but were not run (no NLI model download
  budget was spent on the final run; no LLM keys).
* **Citation validity 0.88:** invalid cases are claim-like sentences produced by sentence-splitting
  quoted evidence containing abbreviations — a splitter issue, not a fabricated citation.

## 6. Process failures (and what changed)

* A run was corrupted by the laptop sleeping (rerank stage logged 36–94 *minutes*). Detected from an
  impossible p95, re-run, and the CLI now requests keep-awake on Windows.
* An embedded-Qdrant file lock made two concurrent evaluations collide; runs are now sequential.
* One test-split run executed unintentionally against the pre-fix graph; it is archived unviewed in
  `results/superseded/test_run_graph_v1_unviewed/`.
