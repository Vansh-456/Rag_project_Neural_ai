# Limitations and future work

## Limitations of the evidence

1. **Single annotator, draft labels.** The builder wrote all 71 queries and their anchors (from the
   paper text, before any retrieval run). No inter-annotator agreement has been measured; the
   guideline's 20% re-labelling pass is still to do. Treat absolute numbers as provisional.
2. **Strict anchors + shallow pooling.** A passage counts as relevant only if it contains a labelled
   anchor. Many "right paper, wrong passage" failures are relevant passages that were never
   labelled, so absolute recall/NDCG are **lower bounds**. The document-level `doc_recall@k` column
   gives the lenient view. Comparisons between systems are fair (identical labels), but systems
   that surface *different* relevant passages are penalised more than ones that surface the
   labelled ones.
3. **Small test split.** ≈33 answerable test queries (≈6-8 per type). Per-type differences below
   ~0.15 NDCG are rarely significant; the report stars only paired-bootstrap p < 0.05.
4. **The router table was re-fit on 34 dev queries** (ADR-015). The test split guards against
   over-fitting, but with this n the tuned table is itself noisy.
5. **CPU-only latency.** Cross-encoder reranking costs ~5 s/query at N=50 on the build machine;
   latency numbers are relative, not representative of GPU serving. One run was corrupted by the
   machine sleeping (detected, re-run; the CLI now holds a keep-awake request).
6. **No LLM in the reported numbers.** Answer-quality metrics (abstention, citation validity /
   precision, lexical support rate) come from the extractive generator; LLM-judge calibration
   against human labels (Phase 13) has not been run.

## Limitations of the system

* **Rule-based graph = gazetteer-bounded recall.** Methods/datasets not in `gazetteer.yaml` are
  invisible to the graph; frequency-based USES edges are ~55% precise (mentions ≠ usage);
  APPLIED_TO is weak. INTRODUCES (77/77) and CITES (23/23) are reliable.
* **Entity linking is lexical.** Paraphrased entity references ("the low-rank method") don't link
  unless the embedding fallback fires (threshold 0.80).
* **Multi-hop retrieval** finds direct edges more easily than 2-hop chains, because path scores
  decay per hop; chains through generic hubs ("BERT", "fine-tuning") are capped but still noisy.
* **Parsing.** Some second-level headings are folded into neighbouring sections; tables and
  equations are extracted as text (e.g. a cue rule once fired inside a garbled results table).
  GROBID was not evaluated.
* **Token counting** is a calibrated heuristic (~7% per-passage error vs `cl100k_base`); context
  budgets for a specific LLM should use that LLM's tokenizer.

## Future work (in priority order)

1. Second-annotator pass + pooled relabelling (top-10 union of all systems) → inter-annotator
   kappa, less biased recall.
2. LLM extraction (`--extractor rule+llm`) on a budget, with the edge audit repeated per extractor.
3. LLM generation + an LLM judge calibrated against ~50 hand-graded answers; measure faithfulness
   and citation precision with a real generator.
4. Learned router (logistic regression on analyzer features, trained on dev oracle labels) behind
   the same `Router` interface; compare with the rule table under the `Q − λC` objective.
5. Cheaper reranking: smaller cross-encoder or rerank only when the first-stage score margin is
   small (uncertainty-gated), measured on the quality-vs-latency Pareto plot.
6. Scale the corpus only after 1–3; the pipeline and evaluation are corpus-agnostic.
