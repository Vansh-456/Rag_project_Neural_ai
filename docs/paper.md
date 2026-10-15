# When does a knowledge graph help retrieval? An inspectable study of adaptive graph + vector RAG over RAG research papers

*Draft write-up of v1. All numbers are reproducible from `results/` (see `docs/experiments.md`).*

## Abstract

We build an inspectable retrieval-augmented generation system that can route each query to dense,
lexical, hybrid, knowledge-graph, graph+vector or multi-hop (graph→vector) retrieval, and evaluate it
on 79 RAG/LLM papers with 71 queries spanning definition, relationship, comparison, multi-hop and
experimental questions. Relevance is defined by verbatim anchor phrases, so chunking strategies can be
compared on identical ground truth. On a frozen test split, fusing dense, BM25 and graph-grounded
chunks (graph+vector) is the strongest system (NDCG@5 0.766 vs 0.621 for dense-only, p = 0.003),
driven by relationship questions, where typed edges grounded in source passages retrieve evidence
that text retrievers miss entirely. In contrast, rule-based adaptive routing — the hypothesis that
motivated the project — does not beat always-hybrid retrieval (Δ = +0.017, p = 0.70) and costs more,
although a per-query oracle shows large headroom (0.885). We also find that lexical retrieval
dominates a small dense embedder on this acronym-heavy corpus on dev, that cross-encoder reranking
gains on dev do not transfer to test, and that lexical claim verification cannot detect irrelevant
answers. We release the code, corpus list, labels, audits and every experiment record.

## 1. Introduction

RAG systems usually fix one retrieval method. Questions about research literature, however, mix
semantic needs ("what is LoRA?") with relational ones ("who introduced LoRA?", "which methods based on
DPR were evaluated on Natural Questions?"). We ask whether choosing and combining semantic and
relational retrieval per query improves evidence quality at controlled cost, and we instrument every
stage so that each claim is backed by a measurement.

## 2. System

* **Ingestion.** Layout-aware PDF parsing (column ordering, heading detection from font/weight and
  numbering, header/footer and hyphenation cleanup); structure-aware chunks of ≤256 tokens prefixed
  with "title > section".
* **Text retrieval.** bge-small-en-v1.5 dense retrieval (embedded Qdrant), a from-scratch BM25 with
  compound/acronym tokenisation, reciprocal rank fusion (k = 60), optional MiniLM cross-encoder
  reranking.
* **Knowledge graph.** 680 typed entities (Paper, Author, Method, Model, Dataset, Metric, Task,
  Concept) and 2,684 typed relations extracted by a seed alias table, cue patterns, mention
  statistics and reference-list resolution; entity resolution with conservative merging. Every edge
  carries the chunk it was extracted from (or its reference/metadata origin). Manual audit precision
  0.78 on 50 random edges; INTRODUCES 77/77; CITES 35/35 across two audits.
* **Graph retrieval.** Query entity linking (exact/alias/fuzzy/embedding), beam-bounded multi-hop
  traversal with score `link · Πconf · decay^(h−1) · cue_bonus · connect`, path serialisation, and
  grounding of paths to their source chunks so graph evidence is evaluated with the same chunk
  metrics as text.
* **Graph+vector.** RRF over dense, BM25 and graph-grounded chunk lists, with graph paths
  interleaved as separate evidence; vector-guided seeding when no entity links.
* **Routing.** A rule-based analyzer (intent cues, linked entities, relation cues, estimated hops)
  and a YAML intent→strategy table; a budgeted variant escalates from cheap to expensive strategies
  while retrieval confidence is low.
* **Answering.** Context packing with citation ids, a grounded LLM prompt (implemented, not used for
  the reported numbers) or an extractive quote-and-cite generator; claim verification and bounded
  re-retrieval.

## 3. Evaluation setup

71 queries (67 answerable) authored from paper text before any retrieval run; labels are
`(paper, anchor phrases, grade)`. Metrics: NDCG@5 (each label counted once), Recall@5, MRR, hit@5,
lenient document recall, path entity/relation recall for graph queries. Dev (34 answerable) was used
for development and to re-fit the router table once; test (33) was run once with the final
configuration. 95% bootstrap CIs; paired bootstrap tests against dense-only.

## 4. Results

| System | NDCG@5 (test) | Δ vs dense (p) | p50 latency |
|---|---|---|---|
| dense only | 0.621 [0.49, 0.75] | – | 52 ms |
| BM25 only | 0.678 | +0.057 (0.35) | 11 ms |
| hybrid (RRF) | 0.683 | +0.062 (0.06) | 62 ms |
| hybrid + rerank | 0.661 | +0.040 (0.52) | 5.0 s |
| graph only | 0.522 | −0.099 (0.37) | 118 ms |
| **graph + vector** | **0.766** [0.65, 0.88] | **+0.144 (0.003)** | 186 ms |
| adaptive (spec rules) | 0.684 | +0.062 (0.13) | 150 ms |
| adaptive (dev-tuned) | 0.700 | +0.079 (0.10) | 175 ms |
| adaptive + budget | 0.694 | +0.072 (0.16) | 200 ms |
| per-query oracle | 0.885 | – | – |

**Graph helps when it is fused, not alone.** Graph-only retrieval is the best system for relationship
questions by path metrics (entity recall 0.97) and significantly better than dense on them, but it
fails on definitional and experimental questions. Fusing graph-grounded chunks with text retrieval
keeps both strengths. Example: "How do REALM and RAG differ?" — every text-only system scores 0.00;
graph+vector reaches 0.76 because the graph links both methods to their introducing papers.

**Rule-based routing does not pay off.** The router's quality-only accuracy against the oracle is
0.64, but the many ties among strategies and the rules' coarse intent classes leave it no better than
always-hybrid while being more expensive. The cost-aware variant rarely escalates to graph
strategies because dense similarity is a poor confidence signal for relational questions.

**Small splits are noisy.** Dense-only scored 0.36 on dev and 0.62 on test; reranking was best on dev
and below hybrid on test. With 5–8 queries per type, only differences flagged by paired tests should
be trusted.

## 5. Negative results and failure analysis

(1) The spec's routing prior "definitions → dense" was contradicted on dev. (2) A "we propose X"
extraction pattern was 0% precise and was removed after audit. (3) The multi-hop strategy
(traverse, then dense search restricted to path documents) underperforms graph+vector on multi-hop
questions because hop decay favours direct edges. (4) Extractive answering abstained on 1 of 4
unanswerable questions, and lexical verification cannot catch irrelevant but verbatim quotes. Full
analysis: `docs/failure_analysis.md`.

## 6. Limitations

Single annotator and strict anchors (absolute scores are lower bounds); small test split; CPU-only
latencies; no LLM in the reported numbers; a gazetteer-bounded rule graph. See
`docs/limitations_and_future_work.md`.

## 7. Conclusion

On research-paper QA, a knowledge graph helps retrieval most as an additional, provenance-grounded
evidence source fused with lexical and dense retrieval, rather than as an alternative path chosen by
a router. Learned routing over a richer feature set, uncertainty-gated reranking and LLM-based
extraction are the natural next experiments, each with the same evaluation harness.
