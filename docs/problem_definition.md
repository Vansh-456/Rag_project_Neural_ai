# Problem definition (Phase 0)

## Research question

> Given a query, can a system intelligently choose and combine semantic (vector) retrieval and
> relational (graph) retrieval to improve evidence quality for factual, comparative, relational and
> multi-hop questions, while controlling latency and retrieval cost?

Retrieval is treated as a **decision problem**: a router reads a query analysis (intent, linked
entities, relation cues, estimated hops) and chooses among `vector`, `bm25`, `hybrid`, `graph`,
`graph_vector` and `multi_hop` strategies, each with its own cost.

## Query taxonomy

| Type | Example | Preferred strategy (hypothesis) | Why |
|---|---|---|---|
| Definition | "What is LoRA?" | vector | answer lives in one explanatory passage |
| Relationship | "Who introduced LoRA?" | graph | answer is an explicit typed edge (INTRODUCES, AUTHORED_BY) |
| Comparison | "How does LoRA differ from adapters?" | graph + vector | graph identifies both entities, text explains the difference |
| Multi-hop | "Which methods based on DPR were evaluated on Natural Questions?" | graph -> vector | requires chaining relations across papers |
| Experimental | "What results show DPR outperforms BM25?" | vector + rerank | needs specific numeric passages; precision matters |

The evaluation set also contains **unanswerable** queries (abstention) and **adversarial** ones
(acronym variants, typos, negation, paraphrase).

## Corpus

79 arXiv papers on RAG/LLM research in 9 topics (`data/corpus/papers.yaml`): foundations,
dense/sparse retrieval, embeddings, reranking, retrieval-augmented LMs, adaptive/iterative RAG,
graph RAG, parameter-efficient fine-tuning, evaluation/benchmarks. Every expected title is checked
against arXiv metadata at download time; ids marked `verify: true` were uncertain when the list was
written and **all 79 matched** (lowest similarity 0.877: C-Pack, whose arXiv title was revised).

## Environment (as found at build time)

| Item | Value |
|---|---|
| OS | Windows 11 Pro (10.0.26300); Docker images are Linux |
| Python | 3.11 |
| GPU | none — everything runs on CPU (12 logical cores, 16 GB RAM) |
| Docker | installed; daemon not running during the build, so the embedded Qdrant / NetworkX backends were used for all experiments |
| LLM API keys | none present — default generator is extractive, default graph extractor is rule-based; Anthropic / OpenAI / Ollama are drop-in via config |
| LLM budget | assumed **$0** for the reported experiments |

## Success metrics and targets

Retrieval is evaluated independently of generation, per query type, with 95% bootstrap CIs.

| Metric | Baseline target (vector-only, dev) | Rationale |
|---|---|---|
| Recall@5 | >= 0.60 | most answerable questions should surface a relevant passage in 5 |
| NDCG@5 | >= 0.45 | graded relevance; rewards putting the best passage first |
| MRR | >= 0.45 | first relevant passage typically in the top 2-3 |
| Parse success (sections identified) | >= 95% of corpus | prerequisite for structure-aware chunking |
| Graph edge precision (manual audit, 50 edges) | >= 0.70 | below this, graph retrieval injects noise |
| Router accuracy vs oracle | report; no target | the oracle is a ceiling, not an expectation |
| Adaptive vs always-hybrid | match quality at lower cost on >= 1 query type | the central claim |

## Constraints

* CPU only: embedding the corpus takes ~20-30 minutes per chunking configuration; the chunking ×
  embedder grid is therefore kept small and the choice of `bge-small-en-v1.5` is driven by cost.
* No paid LLM calls in the reported numbers. Generation metrics use the extractive generator; the
  LLM generator, LLM extractor and LLM verifier are implemented and tested with fakes.
* Ground truth is authored by one annotator (the system builder) — see risks.

## Risks

| Risk | Mitigation |
|---|---|
| Single-annotator labels bias the evaluation | labeling guideline, anchor-based labels anyone can audit, `agvrag label` tool for re-labelling, dev/test split |
| Small test set (≈33 queries) → wide CIs | report CIs and paired-bootstrap p-values; do not claim unstarred differences |
| Rule-based graph has low recall | measured edge counts + audit sample; LLM extractor available behind the same interface |
| Labels written with knowledge of the retrieval system | anchors were chosen from paper text *before* any retrieval run on the real index |
| PDF parsing errors propagate | parse-quality report per paper, parser comparison, unit tests on synthetic two-column PDFs |
