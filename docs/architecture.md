# Architecture

## Data flow

```
                 ┌────────────── offline (agvrag bootstrap) ──────────────┐
 arXiv API ──► PDFs ──► Parser ──► Documents ──► Chunker ──► Chunks ──┬──► Embedder ──► Qdrant / memory
 (rate-limited,         (layout-aware:          (4 strategies,        ├──► BM25 inverted index (built at load)
  title-verified)        columns, headings,      deterministic ids)   └──► Rule / LLM extractor ──► Resolver ──► Graph
                         refs, cleaning)                                       (provenance on every edge)   (NetworkX / Neo4j)
                 └─────────────────────────────────────────────────────────┘

 query ──► QueryAnalyzer ──► Router ──► Retriever(strategy) ──► [Reranker] ──► ContextBuilder ──► [Generator] ──► [Verifier]
            intent, linked     rule-based   vector | bm25 | hybrid             token budget,       cites [E#]/[G#]    claim support,
            entities, cues,    (YAML)       graph | graph_vector |             dedup, ordering     or abstains        sufficiency ──┐
            hops, complexity                multi_hop                                                                             │
                                  ▲                                                                                               │
                                  └─────────────── reformulate / escalate strategy / larger k (bounded by Budget) ◄───────────────┘
 every stage writes into a PipelineTrace (lists, scores, paths, latency per stage, tokens, cost units, retries)
```

## Contracts

All components exchange the typed objects in `src/agvrag/schemas.py` (`Document`, `Section`,
`Chunk`, `Evidence`, `QueryAnalysis`, `RoutingDecision`, `PipelineTrace`, `Entity`, `Relation`,
`GraphPath`) and implement the Protocols in `src/agvrag/interfaces.py`:

| Protocol | Real implementation(s) | Test / fake implementation |
|---|---|---|
| Parser | `PyMuPDFLayoutParser` | `PyMuPDFPlainParser` (baseline), synthetic PDFs in tests |
| Chunker | fixed, recursive, semantic, structure_aware | same (deterministic) |
| Embedder | `SentenceTransformerEmbedder` + `CachedEmbedder` | `HashingEmbedder` |
| VectorStore | `QdrantVectorStore` (server or embedded) | `InMemoryVectorStore` |
| SparseIndex | `ScratchBM25` | `RankBM25Index` (cross-check) |
| GraphStore | `Neo4jGraphStore` | `NetworkXGraphStore` |
| Retriever | Dense, Sparse, Hybrid, Graph, GraphVector, MultiHop | same over fakes |
| Router | `RuleRouter` | `FixedRouter` |
| Reranker | `CrossEncoderReranker` | `LexicalReranker`, `IdentityReranker` |
| ContextBuilder | `StructuredContextBuilder` | same |
| LLMClient | `AnthropicClient`, `OpenAIClient`, `OllamaClient`, `CachedLLMClient` | `FakeLLMClient` |
| Generator | `LLMGenerator` | `ExtractiveGenerator` (no LLM) |
| Verifier | `ClaimVerifier` (lexical / NLI cross-encoder / LLM judge) | lexical |

**A pipeline is a configuration.** `configs/experiments/*.yaml` differ only in the `pipeline`,
`router`, `retrieval`, `verification` and `budget` sections; `RAGPipeline` wires the same
components accordingly. This is what makes ablations controlled: the corpus, index, graph,
embedder and evaluation code are identical across systems.

## Module map

| Module | Responsibility |
|---|---|
| `ingestion/` | arXiv download, PDF parsing, cleaning, parse-quality metrics |
| `chunking/` | four chunkers + chunk statistics |
| `embeddings/` | embedders + persistent embedding cache |
| `vector_store/` | Qdrant and in-memory stores |
| `graph/` | gazetteer, extraction (rule/LLM), entity resolution, stores, linking, traversal |
| `retrieval/` | BM25, fusion (RRF / weighted), retrievers for each strategy |
| `routing/` | query analyzer + routers |
| `ranking/` | rerankers |
| `context/` | dedup, MMR, ordering, token-budgeted packing |
| `generation/` | LLM clients, grounded prompts, citation parsing/validation, extractive generator |
| `verification/` | claim extraction, support checks, sufficiency |
| `pipeline/` | budget, component registry, indexing, the pipeline |
| `evaluation/` | metrics, bootstrap, dataset, runner/registry, oracle, ablation report, labelling tool |
| `api/` | FastAPI service, trace store, caches, rate limiting |

## Key design choices (details in `DECISIONS.md`)

* **Label-based, chunking-independent ground truth** (anchors) so chunkers can be compared fairly.
* **Graph evidence is grounded**: every path carries the source chunk ids of its edges, so graph
  retrieval is evaluated with the same chunk-level metrics as text retrieval, plus path metrics.
* **Paths travel alongside chunks**: fusion ranks chunks; graph paths are interleaved as separate
  evidence that does not consume the chunk budget `k`.
* **Budgets are enforced before actions** (`BudgetTracker.charge` raises), and every retry is logged
  in `PipelineTrace.iterations`.
* **Deterministic offline defaults**: token counting, hashing embedder, extractive generator and
  rule extractor make the full test suite run offline in under a minute.

## Deployment topology (docker compose)

```
browser ──► frontend (Next.js, :3000) ──/api/* proxy + server-side basic auth──► api (FastAPI, :8000)
                                                                                   ├──► qdrant (:6333)
                                                                                   └──► neo4j  (:7687)
```

The API loads pipelines lazily and shares heavy components (embedder, indexes, graph, models)
through `ComponentRegistry`. Traces are persisted in SQLite; query results, embeddings and LLM
responses are cached with hit-rate metrics at `/metrics`.
