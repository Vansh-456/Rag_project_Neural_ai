# PROJECT_STATE

## Current Phase
Phase 17 — Productionization & Deployment — status: gate passed (All Acceptance Criteria Met & Tested)

## Environment
- **OS:** Windows 11 (PowerShell, Docker Desktop available)
- **Python:** 3.11.0 (virtualenv `.venv` active)
- **Node.js:** v24.15.0 / npm 11.12.1
- **Vector DB:** Qdrant (client installed, in-memory + Docker endpoints configured)
- **Graph DB:** NetworkX (in-memory test & fast path) + Neo4j (Docker configuration)
- **LLM Providers:** Agnostic `LLMClient` supporting OpenAI, Anthropic, Ollama, and offline Deterministic Mock
- **Budget:** Controlled via explicit `RetrievalBudget` objects with parameterizable ceilings

## Completed Milestones (Phases 0 through 17)
- **Phase 0:** Defined research question, query taxonomy, 50-paper foundational corpus, and evaluation metrics in `docs/problem_definition.md`.
- **Phase 1:** Repo skeleton initialized with full type contracts, strict schemas, and configuration models.
- **Phase 2:** Document ingestion pipeline with cleaner, section hierarchy detection, and parse quality reporting.
- **Phase 3:** Four chunking strategies implemented behind a unified interface: Fixed, Recursive, Semantic, Structure-Aware.
- **Phase 4:** Dense vector retrieval baseline with BGE embeddings, caching, L2-normalization, and Qdrant / InMemory vector stores.
- **Phase 5:** 50-query evaluation dataset operational across 5 query types with Recall@K, Precision@K, MRR, NDCG@K, hit-rate, and bootstrap CIs.
- **Phase 6:** BM25 lexical retrieval and Reciprocal Rank Fusion (RRF) & Weighted Score Fusion.
- **Phase 7:** Cross-encoder reranker with candidate pool slicing and latency profiling.
- **Phase 8:** Knowledge graph construction with typed entities/relations, alias normalization, and provenance.
- **Phase 9:** Graph retrieval with 1-hop / multi-hop traversals, path decay, and serialization into textual evidence.
- **Phase 10:** Heterogeneous evidence pooling and multi-hop graph-to-vector expansion.
- **Phase 11:** QueryAnalyzer and Adaptive Router selecting optimal strategies based on query taxonomy.
- **Phase 12:** Context construction with token budget packing, lost-in-the-middle ordering, and explicit `[E1]` citation IDs.
- **Phase 13:** Provider-agnostic LLM client (OpenAI, Anthropic, Ollama, Mock) with forced citations and citation validator.
- **Phase 14:** Atomic claim extraction, NLI entailment checking, contradiction detection, and sufficiency verification.
- **Phase 15:** Retrieval budget enforcement with formal $\max_{\pi} \mathbb{E}[Q(A, E) - \lambda C(R)]$ objective.
- **Phase 16:** Final ablation study across all 8 systems with bootstrap confidence intervals and failure categorization.
- **Phase 17:** Turnkey deployment with FastAPI API, interactive 9-panel research dashboard, Dockerfile, and docker-compose.yml.

## Key Decisions (see DECISIONS.md)
- Low-level first-principles retrieval without black-box framework abstractions — 2026-10-14
- Dual vector chunk and knowledge graph representation with provenance tracking — 2026-10-14
- Pluggable backend architectures with zero-dependency in-memory testing fallbacks — 2026-10-14
- Reciprocal Rank Fusion (RRF) for robust heterogeneous candidate merging — 2026-10-14
- Formal retrieval budget enforcement with bounded self-correction — 2026-10-14

## Current Best Configs
- **Chunking:** Structure-aware (512 tokens, 64 token overlap, heading prepended)
- **Embedder:** BAAI/bge-small-en-v1.5 / DeterministicMockEmbedder (384-d, L2-normalized)
- **Retriever:** Adaptive router selecting between Dense, BM25, Hybrid, and Graph Multi-Hop
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2 (with feature-based fallback)

## Latest Metrics (Test set of 50 queries across 5 taxonomy types)
| System | Recall@5 | 95% Bootstrap CI | MRR | NDCG@5 | Hit-Rate | p95 Latency |
|---|---|---|---|---|---|---|
| Vector | 0.6267 | [0.520, 0.737] | 0.6220 | 0.8001 | 0.7800 | 18.3 ms |
| BM25 | 0.8900 | [0.823, 0.950] | 0.9067 | 1.4230 | 0.9800 | 26.1 ms |
| Hybrid | 0.8800 | [0.807, 0.943] | 0.8683 | 1.2601 | 0.9800 | 25.0 ms |
| Graph | 0.6900 | [0.577, 0.797] | 0.7400 | 1.2135 | 0.8000 | 11.4 ms |
| Multi-Hop | 0.8233 | [0.727, 0.907] | 0.7373 | 1.0333 | 0.9200 | 38.3 ms |
| Adaptive | 0.8033 | [0.710, 0.887] | 0.8047 | 1.2575 | 0.9000 | 19.8 ms |
| Adaptive+Verified | 0.8033 | [0.710, 0.887] | 0.8047 | 1.2575 | 0.9000 | 22.3 ms |

## Test Suite Status
- **25 / 25 pytest tests passing (100%)**
- Execution time: 1.99s
- Integration coverage: API endpoints, chunking, graph traversal, RRF fusion, query analyzer, reranker, claim verifier, and metrics calculations.
