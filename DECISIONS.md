# Architecture Decision Records (ADR)

## ADR-001: First-Principles Implementation Over Framework Abstractions
- **Status:** Accepted
- **Context:** Frameworks like LangChain, LlamaIndex, or Haystack wrap retrieval, prompting, and chaining inside high-level abstractions.
- **Decision:** Implement all core retrieval, ranking, fusion, routing, and verification mechanisms using direct low-level libraries (`rank-bm25`, `qdrant-client`, `networkx`, `neo4j`, `numpy`, `sentence-transformers`).
- **Consequences:** Every mathematical operation (cosine similarity, BM25 saturation, RRF scoring, path decay) is explicit, inspectable in traces, unit-testable, and directly ablative.

## ADR-002: Dual Knowledge Representation (Vector + Typed Knowledge Graph)
- **Status:** Accepted
- **Context:** Dense vector retrieval fails on multi-hop entity reasoning ("Which method derived from X improved Y?"), relationship queries, and structured provenance. Pure knowledge graphs struggle with fuzzy conceptual descriptions and nuanced continuous semantics.
- **Decision:** Maintain two synchronized representations:
  1. Structure-aware text chunks in a vector store (with page, heading, and doc metadata).
  2. Directed multigraph $G = (V, E)$ with typed entities (`Paper`, `Author`, `Method`, `Model`, `Dataset`, `Metric`, `Task`, `Concept`) and typed relations (`AUTHORED_BY`, `CITES`, `INTRODUCES`, `USES`, `EVALUATES_ON`, `IMPROVES`, `EXTENDS`, `BASED_ON`, `APPLIED_TO`, `INFLUENCES`).
- **Consequences:** Enables hybrid evidence pooling, graph-guided vector expansion, and multi-hop path extraction.

## ADR-003: Pluggable Backend with Zero-Dependency In-Memory Fallbacks
- **Status:** Accepted
- **Context:** In CI, testing, or development environments without active Docker daemons or cloud API keys, tests and demos must not fail.
- **Decision:** Define abstract protocols for all storage and model components (`VectorStore`, `GraphStore`, `Embedder`, `LLMClient`, `Reranker`) with dual implementations:
  - Vector: `QdrantVectorStore` (production) and `InMemoryVectorStore` (testing/dev).
  - Graph: `Neo4jGraphStore` (production) and `NetworkXGraphStore` (testing/dev).
  - LLM: `OpenAIClient`, `AnthropicClient`, `OllamaClient`, and `DeterministicMockLLMClient`.
- **Consequences:** Complete end-to-end pipelines, evaluation benchmarks, and test suites run completely offline and in CI in seconds, while remaining production-deployable via Docker Compose.

## ADR-004: Reciprocal Rank Fusion (RRF) for Heterogeneous Fusion
- **Status:** Accepted
- **Context:** Dense vectors produce cosine similarity scores $[ -1, 1 ]$ or $[0, 1]$, BM25 produces unbounded positive scores $[0, \infty)$, and graph path traversals produce hop-decayed confidence scores. Normalizing disparate score distributions directly can be unstable.
- **Decision:** Support both Reciprocal Rank Fusion (RRF):
  $$\text{score}_{\text{RRF}}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k_{\text{RRF}} + \text{rank}_r(d)}$$
  (default $k_{\text{RRF}} = 60$) and Min-Max / Z-score normalized weighted fusion.
- **Consequences:** Robust candidate merging across dense, lexical, and graph candidate lists without arbitrary scale calibration.

## ADR-005: Bounded Self-Correction with Explicit Retrieval Budget
- **Status:** Accepted
- **Context:** Verification loops that trigger re-retrieval can spiral into infinite loops, runaway latency, and unbounded API costs.
- **Decision:** Define an immutable `RetrievalBudget` object with hard ceilings (`max_retrieval_calls`, `max_graph_hops`, `max_candidate_pool`, `max_latency_ms`, `max_token_budget`). If claim verification fails, re-retrieval is only attempted if remaining budget permits.
- **Consequences:** Enforces guaranteed termination, strict p95 latency bounds, and clear cost attribution.
