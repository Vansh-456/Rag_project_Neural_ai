# Limitations & Future Research Directions

## 1. System Limitations

1. **Entity Extraction Coverage:** Current rule-based and canonical alias extraction captures foundational NLP entities with high precision ($>95\%$), but rare emerging paper concepts require ongoing synchronization or LLM-driven batch extraction.
2. **In-Memory Graph Scalability:** The `NetworkXGraphStore` provides rapid sub-millisecond traversal for hundreds of papers, but scaling to hundreds of thousands of papers requires transitioning to Neo4j clustered deployment.
3. **Cross-Encoder Reranker Latency:** Full cross-encoders add $15\text{--}25\text{ ms}$ of inference time. For latency-critical applications ($<10\text{ ms}$), ColBERT late interaction or lightweight learned linear fusion is preferred.

---

## 2. Future Research Directions

1. **Learned Policy Routing via Reinforcement Learning:**
   Formulate retrieval routing as a contextual bandit optimizing the reward:
   $$R = Q(A, E) - \lambda C(R)$$
   learning when to stop traversal dynamically based on partial graph confidence.

2. **Temporal Knowledge Graph RAG:**
   Add timestamp annotations to relation edges (`INTRODUCED_IN_YEAR`, `SUPERSEDED_BY`) to support chronological queries (e.g., "What was the state of the art in dense retrieval prior to 2022?").

3. **Diversity Optimization (MMR / DPP):**
   Incorporate Determinantal Point Processes (DPP) or Maximal Marginal Relevance into candidate pooling to minimize redundant citations from the same paper sections.
