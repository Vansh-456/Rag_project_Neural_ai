# Architecture Specification: Adaptive Graph + Vector RAG

## 1. High-Level Architecture & Lifecycle

The Adaptive Graph + Vector RAG system treats retrieval as an **inspectable decision problem** rather than a fixed linear pipeline. Every stage communicates across strictly typed schema contracts (`src/schemas.py`).

```
                    ┌─────────────────────────┐
                    │       User Query        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Query Analyzer      │
                    │ (Intent, Entities, Hops)│
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Adaptive Router     │
                    └──────┬───┬───┬───┬──────┘
             ┌─────────────┘   │   │   └─────────────┐
             ▼                 ▼   ▼                 ▼
      ┌─────────────┐   ┌─────────────┐   ┌──────────────────────┐
      │ Dense Vector│   │ BM25 Lexical│   │ Knowledge Graph      │
      │ Search      │   │ Search      │   │ Multi-Hop Traversal  │
      └──────┬──────┘   └──────┬──────┘   └──────────┬───────────┘
             │                 │                     │
             └─────────────┬───┴─────────────────────┘
                           │
             ┌─────────────▼─────────────┐
             │  Evidence Fusion & Pool   │
             │   (RRF / Weighted Fusion) │
             └─────────────┬─────────────┘
                           │
             ┌─────────────▼─────────────┐
             │   Cross-Encoder Reranker  │
             └─────────────┬─────────────┘
                           │
             ┌─────────────▼─────────────┐
             │      Context Builder      │
             │ (Lost-in-the-Middle, [E1])│
             └─────────────┬─────────────┘
                           │
             ┌─────────────▼─────────────┐
             │         LLM Client        │
             │   (Strict In-Context QA)  │
             └─────────────┬─────────────┘
                           │
             ┌─────────────▼─────────────┐
             │      Claim Verifier       │
             │ (Atomic Entailment Check) │
             └─────────────┬─────────────┘
                           │
                Sufficient & Supported?
               ├── YES ──► Final Answer with [E1] Citations + Trace
               └── NO  ──► Reformulate & Escalate (within Budget ceiling)
```

---

## 2. Dual Knowledge Representation

To maximize evidence coverage across diverse query types, the knowledge base maintains two synchronized representations of the corpus:

### 2.1 Structure-Aware Text Chunks (Vector & Lexical Indices)
- **Unit of retrieval:** Sections decomposed into token-bounded passages ($512$ tokens target, $64$ token overlap).
- **Context preservation:** Each chunk prepends Document Title and Section Breadcrumbs (`Document: X \n Section: Y`) to ensure isolated chunks retain semantic grounding.
- **Dual indexing:** Chunks are simultaneously embedded into dense continuous space ($384$-d via BGE/MiniLM) and indexed lexically via BM25Okapi ($k_1 = 1.5, b = 0.75$) with acronym preservation.

### 2.2 Directed Knowledge Multigraph $G = (V, E)$
- **Vertices ($V$):** Typed entities extracted from papers:
  - `Paper`, `Author`, `Method`, `Model`, `Dataset`, `Metric`, `Task`, `Concept`
- **Edges ($E$):** Typed, directed relational dependencies:
  - `AUTHORED_BY`, `CITES`, `INTRODUCES`, `USES`, `EVALUATES_ON`, `IMPROVES`, `EXTENDS`, `BASED_ON`, `APPLIED_TO`, `INFLUENCES`
- **Path Scoring with Hop Decay:** For a path $P = (v_0, e_1, v_1, \dots, e_k, v_k)$:
  $$\text{Score}(P) = \prod_{i=1}^k \left( \gamma \cdot \text{conf}(e_i) \right)$$
  where $\gamma = 0.85$ is the geometric decay factor and $\text{conf}(e) \in [0, 1]$ is edge confidence.

---

## 3. Evidence Fusion Mechanics

### 3.1 Reciprocal Rank Fusion (RRF)
When pooling heterogeneous candidate sets (dense embeddings, BM25 scores, graph path scores), disparate score scales are merged using rank order:
$$\text{RRF}(d) = \sum_{r \in \mathcal{R}} \frac{1}{k_{\text{RRF}} + \text{rank}_r(d)}$$
where $k_{\text{RRF}} = 60$ prevents top-rank saturation.

### 3.2 Attention-Aware Context Packing (Lost-in-the-Middle)
Transformer language models exhibit positional bias, disproportionately attending to information at the extreme boundaries of their context window. The `ContextBuilder` orders deduplicated evidence using an alternating sandwich sort:
$$\text{Order} = [E_1, E_3, E_5, \dots, E_6, E_4, E_2]$$
ensuring the highest-ranked evidence items occupy the prompt header and prompt footer.

---

## 4. Formal Retrieval Budget Optimization

Retrieval and self-correction actions are governed by an explicit constrained utility objective:
$$\max_{\pi} \mathbb{E} \left[ Q(A, E) - \lambda \cdot C(R) \right]$$
where:
- $\pi$ is the retrieval policy.
- $Q(A, E) \in [0, 1]$ is the measured evidence and answer faithfulness.
- $C(R) = \sum_{a \in R} c(a)$ is total retrieval cost (latency, tokens, API fees).
- $\lambda$ is the trade-off parameter.

Ceilings are strictly enforced by the `RetrievalBudget` object:
- `max_retrieval_calls = 5`
- `max_graph_hops = 3`
- `max_candidate_pool = 50`
- `max_latency_ms = 5000.0`
- `max_token_budget = 4096`
