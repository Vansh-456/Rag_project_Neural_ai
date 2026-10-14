# Adaptive Graph + Vector RAG

> **An inspectable, portfolio-grade Retrieval-Augmented Generation (RAG) research platform built from first principles.**  
> Simultaneously a deep learning vehicle for RAG internals, an empirical research testbed with formal ablations, and a deployment-ready production service.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Tests Passing](https://img.shields.io/badge/tests-25%20passed%20(100%25)-emerald.svg)](tests/)
[![Docker Ready](https://img.shields.io/badge/docker-ready-2496ED.svg)](docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. System Overview

Standard RAG architectures rely on static vector similarity over arbitrary text chunks. While vector retrieval works well for definitional questions, it systematically breaks on **relational inquiries** ("Who introduced FiD?"), **comparative reasoning** across entities, and **multi-hop questions** requiring entity chaining ("Which methods derived from Transformers improved GLUE?").

This system treats retrieval as an **adaptive decision problem**:

```
                              User Query
                                  │
                     Query Understanding & Intent
                    (Definition | Relational | Multi-Hop)
                                  │
                       Adaptive Retrieval Router
               ┌──────────────────┼──────────────────┐
               ▼                  ▼                  ▼
       Dense Vector Search   BM25 Lexical   Knowledge Graph Traversal
       (Structure-Aware)      (Acronyms)      (Hops with Decay)
               └──────────────────┬──────────────────┘
                                  ▼
                      Heterogeneous Evidence Pool
                     (Reciprocal Rank Fusion - RRF)
                                  │
                        Cross-Encoder Reranker
                                  │
                       Attention-Aware Context
                     (Lost-in-the-Middle Sandwich)
                                  │
                            LLM Generation
                     (Forced [E1] In-Context Citations)
                                  │
                         Atomic Claim Verifier
                    (Entailment & Sufficiency Check)
                                  │
                   Sufficient? ──► YES ──► Verified Answer + Trace
                               └─► NO  ──► Bounded Retry (within Budget)
```

---

## 2. Core Architectural Principles (Zero Black-Box Abstractions)

- **No Framework Hiding:** Implemented without LangChain, LlamaIndex, or Haystack for core retrieval/ranking logic. Every mathematical operation (cosine similarity, BM25 saturation, RRF scoring, path decay) is explicit and inspectable.
- **Dual Knowledge Representation:**
  - **Vector / Semantic:** Structure-aware text chunks (preserving section headings and document titles as context breadcrumbs) embedded into continuous dense space.
  - **Relational Multigraph:** Typed entity nodes (`Paper`, `Author`, `Method`, `Model`, `Dataset`, `Task`) and typed edges (`AUTHORED_BY`, `CITES`, `INTRODUCES`, `USES`, `EVALUATES_ON`, `IMPROVES`, `EXTENDS`, `BASED_ON`).
- **Reciprocal Rank Fusion (RRF):**
  $$\text{score}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k_{\text{RRF}} + \text{rank}_r(d)}$$
  Robustly merges dense, lexical, and graph candidate lists without arbitrary scale calibration.
- **Attention-Aware Context Packing:** Re-orders evidence using an alternating sandwich sort to place top-ranked items at the start and end of prompt context, mitigating transformer "Lost-in-the-Middle" degradation.
- **Formal Retrieval Budget Object:** Constrains re-retrieval loops using the formal utility objective:
  $$\max_{\pi} \mathbb{E} \left[ Q(A, E) - \lambda \cdot C(R) \right]$$
  guaranteeing bounded latency and strict termination.

---

## 3. Headline Ablation Results (50 Research Queries)

Evaluated over a curated 50-query benchmark spanning Definitional, Relational, Comparative, Multi-Hop, and Experimental query taxonomies. 95% Confidence Intervals computed using **1,000 bootstrap resamples**:

| System Variant | Recall@5 | 95% Bootstrap CI | MRR | NDCG@5 | Hit-Rate@5 | p95 Latency (ms) |
|---|---|---|---|---|---|---|
| **Vector Baseline** | 0.6267 | [0.520, 0.737] | 0.6220 | 0.8001 | 0.7800 | 18.3 ms |
| **BM25 Lexical** | 0.8900 | [0.823, 0.950] | 0.9067 | 1.4230 | 0.9800 | 26.1 ms |
| **Hybrid (Dense + BM25)** | 0.8800 | [0.807, 0.943] | 0.8683 | 1.2601 | 0.9800 | 25.0 ms |
| **Hybrid + Reranking** | 0.8800 | [0.807, 0.943] | 0.8683 | 1.2601 | 0.9800 | 28.8 ms |
| **Graph Traversal Only** | 0.6900 | [0.577, 0.797] | 0.7400 | 1.2135 | 0.8000 | 11.4 ms |
| **Multi-Hop (Graph → Vector)**| 0.8233 | [0.727, 0.907] | 0.7373 | 1.0333 | 0.9200 | 38.3 ms |
| **Adaptive Router** | **0.8033** | [0.710, 0.887] | **0.8047** | **1.2575** | **0.9000** | **19.8 ms** |
| **Adaptive + Verified** | **0.8033** | [0.710, 0.887] | **0.8047** | **1.2575** | **0.9000** | **22.3 ms** |

---

## 4. Quickstart Guide

### 4.1 Local Setup (< 2 minutes)

```bash
# 1. Clone & create virtual environment
git clone https://github.com/example/adaptive-graph-vector-rag.git
cd adaptive-graph-vector-rag
python -m venv .venv

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# 2. Install dependencies & editable package
pip install -r requirements.txt
pip install -e .

# 3. Seed foundational papers & evaluation dataset
python scripts/seed_database.py

# 4. Launch FastAPI server & interactive Web Dashboard
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to **`http://localhost:8000/`** to access the research dashboard.

---

### 4.2 Docker Compose Deployment (Turnkey Production)

```bash
# Start Backend API, Qdrant Vector DB, and Neo4j Graph DB:
docker compose up -d

# Verify system health:
curl http://localhost:8000/health
```

---

## 5. Interactive Research Dashboard (All 9 Required Panels)

Open `http://localhost:8000/` in any browser:
1. **Query Input & Pipeline Selector:** Switch dynamically between Vector, BM25, Hybrid, Graph, Multi-Hop, Adaptive, and Adaptive+Verified.
2. **Router Decision Panel:** Real-time intent classification, linked entities, complexity score, and router rationale.
3. **Retrieval & Evidence Pool:** Inspect raw retriever scores, RRF ranks, and cross-encoder re-rankings.
4. **Grounded Answer Panel:** Strict in-context answers with clickable `[E1]`, `[E2]` citation tags.
5. **Interactive Source Inspector:** Click any citation marker to view the full source chunk, document title, and page.
6. **Claim Verification Panel:** Real-time atomic claim extraction with support status and contradiction detection.
7. **Latency & Budget Breakdown:** Microsecond-level stage timings (`routing_ms`, `retrieval_ms`, `reranking_ms`, `verification_ms`) and token cost attribution.
8. **Interactive Multigraph Visualizer:** Dynamic canvas representation of entities and relations with degree metrics.
9. **Side-by-Side Comparator:** Run identical queries simultaneously through two different architectures (e.g. Vector Baseline vs Adaptive Verified).

---

## 6. Command-Line Interface (CLI)

### Interactive Query Inspection
```bash
python scripts/query.py "What is LoRA?" --pipeline adaptive
python scripts/query.py "Who introduced LoRA?" --pipeline graph
python scripts/query.py "How does LoRA differ from adapters?" --pipeline hybrid
```

### Automated Benchmark Suite
```bash
# Run benchmark across all 8 configurations:
python scripts/benchmark.py --pipeline all --top_k 5
```

---

## 7. Running Tests

```bash
# Run full unit and integration test suite:
pytest -v
```
All 25 tests pass in $<2\text{ seconds}$ with zero external dependencies required.

---

## 8. Repository Structure

```
adaptive-graph-vector-rag/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── pyproject.toml                     # Package metadata & build config
├── requirements.txt                   # Pip dependency manifest
├── Dockerfile                         # Production container image
├── docker-compose.yml                 # Multi-service stack (App, Qdrant, Neo4j)
├── PROJECT_STATE.md                   # Session memory & tracking
├── DECISIONS.md                       # Architecture Decision Records (ADRs)
├── configs/
│   ├── base.yaml                      # Default pipeline configuration
│   ├── development.yaml               # Dev overrides (in-memory fast path)
│   ├── production.yaml                # Production overrides (Qdrant + Neo4j)
│   └── experiments/                   # 8 ablation YAML configurations
├── data/
│   ├── processed/documents/           # Structured Document JSON corpus
│   ├── processed/chunks/              # Structure-aware chunks
│   ├── processed/entities/            # Knowledge graph entities/relations
│   └── evaluation/                    # 50 benchmark queries & ground truth
├── src/
│   ├── schemas.py                     # Strict dataclasses, models & protocols
│   ├── config.py                      # YAML & Pydantic settings loader
│   ├── ingestion/                     # Academic paper cleaner & parser
│   ├── chunking/                      # Fixed, Recursive, Semantic, Structure-Aware
│   ├── embeddings/                    # SentenceTransformers & Deterministic Mock
│   ├── vector_store/                  # Qdrant client & InMemoryVectorStore
│   ├── graph/                         # NetworkX multigraph, Neo4j, Alias normalizer
│   ├── retrieval/                     # Dense, BM25, RRF Fusion, Graph, Multi-Hop
│   ├── routing/                       # QueryAnalyzer & AdaptiveRouter
│   ├── ranking/                       # Cross-encoder reranker
│   ├── context/                       # ContextBuilder & Lost-in-the-Middle packer
│   ├── generation/                    # Agnostic LLMClient & Citation validator
│   ├── verification/                  # Atomic claim extractor & entailment checker
│   ├── pipeline/                      # Composable UnifiedRAGPipeline
│   ├── evaluation/                    # Recall@K, NDCG@K, MRR, bootstrap CIs
│   └── api/                           # FastAPI service & interactive web UI
├── scripts/
│   ├── seed_database.py               # Generates foundational corpus & eval data
│   ├── query.py                       # CLI inspection query runner
│   ├── benchmark.py                   # Automated benchmark evaluation runner
│   └── download_papers.py             # arXiv rate-limited paper downloader
├── tests/                             # 25 thorough unit and integration tests
└── docs/                              # Formal research specifications
```

---

## 9. License

Released under the [MIT License](LICENSE).
