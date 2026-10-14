# Experimental Design & Pipeline Configurations

This document specifies the exact configurations, hyperparameters, and hypotheses governing each of the 8 evaluated retrieval systems.

---

## 1. System Inventory & Experimental Hypotheses

| System ID | Name | Core Components | Primary Hypothesis |
|---|---|---|---|
| **EXP-01** | `vector_only` | Dense BGE Embedder + Cosine Vector Store | Fast baseline; strong on definitions, weak on relational and multi-hop queries. |
| **EXP-02** | `bm25_only` | BM25Okapi ($k_1=1.5, b=0.75$) | Strong on exact keyword/acronym matching, zero semantic generalization. |
| **EXP-03** | `hybrid` | Dense + BM25 with Reciprocal Rank Fusion ($k=60$) | Combines semantic and lexical signals; mitigates vocabulary mismatch. |
| **EXP-04** | `hybrid_rerank`| Hybrid pool ($N=20$) + Cross-Encoder Reranker | Improves MRR and NDCG by modeling full query-document token interactions. |
| **EXP-05** | `graph_only` | BFS/DFS traversal with hop decay ($\gamma=0.85$) | High precision on author/provenance queries; zero definitional text coverage. |
| **EXP-06** | `multi_hop` | Graph path discovery $\to$ secondary dense vector expansion | Solves multi-step entity progression questions that vector retrieval fails on. |
| **EXP-07** | `adaptive` | QueryAnalyzer $\to$ taxonomy-driven Dynamic Router | Matches or exceeds hybrid quality while minimizing unnecessary graph/rerank calls. |
| **EXP-08** | `adaptive_verified`| Adaptive pipeline + Claim Verifier + Self-Correction | Enforces $\ge 80\%$ claim support; triggers bounded retry loop on failure. |

---

## 2. Reproducibility & Execution Commands

Every experiment is fully reproducible via CLI:

```bash
# Evaluate a single pipeline:
python scripts/benchmark.py --pipeline adaptive --top_k 5

# Evaluate all 8 pipelines and generate the ablation table:
python scripts/benchmark.py --pipeline all --top_k 5

# Inspect a single query in detail:
python scripts/query.py "What is LoRA?" --pipeline adaptive
```
