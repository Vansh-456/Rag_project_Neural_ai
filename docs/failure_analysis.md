# Failure Mode Taxonomy & Empirical Analysis

In accordance with Section 2.1.9 of the project operating rules ("Failure cases are data. Always log and categorize failures; propose architecture changes from them"), this document catalogs systematic failure patterns identified during benchmark evaluation.

---

## 1. Failure Categories Taxonomy

| Failure Category | Mechanism | Typical Symptom | Mitigation in Adaptive System |
|---|---|---|---|
| **`VOCABULARY_MISMATCH`** | Dense vectors fail on rare acronyms or exact metric digits. | Query retrieves semantically related text missing exact target table numbers. | Hybrid lexical BM25 fusion with acronym preservation. |
| **`MULTI_HOP_CHAIN_BROKEN`** | Question requires $A \to B \to C$ traversal, but link $B \to C$ is missing in evidence. | Missing the final entity in reasoning chain. | Graph-guided query expansion with BFS traversal. |
| **`ENTITY_UNRESOLVED`** | Surface variant (e.g. "Houlsby adapters") not linked to canonical ID. | Router routes to fallback hybrid rather than dedicated graph traversal. | Canonical alias dictionary + Levenshtein fuzzy matching. |
| **`HALLUCINATED_CITATION`** | Generator synthesizes claim and appends non-existent evidence marker `[E9]`. | Invalid citation detected in trace. | Schema-level citation validator rejects answer and flags claim. |
| **`ATTENTION_DEGRADATION`** | Crucial evidence placed in middle of large prompt context is ignored by LLM. | Model outputs generic answer despite relevant chunk present. | Attention-aware "Lost-in-the-Middle" sandwich reordering. |

---

## 2. Case Studies from Benchmark Run

### Case Study 1: Multi-Hop GLUE Progression (`q_hop_01`)
- **Query:** *"Which methods derived from Transformers improved GLUE benchmark scores?"*
- **Vector-Only Failure:** Retrieved generic Transformer architecture passages from Vaswani et al. (2017) without mentioning BERT or RoBERTa.
- **Adaptive Solution:** Adaptive Router routed to `multi_hop`. Graph traversal identified path `Transformer -> EXTENDS -> BERT -> EVALUATES_ON -> GLUE`, which seeded a secondary dense query returning the Devlin et al. experimental evaluation chunk.

### Case Study 2: Inference Overhead Comparison (`q_comp_01`)
- **Query:** *"How does LoRA differ from adapters in terms of inference latency?"*
- **Graph-Only Failure:** Graph contained `LoRA -IMPROVES-> Adapters`, but lacked the continuous explanation of matrix folding vs sequential bottleneck layers.
- **Adaptive Solution:** Routed to `hybrid`. Pooled dense passages explaining low-rank matrix folding alongside adapter bottleneck latency metrics.
