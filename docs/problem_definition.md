# Problem Definition: Adaptive Graph + Vector RAG

## 1. Executive Summary & Research Question
Standard Retrieval-Augmented Generation (RAG) typically relies on static semantic vector search over fixed-size text chunks. While vector search excels at retrieving local topical passages and fuzzy conceptual definitions, it systematically fails on relational queries ("Who introduced X?"), multi-hop queries requiring multi-edge traversal ("Which methods derived from X improved Y?"), and global comparative reasoning across entities.

### Core Research Question
> **Given an input query, can an adaptive system intelligently determine and orchestrate semantic (vector) retrieval and relational (knowledge graph) retrieval to maximize evidence quality for factual, relational, comparative, and multi-hop queries while strictly bounding retrieval latency and inference cost?**

---

## 2. Query Taxonomy & Routing Strategy

We categorize incoming queries into five formal types, each mapped to an empirical retrieval strategy:

| Query Type | Typical Intent / Pattern | Failure Mode of Pure Vector | Preferred Strategy |
|---|---|---|---|
| **Definition** | "What is LoRA?" / "Define prefix-tuning" | None (dense search works well) | **Vector** (Dense top-K) |
| **Relationship** | "Who authored BERT?" / "Which paper introduced FiD?" | Misses explicit graph relation in noise | **Graph** (1-hop traversal) |
| **Comparison** | "How does LoRA differ from adapters?" | May retrieve only one side of comparison | **Hybrid** (Dense + BM25 + Graph entities) |
| **Multi-hop** | "Which methods derived from Transformers improved GLUE?" | Cannot chain intermediate entities | **Multi-hop** (Graph traversal $\to$ Vector evidence) |
| **Experimental Evidence** | "What quantitative improvements did ColBERT achieve on MS MARCO?" | Low lexical precision on specific metric tables | **Vector + BM25 + Cross-Encoder Rerank** |

---

## 3. Initial Target Corpus: 50 Foundational LLM & RAG Papers

The initial experimental corpus consists of 50 core peer-reviewed papers spanning 5 key thematic clusters in modern RAG and LLM architecture.

### Cluster 1: Foundational Attention & Pretrained Transformers
1. **Attention Is All You Need** — Vaswani et al. (2017) — `arXiv:1706.03762`
2. **BERT: Pre-training of Deep Bidirectional Transformers** — Devlin et al. (2018) — `arXiv:1810.04805`
3. **RoBERTa: A Robustly Optimized BERT Pretraining Approach** — Liu et al. (2019) — `arXiv:1907.11692`
4. **Exploring the Limits of Transfer Learning with T5** — Raffel et al. (2019) — `arXiv:1910.10683`
5. **Language Models are Few-Shot Learners (GPT-3)** — Brown et al. (2020) — `arXiv:2005.14165`
6. **BART: Denoising Sequence-to-Sequence Pre-training** — Lewis et al. (2019) — `arXiv:1910.13461`
7. **ELECTRA: Pre-training Text Encoders as Discriminators** — Clark et al. (2020) — `arXiv:2003.10555`
8. **DeBERTa: Decoding-enhanced BERT with Disentangled Attention** — He et al. (2020) — `arXiv:2006.03654`
9. **ALBERT: A Lite BERT for Self-supervised Learning** — Lan et al. (2019) — `arXiv:1909.11942`
10. **XLNet: Generalized Autoregressive Pretraining** — Yang et al. (2019) — `arXiv:1906.08237`

### Cluster 2: Dense & Neural Retrieval Foundations
11. **Dense Passage Retrieval for Open-Domain QA (DPR)** — Karpukhin et al. (2020) — `arXiv:2004.04906`
12. **REALM: Retrieval-Augmented Language Model Pre-Training** — Guu et al. (2020) — `arXiv:2002.08909`
13. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)** — Lewis et al. (2020) — `arXiv:2005.11401`
14. **Leveraging Passage Retrieval with Generative Models for Open Domain QA (FiD)** — Izacard & Grave (2020) — `arXiv:2007.01282`
15. **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction** — Khattab & Zaharia (2020) — `arXiv:2004.12832`
16. **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction** — Santhanam et al. (2021) — `arXiv:2112.01488`
17. **Improving Language Models by Retrieving from Trillions of Tokens (RETRO)** — Borgeaud et al. (2021) — `arXiv:2112.04426`
18. **Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)** — Gao et al. (2022) — `arXiv:2212.10496`
19. **Text and Code Embeddings by Contrasting Cluster Assignments** — Neelakantan et al. (2022) — `arXiv:2201.10005`
20. **BGE: C-Pack: Packaged Resources for General Chinese and English Embeddings** — Xiao et al. (2023) — `arXiv:2309.07597`

### Cluster 3: Knowledge Graphs & Graph-Augmented RAG
21. **From Local to Global: A Graph RAG Approach to Query-Focused Summarization** — Edge et al. (Microsoft, 2024) — `arXiv:2404.16130`
22. **KG-BART: Knowledge Graph-Augmented BART for Generative Commonsense Reasoning** — Liu et al. (2021) — `arXiv:2009.12677`
23. **QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering** — Yasunaga et al. (2021) — `arXiv:2104.06378`
24. **GreaseLM: Graph REASoning Enhanced Language Models** — Zhang et al. (2022) — `arXiv:2201.08860`
25. **Unifying Large Language Models and Knowledge Graphs: A Roadmap** — Pan et al. (2023) — `arXiv:2306.08302`
26. **Graph Neural Networks for Knowledge Graph Completion** — Schlichtkrull et al. (2018) — `arXiv:1703.06103`
27. **Knowledge-Driven Question Answering with Graph Neural Networks** — Sun et al. (2020) — `arXiv:2004.14088`
28. **MindMap: Knowledge Graph Prompting for Multi-Hop Reasoning in Large Language Models** — Wen et al. (2023) — `arXiv:2308.09729`
29. **Think-on-Graph: Deep and Responsible Reasoning of Large Language Models on Knowledge Graphs** — Sun et al. (2023) — `arXiv:2307.07697`
30. **G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering** — He et al. (2024) — `arXiv:2402.07630`

### Cluster 4: Parameter-Efficient Fine-Tuning (PEFT)
31. **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al. (2021) — `arXiv:2106.09685`
32. **Parameter-Efficient Transfer Learning for NLP (Adapters)** — Houlsby et al. (2019) — `arXiv:1902.00751`
33. **Prefix-Tuning: Optimizing Continuous Prompts for Generation** — Li & Liang (2021) — `arXiv:2101.00190`
34. **The Power of Scale for Parameter-Efficient Prompt Tuning** — Lester et al. (2021) — `arXiv:2104.08691`
35. **QLoRA: Efficient Finetuning of Quantized LLMs** — Dettmers et al. (2023) — `arXiv:2305.14314`
36. **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning** — Zhang et al. (2023) — `arXiv:2303.10512`
37. **BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked LMs** — Zaken et al. (2021) — `arXiv:2106.10199`
38. **LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention** — Zhang et al. (2023) — `arXiv:2303.16199`
39. **IA3: Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper** — Liu et al. (2022) — `arXiv:2205.05638`
40. **Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning** — Lialin et al. (2023) — `arXiv:2303.15647`

### Cluster 5: RAG Evaluation, Hallucination, & Verification
41. **RAGAS: Automated Evaluation of Retrieval Augmented Generation** — Es et al. (2023) — `arXiv:2309.15217`
42. **ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems** — Saad-Falcon et al. (2023) — `arXiv:2311.09476`
43. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** — Asai et al. (2023) — `arXiv:2310.11511`
44. **Corrective Retrieval Augmented Generation (CRAG)** — Yan et al. (2024) — `arXiv:2401.15884`
45. **Active Retrieval Augmented Generation (FLARE)** — Jiang et al. (2023) — `arXiv:2305.06983`
46. **Seven Failure Points When Engineering a Retrieval Augmented Generation System** — Barnett et al. (2024) — `arXiv:2401.05856`
47. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — Zheng et al. (2023) — `arXiv:2306.05685`
48. **Lost in the Middle: How Language Models Use Long Contexts** — Liu et al. (2023) — `arXiv:2307.03172`
49. **Faithful and Factually Grounded Question Answering** — Dziri et al. (2021) — `arXiv:2104.08728`
50. **Retrieval-Augmented Generation for Large Language Models: A Survey** — Gao et al. (2023) — `arXiv:2312.10997`

---

## 4. Evaluation Targets & Success Criteria

| Metric | Vector Baseline Target | Hybrid + Rerank Target | Adaptive Graph + Vector Target |
|---|---|---|---|
| **Recall@5** (Overall) | $\ge 0.65$ | $\ge 0.75$ | $\ge 0.85$ |
| **Recall@5** (Multi-hop) | $\le 0.40$ (Expected limit) | $\le 0.55$ | $\ge 0.78$ |
| **NDCG@5** | $\ge 0.55$ | $\ge 0.68$ | $\ge 0.79$ |
| **MRR** | $\ge 0.50$ | $\ge 0.62$ | $\ge 0.74$ |
| **Answer Faithfulness** | $\ge 70\%$ | $\ge 80\%$ | $\ge 92\%$ |
| **Citation Precision** | $\ge 75\%$ | $\ge 82\%$ | $\ge 95\%$ |
| **p95 Latency** | $\le 250\text{ ms}$ | $\le 450\text{ ms}$ | $\le 850\text{ ms}$ |

---

## 5. Environment Specifications
- **Operating System:** Windows 11 (PowerShell & Docker Desktop ready)
- **Runtime:** Python 3.11.0 + Node.js v24.15.0
- **Vector Storage:** Qdrant (Docker / In-Memory client)
- **Graph Storage:** NetworkX (in-memory test & fast path) + Neo4j (Docker)
- **LLM Integrations:** Anthropic, OpenAI, Ollama, plus deterministic offline test client
- **CI / Quality:** pytest, pytest-cov, ruff, mypy
