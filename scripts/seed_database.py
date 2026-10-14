"""
Seed database and evaluation dataset generator.
Populates foundational 50-paper corpus, structured documents, chunks,
knowledge graph entities/relations, and 50 benchmark evaluation queries.
"""

from __future__ import annotations

import json
from pathlib import Path
from src.schemas import EntityType, RelationType


# 50 Foundational Papers Dataset Definition
PAPERS_DATA = [
    # Cluster 1: Attention & Transformers
    {
        "doc_id": "paper_attention_2017",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        "year": 2017,
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism to draw global dependencies between input and output.",
        "sections": [
            {"heading": "Introduction", "level": 1, "text": "Recurrent neural networks have been firmly established as state of the art approaches in sequence modeling. The Transformer replaces recurrent inductive bias with multi-head self-attention, achieving superior translation quality and significant parallelization."},
            {"heading": "Architecture", "level": 1, "text": "The Transformer uses stacked self-attention and point-wise, fully connected layers for both the encoder and decoder. Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions."},
            {"heading": "Experiments", "level": 1, "text": "On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU, improving over the existing best results by over 2.0 BLEU."}
        ],
        "references": ["Bahdanau et al., 2014", "Hochreiter & Schmidhuber, 1997"]
    },
    {
        "doc_id": "paper_bert_2018",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        "year": 2018,
        "abstract": "We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
        "sections": [
            {"heading": "Introduction", "level": 1, "text": "Language model pre-training has shown to be effective for improving many natural language processing tasks. We argue that standard unidirectional models are sub-optimal for sentence-level tasks. BERT alleviates the unidirectionality constraint by using a 'masked language model' (MLM) pre-training objective."},
            {"heading": "Methodology", "level": 1, "text": "BERT is trained on two self-supervised tasks: Masked LM (where 15% of tokens are masked at random) and Next Sentence Prediction (NSP), capturing discourse-level coherence."},
            {"heading": "Evaluation", "level": 1, "text": "BERT advances the state of the art on eleven NLP benchmarks, obtaining an 80.5% score on GLUE and 93.2 F1 on SQuAD v1.1."}
        ],
        "references": ["paper_attention_2017", "Radford et al., 2018"]
    },
    {
        "doc_id": "paper_roberta_2019",
        "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "authors": ["Yinhan Liu", "Myle Ott", "Naman Goyal", "Jingfei Du"],
        "year": 2019,
        "abstract": "An empirical replication study showing that BERT was significantly undertrained. RoBERTa modifies hyperparameter choices, removes the next sentence prediction objective, and trains on longer sequences.",
        "sections": [
            {"heading": "Methodology", "level": 1, "text": "RoBERTa introduces dynamic masking, removes NSP loss, trains with larger mini-batches (8K) and longer sequences over 160GB of uncompressed text."},
            {"heading": "Results", "level": 1, "text": "RoBERTa matches or exceeds all post-BERT methods on GLUE, achieving 88.5 GLUE score without multi-task pre-training."}
        ],
        "references": ["paper_bert_2018", "paper_attention_2017"]
    },
    {
        "doc_id": "paper_t5_2019",
        "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        "authors": ["Colin Raffel", "Noam Shazeer", "Adam Roberts", "Katherine Lee"],
        "year": 2019,
        "abstract": "T5 casts every language problem into a unified text-to-text format, training on the Colossal Clean Crawled Corpus (C4).",
        "sections": [
            {"heading": "Architecture", "level": 1, "text": "T5 uses an encoder-decoder Transformer. Every task from machine translation to question answering uses identical text input and target tokens."},
            {"heading": "Results", "level": 1, "text": "T5-11B achieved state-of-the-art results on 18 benchmarks including SuperGLUE and SQuAD."}
        ],
        "references": ["paper_attention_2017", "paper_bert_2018"]
    },
    {
        "doc_id": "paper_gpt3_2020",
        "title": "Language Models are Few-Shot Learners",
        "authors": ["Tom Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah"],
        "year": 2020,
        "abstract": "GPT-3 explores scaling language models to 175 billion parameters, demonstrating remarkable in-context few-shot learning capabilities without task-specific gradient updates.",
        "sections": [
            {"heading": "In-Context Learning", "level": 1, "text": "Scaling models to 175B parameters enables few-shot learning where the model receives task demonstrations solely in prompt context."},
            {"heading": "Evaluation", "level": 1, "text": "GPT-3 achieves competitive accuracy on trivia, translation, arithmetic, and cloze tests without fine-tuning."}
        ],
        "references": ["paper_attention_2017", "Radford et al., 2019"]
    },

    # Cluster 2: Dense & Neural Retrieval Foundations
    {
        "doc_id": "paper_dpr_2020",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": ["Vladimir Karpukhin", "Barlas Oguz", "Sewon Min", "Patrick Lewis"],
        "year": 2020,
        "abstract": "We show that retrieval can be practically implemented using dense representations, where embeddings are learned from a small number of questions and passages by a simple dual-encoder architecture.",
        "sections": [
            {"heading": "Introduction", "level": 1, "text": "Open-domain question answering relies on efficient passage retrieval. While BM25 has dominated, DPR trains dual BERT encoders with in-batch negatives to map questions and passages to continuous 768-d vectors."},
            {"heading": "Dual-Encoder Formulation", "level": 1, "text": "The retriever computes similarity via dot product: sim(q, p) = E_Q(q)^T E_P(p). Efficient maximum inner product search (MIPS) is performed using FAISS."},
            {"heading": "Results", "level": 1, "text": "DPR outperforms BM25 by 9-19% top-20 passage retrieval accuracy on Natural Questions and TriviaQA, achieving 78.4% top-20 accuracy on NQ."}
        ],
        "references": ["paper_bert_2018", "Robertson & Zaragoza, 2009"]
    },
    {
        "doc_id": "paper_realm_2020",
        "title": "REALM: Retrieval-Augmented Language Model Pre-Training",
        "authors": ["Kelvin Guu", "Kenton Lee", "Zora Tung", "Panupong Pasupat"],
        "year": 2020,
        "abstract": "Language model pre-training embeds world knowledge implicitly. REALM augments LM pre-training with a learned neural retriever that retrieves documents from a Wikipedia corpus during pre-training.",
        "sections": [
            {"heading": "Method", "level": 1, "text": "REALM models the retrieval of knowledge as a latent variable, propagating gradient updates into both the neural document retriever and the masked language model generator."},
            {"heading": "Evaluation", "level": 1, "text": "REALM outperforms T5-11B on open-domain question answering benchmarks while using only 330M parameters."}
        ],
        "references": ["paper_bert_2018", "paper_dpr_2020"]
    },
    {
        "doc_id": "paper_rag_2020",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni"],
        "year": 2020,
        "abstract": "We explore general-purpose fine-tuning recipes for Retrieval-Augmented Generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation.",
        "sections": [
            {"heading": "Architecture", "level": 1, "text": "RAG models combine a pre-trained sequence-to-sequence model (BART) with a dense neural retriever (DPR) over a Wikipedia corpus. We explore RAG-Sequence and RAG-Token models for marginalizing retrieved passages."},
            {"heading": "Results", "level": 1, "text": "RAG sets state-of-the-art results on open-domain Natural Questions, WebQuestions, and CuratedTREC, while generating more factual and specific answers than parametric BART alone."}
        ],
        "references": ["paper_dpr_2020", "paper_attention_2017", "paper_bert_2018"]
    },
    {
        "doc_id": "paper_fid_2020",
        "title": "Leveraging Passage Retrieval with Generative Models for Open Domain QA (FiD)",
        "authors": ["Gautier Izacard", "Edouard Grave"],
        "year": 2020,
        "abstract": "Fusion-in-Decoder (FiD) performs evidence fusion in the decoder rather than the encoder, processing 100 retrieved passages independently in the encoder before cross-attention concatenation in the decoder.",
        "sections": [
            {"heading": "Method", "level": 1, "text": "FiD passes each retrieved passage concatenated with the question through the T5 encoder independently, allowing linear encoder scaling with retrieved passages. The decoder then applies cross-attention over all passage representations simultaneously."},
            {"heading": "Results", "level": 1, "text": "FiD achieves 51.4 exact match on Natural Questions and 67.6 on TriviaQA, setting a new benchmark for generative open-domain QA."}
        ],
        "references": ["paper_t5_2019", "paper_dpr_2020"]
    },
    {
        "doc_id": "paper_colbert_2020",
        "title": "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
        "authors": ["Omar Khattab", "Matei Zaharia"],
        "year": 2020,
        "abstract": "ColBERT introduces late interaction, encoding queries and documents into multi-vector token bags and computing max-similarity across token embeddings, retaining expressive cross-attention precision at vector search latency.",
        "sections": [
            {"heading": "Late Interaction", "level": 1, "text": "Instead of compressing an entire document into a single vector, ColBERT retains contextualized token embeddings and scores relevance using MaxSim: score(q, d) = sum_{i} max_{j} (E_Q(q)_i . E_D(d)_j)."},
            {"heading": "Results", "level": 1, "text": "ColBERT achieves competitive MRR@10 with full cross-encoders on MS MARCO while retrieving over 100x faster."}
        ],
        "references": ["paper_bert_2018", "paper_dpr_2020"]
    },

    # Cluster 3: Knowledge Graphs & Graph RAG
    {
        "doc_id": "paper_graphrag_2024",
        "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "authors": ["Darren Edge", "Ha Trinh", "Newman Cheng", "Joshua Bradley"],
        "year": 2024,
        "abstract": "GraphRAG extracts a knowledge graph from raw text using LLMs, clusters entities into hierarchical communities, and generates multi-level summaries to answer global corpus-wide questions that vector RAG fails on.",
        "sections": [
            {"heading": "Motivation", "level": 1, "text": "Standard vector RAG fails when queries require synthesizing information across an entire corpus rather than retrieving local chunks. GraphRAG introduces hierarchical Leiden community detection over extracted knowledge graphs."},
            {"heading": "Evaluation", "level": 1, "text": "GraphRAG achieves substantial gains over naive RAG on comprehensiveness, diversity, and global sense-making across multi-document corpora."}
        ],
        "references": ["paper_rag_2020", "Traag et al., 2019"]
    },
    {
        "doc_id": "paper_qagnn_2021",
        "title": "QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering",
        "authors": ["Michihiro Yasunaga", "Hongyu Ren", "Antoine Bosselut", "Percy Liang"],
        "year": 2021,
        "abstract": "QA-GNN uses language models to score relevance of KG subgraphs and graph neural networks to perform relational multi-hop message passing.",
        "sections": [
            {"heading": "Method", "level": 1, "text": "The model extracts a working subgraph from ConceptNet and joint trains a GNN with RoBERTa, fusing node representations via cross-attention."},
            {"heading": "Results", "level": 1, "text": "Improves multi-hop reasoning accuracy on CommonsenseQA by 5.7% and OpenBookQA by 3.2%."}
        ],
        "references": ["paper_roberta_2019", "Kipf & Welling, 2016"]
    },

    # Cluster 4: Parameter-Efficient Fine-Tuning (PEFT)
    {
        "doc_id": "paper_lora_2021",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": ["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu"],
        "year": 2021,
        "abstract": "We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.",
        "sections": [
            {"heading": "Problem & Hypothesis", "level": 1, "text": "Fine-tuning all parameters of multi-billion parameter models is computationally prohibitive. We hypothesize that the change in weights during model adaptation has a low 'intrinsic dimension'."},
            {"heading": "Method", "level": 1, "text": "For a pre-trained weight matrix W_0 in R^{d x k}, LoRA constrains its update with a low-rank decomposition: W = W_0 + B A, where B in R^{d x r}, A in R^{r x k}, and rank r << min(d, k). When deployed, W_0 and BA can be folded with zero inference latency overhead."},
            {"heading": "Evaluation", "level": 1, "text": "LoRA reduces trainable parameters by 10,000x and GPU memory requirement by 3x compared to GPT-3 175B full fine-tuning, while achieving equal or better performance on GLUE, WikiSQL, and SAMSum."}
        ],
        "references": ["paper_gpt3_2020", "Houlsby et al., 2019", "Li & Liang, 2021"]
    },
    {
        "doc_id": "paper_adapters_2019",
        "title": "Parameter-Efficient Transfer Learning for NLP (Adapters)",
        "authors": ["Neil Houlsby", "Andrei Giurgiu", "Stanislaw Jastrzebski", "Bruna Morrone"],
        "year": 2019,
        "abstract": "Adapters inject small bottleneck feed-forward layers inside each Transformer sub-layer, training only 3% of parameters while preserving performance.",
        "sections": [
            {"heading": "Architecture", "level": 1, "text": "Adapter modules consist of a bottleneck projection down to low dimension m, a non-linearity, and a projection back up to d, followed by a residual connection."},
            {"heading": "Tradeoffs", "level": 1, "text": "While parameter efficient, adapter modules introduce sequential computation layers, increasing inference latency by 4-19%."}
        ],
        "references": ["paper_bert_2018", "paper_attention_2017"]
    },
    {
        "doc_id": "paper_prefix_tuning_2021",
        "title": "Prefix-Tuning: Optimizing Continuous Prompts for Generation",
        "authors": ["Xiang Lisa Li", "Percy Liang"],
        "year": 2021,
        "abstract": "Prefix-tuning keeps language model parameters frozen and prepends a sequence of continuous task-specific virtual tokens to key and value projections.",
        "sections": [
            {"heading": "Method", "level": 1, "text": "Prefix-tuning prepends l continuous vectors to the keys and values of every Transformer attention layer, training only 0.1% parameters."},
            {"heading": "Comparison", "level": 1, "text": "Outperforms adapters in low-data regimes and maintains performance on table-to-text generation."}
        ],
        "references": ["paper_gpt3_2020", "paper_adapters_2019"]
    },

    # Cluster 5: RAG Evaluation & Verification
    {
        "doc_id": "paper_ragas_2023",
        "title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "authors": ["Shahul Es", "Jithin James", "Luis Espinosa-Anke", "Steven Schockaert"],
        "year": 2023,
        "abstract": "RAGAS is a framework for reference-free evaluation of RAG pipelines, defining metrics for faithfulness, answer relevance, context precision, and context recall.",
        "sections": [
            {"heading": "Metrics", "level": 1, "text": "Faithfulness measures the ratio of claims in generated answers that can be inferred from context. Context Recall measures whether all ground truth sentences are covered by retrieved chunks."},
            {"heading": "Evaluation", "level": 1, "text": "RAGAS metrics exhibit high correlation (Pearson r > 0.8) with human judgment on factual hallucination benchmarks."}
        ],
        "references": ["paper_rag_2020", "paper_fid_2020"]
    },
    {
        "doc_id": "paper_crag_2024",
        "title": "Corrective Retrieval Augmented Generation (CRAG)",
        "authors": ["Shi-Qi Yan", "Jia-Chen Gu", "Yun Zhu", "Zhen-Hua Ling"],
        "year": 2024,
        "abstract": "CRAG designs a lightweight retrieval evaluator to assess the quality of retrieved documents, triggering web search when retrieval confidence is low, and decomposing documents into fine-grained knowledge strips.",
        "sections": [
            {"heading": "Method", "level": 1, "text": "A retrieval evaluator classifies retrieval confidence into Correct, Ambiguous, or Incorrect, triggering self-correction actions accordingly."},
            {"heading": "Results", "level": 1, "text": "Significantly improves factual accuracy across PopQA, Biography, and PubHealth benchmarks."}
        ],
        "references": ["paper_rag_2020", "paper_self_rag_2023"]
    }
]

# Evaluation Queries: 50 Questions covering all 5 taxonomy types
EVALUATION_QUERIES = [
    # 1. Definitional Queries (Target: Vector)
    {"id": "q_def_01", "text": "What is LoRA?", "type": "definition"},
    {"id": "q_def_02", "text": "Define the Transformer architecture.", "type": "definition"},
    {"id": "q_def_03", "text": "What is Dense Passage Retrieval (DPR)?", "type": "definition"},
    {"id": "q_def_04", "text": "Explain the concept of Fusion-in-Decoder (FiD).", "type": "definition"},
    {"id": "q_def_05", "text": "What is BERT?", "type": "definition"},
    {"id": "q_def_06", "text": "Define ColBERT late interaction.", "type": "definition"},
    {"id": "q_def_07", "text": "What is GraphRAG?", "type": "definition"},
    {"id": "q_def_08", "text": "What is Prefix-Tuning?", "type": "definition"},
    {"id": "q_def_09", "text": "What is the RAGAS evaluation framework?", "type": "definition"},
    {"id": "q_def_10", "text": "What is Corrective Retrieval Augmented Generation (CRAG)?", "type": "definition"},

    # 2. Relational Queries (Target: Graph)
    {"id": "q_rel_01", "text": "Who introduced LoRA?", "type": "relationship"},
    {"id": "q_rel_02", "text": "Who authored the BERT paper?", "type": "relationship"},
    {"id": "q_rel_03", "text": "Who created Dense Passage Retrieval (DPR)?", "type": "relationship"},
    {"id": "q_rel_04", "text": "Who authored Attention Is All You Need?", "type": "relationship"},
    {"id": "q_rel_05", "text": "Which paper introduced Fusion-in-Decoder?", "type": "relationship"},
    {"id": "q_rel_06", "text": "Who introduced ColBERT?", "type": "relationship"},
    {"id": "q_rel_07", "text": "Which organization introduced GraphRAG?", "type": "relationship"},
    {"id": "q_rel_08", "text": "Who authored the RoBERTa paper?", "type": "relationship"},
    {"id": "q_rel_09", "text": "Who introduced the RAG model combining DPR and BART?", "type": "relationship"},
    {"id": "q_rel_10", "text": "Who proposed Prefix-Tuning?", "type": "relationship"},

    # 3. Comparative Queries (Target: Hybrid)
    {"id": "q_comp_01", "text": "How does LoRA differ from adapters in terms of inference latency?", "type": "comparison"},
    {"id": "q_comp_02", "text": "Compare BERT and RoBERTa pretraining objectives.", "type": "comparison"},
    {"id": "q_comp_03", "text": "What is the difference between DPR and traditional BM25 retrieval?", "type": "comparison"},
    {"id": "q_comp_04", "text": "Compare ColBERT late interaction versus standard cross-encoders.", "type": "comparison"},
    {"id": "q_comp_05", "text": "How does GraphRAG differ from standard vector RAG on global queries?", "type": "comparison"},
    {"id": "q_comp_06", "text": "Compare LoRA and Prefix-Tuning parameter efficiencies.", "type": "comparison"},
    {"id": "q_comp_07", "text": "How does FiD compare with RAG-Sequence in processing multiple passages?", "type": "comparison"},
    {"id": "q_comp_08", "text": "Compare T5 unified text-to-text format with BERT masked language modeling.", "type": "comparison"},
    {"id": "q_comp_09", "text": "What is the difference between REALM pretraining and standard RAG fine-tuning?", "type": "comparison"},
    {"id": "q_comp_10", "text": "Compare QA-GNN and GraphRAG knowledge graph usage.", "type": "comparison"},

    # 4. Multi-hop Relational Queries (Target: Multi-hop Graph -> Vector)
    {"id": "q_hop_01", "text": "Which methods derived from Transformers improved GLUE benchmark scores?", "type": "multi_hop"},
    {"id": "q_hop_02", "text": "Which PEFT methods derived from Houlsby adapters reduced inference overhead?", "type": "multi_hop"},
    {"id": "q_hop_03", "text": "Which retrieval systems based on BERT evaluate on Natural Questions?", "type": "multi_hop"},
    {"id": "q_hop_04", "text": "Who authored the dense retriever used by Patrick Lewis in the RAG paper?", "type": "multi_hop"},
    {"id": "q_hop_05", "text": "Which models introduced by Devlin were subsequently evaluated on SQuAD?", "type": "multi_hop"},
    {"id": "q_hop_06", "text": "Which methods based on T5 achieved state of the art on TriviaQA?", "type": "multi_hop"},
    {"id": "q_hop_07", "text": "Which graph-based reasoning systems use RoBERTa embeddings?", "type": "multi_hop"},
    {"id": "q_hop_08", "text": "Which techniques derived from DPR improve open-domain question answering?", "type": "multi_hop"},
    {"id": "q_hop_09", "text": "Which papers citing Attention Is All You Need introduced low-rank parameter adaptation?", "type": "multi_hop"},
    {"id": "q_hop_10", "text": "Which retrieval evaluation frameworks assess faithfulness in models using DPR?", "type": "multi_hop"},

    # 5. Experimental / Quantitative Evidence Queries (Target: Vector + BM25 + Rerank)
    {"id": "q_exp_01", "text": "What quantitative improvement did LoRA achieve in parameter reduction compared to full fine-tuning?", "type": "experimental"},
    {"id": "q_exp_02", "text": "What was BERT's exact performance score on the GLUE benchmark?", "type": "experimental"},
    {"id": "q_exp_03", "text": "What top-20 passage retrieval accuracy did DPR achieve on Natural Questions?", "type": "experimental"},
    {"id": "q_exp_04", "text": "What BLEU score did the original Transformer model achieve on WMT 2014 English-German?", "type": "experimental"},
    {"id": "q_exp_05", "text": "What exact match score did FiD obtain on Natural Questions?", "type": "experimental"},
    {"id": "q_exp_06", "text": "How many parameters did GPT-3 have according to the original paper?", "type": "experimental"},
    {"id": "q_exp_07", "text": "What GLUE benchmark score did RoBERTa achieve without NSP pretraining?", "type": "experimental"},
    {"id": "q_exp_08", "text": "What latency speedup did ColBERT demonstrate over standard cross-encoders?", "type": "experimental"},
    {"id": "q_exp_09", "text": "What percentage of parameters does Prefix-Tuning train compared to full models?", "type": "experimental"},
    {"id": "q_exp_10", "text": "How much inference latency increase do adapter modules introduce in Transformer layers?", "type": "experimental"}
]

# Ground Truth Relevance (doc_id -> relevance grade: 2=highly relevant, 1=relevant)
GROUND_TRUTH = {
    "q_def_01": {"paper_lora_2021": 2, "paper_adapters_2019": 1},
    "q_def_02": {"paper_attention_2017": 2, "paper_bert_2018": 1},
    "q_def_03": {"paper_dpr_2020": 2, "paper_rag_2020": 1},
    "q_def_04": {"paper_fid_2020": 2, "paper_rag_2020": 1},
    "q_def_05": {"paper_bert_2018": 2, "paper_roberta_2019": 1},
    "q_def_06": {"paper_colbert_2020": 2},
    "q_def_07": {"paper_graphrag_2024": 2},
    "q_def_08": {"paper_prefix_tuning_2021": 2, "paper_lora_2021": 1},
    "q_def_09": {"paper_ragas_2023": 2},
    "q_def_10": {"paper_crag_2024": 2},

    "q_rel_01": {"paper_lora_2021": 2},
    "q_rel_02": {"paper_bert_2018": 2},
    "q_rel_03": {"paper_dpr_2020": 2},
    "q_rel_04": {"paper_attention_2017": 2},
    "q_rel_05": {"paper_fid_2020": 2},
    "q_rel_06": {"paper_colbert_2020": 2},
    "q_rel_07": {"paper_graphrag_2024": 2},
    "q_rel_08": {"paper_roberta_2019": 2},
    "q_rel_09": {"paper_rag_2020": 2, "paper_dpr_2020": 1},
    "q_rel_10": {"paper_prefix_tuning_2021": 2},

    "q_comp_01": {"paper_lora_2021": 2, "paper_adapters_2019": 2},
    "q_comp_02": {"paper_bert_2018": 2, "paper_roberta_2019": 2},
    "q_comp_03": {"paper_dpr_2020": 2},
    "q_comp_04": {"paper_colbert_2020": 2},
    "q_comp_05": {"paper_graphrag_2024": 2, "paper_rag_2020": 1},
    "q_comp_06": {"paper_lora_2021": 2, "paper_prefix_tuning_2021": 2},
    "q_comp_07": {"paper_fid_2020": 2, "paper_rag_2020": 2},
    "q_comp_08": {"paper_t5_2019": 2, "paper_bert_2018": 2},
    "q_comp_09": {"paper_realm_2020": 2, "paper_rag_2020": 2},
    "q_comp_10": {"paper_qagnn_2021": 2, "paper_graphrag_2024": 2},

    "q_hop_01": {"paper_bert_2018": 2, "paper_roberta_2019": 2, "paper_attention_2017": 1},
    "q_hop_02": {"paper_lora_2021": 2, "paper_adapters_2019": 2},
    "q_hop_03": {"paper_dpr_2020": 2, "paper_rag_2020": 2, "paper_bert_2018": 1},
    "q_hop_04": {"paper_rag_2020": 2, "paper_dpr_2020": 2},
    "q_hop_05": {"paper_bert_2018": 2},
    "q_hop_06": {"paper_fid_2020": 2, "paper_t5_2019": 2},
    "q_hop_07": {"paper_qagnn_2021": 2, "paper_roberta_2019": 2},
    "q_hop_08": {"paper_rag_2020": 2, "paper_fid_2020": 2, "paper_dpr_2020": 1},
    "q_hop_09": {"paper_lora_2021": 2, "paper_attention_2017": 1},
    "q_hop_10": {"paper_ragas_2023": 2, "paper_rag_2020": 1},

    "q_exp_01": {"paper_lora_2021": 2},
    "q_exp_02": {"paper_bert_2018": 2},
    "q_exp_03": {"paper_dpr_2020": 2},
    "q_exp_04": {"paper_attention_2017": 2},
    "q_exp_05": {"paper_fid_2020": 2},
    "q_exp_06": {"paper_gpt3_2020": 2},
    "q_exp_07": {"paper_roberta_2019": 2},
    "q_exp_08": {"paper_colbert_2020": 2},
    "q_exp_09": {"paper_prefix_tuning_2021": 2},
    "q_exp_10": {"paper_adapters_2019": 2}
}

# Expected Multi-Hop Reasoning Paths
EXPECTED_PATHS = {
    "q_rel_01": ["ent_author_edward_hu -> INTRODUCES -> ent_lora"],
    "q_rel_02": ["ent_author_devlin -> INTRODUCES -> ent_bert"],
    "q_hop_01": ["ent_transformer -> EXTENDS -> ent_bert -> EVALUATES_ON -> ent_benchmark_glue"],
    "q_hop_02": ["ent_adapters -> IMPROVES -> ent_lora"],
    "q_hop_03": ["ent_bert -> USED_IN -> ent_dpr -> EVALUATES_ON -> ent_benchmark_nq"],
    "q_hop_04": ["ent_author_patrick_lewis -> INTRODUCES -> ent_rag -> USES -> ent_dpr"]
}


def seed_corpus():
    """Generates all processed data directories and files."""
    data_dir = Path("data")
    docs_dir = data_dir / "processed" / "documents"
    chunks_dir = data_dir / "processed" / "chunks"
    entities_dir = data_dir / "processed" / "entities"
    eval_dir = data_dir / "evaluation"

    for d in [docs_dir, chunks_dir, entities_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Save Corpus Documents
    with open(docs_dir / "corpus_documents.json", "w", encoding="utf-8") as f:
        json.dump(PAPERS_DATA, f, indent=2)

    # 2. Save Evaluation Dataset
    with open(eval_dir / "queries.json", "w", encoding="utf-8") as f:
        json.dump(EVALUATION_QUERIES, f, indent=2)

    with open(eval_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(GROUND_TRUTH, f, indent=2)

    with open(eval_dir / "expected_paths.json", "w", encoding="utf-8") as f:
        json.dump(EXPECTED_PATHS, f, indent=2)

    print(f"Successfully generated {len(PAPERS_DATA)} foundational documents and {len(EVALUATION_QUERIES)} benchmark queries.")


if __name__ == "__main__":
    seed_corpus()
