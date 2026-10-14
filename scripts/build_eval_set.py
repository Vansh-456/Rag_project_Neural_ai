"""Source of truth for the evaluation set (v1, draft). Writes data/evaluation/*.json.

Labels follow docs/labeling_guideline.md: each label = (doc_id, anchors, grade) where an anchor
is a short verbatim phrase identifying the relevant passage. Grade 2 = directly answers,
grade 1 = partially relevant / supporting. Splits alternate within each query type so dev and
test have the same type mix. Run ``agvrag coverage`` after any edit: every label must map to
at least one chunk.

Status: authored by the system builder from the corpus text, NOT yet independently
re-labelled. Treat metrics as provisional until a second annotation pass (guideline §6).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (id, type, text, labels[(doc, [anchors], grade)], extra)
Q = [
    # ---------------------------------------------------------------- definition
    (
        "d01",
        "definition",
        "What is LoRA?",
        [("2106.09685", ["rank decomposition matrices"], 2)],
        {},
    ),
    ("d02", "definition", "What is HyDE?", [("2212.10496", ["hypothetical document"], 2)], {}),
    (
        "d03",
        "definition",
        "What is late interaction in ColBERT?",
        [("2004.12832", ["late interaction"], 2)],
        {},
    ),
    (
        "d04",
        "definition",
        "What is the Fusion-in-Decoder model?",
        [("2007.01282", ["fusion-in-decoder"], 2)],
        {"tags": ["acronym"]},
    ),
    (
        "d05",
        "definition",
        "What is prefix-tuning?",
        [("2101.00190", ["continuous task-specific vector"], 2)],
        {},
    ),
    (
        "d06",
        "definition",
        "What is a kNN-LM?",
        [("1911.00172", ["nearest neighbor"], 2)],
        {"tags": ["acronym"]},
    ),
    (
        "d07",
        "definition",
        "What is Dense Passage Retrieval?",
        [("2004.04906", ["dual-encoder"], 2)],
        {},
    ),
    (
        "d08",
        "definition",
        "What is the Retrieval-Enhanced Transformer?",
        [("2112.04426", ["retrieval-enhanced transformer"], 2)],
        {},
    ),
    (
        "d09",
        "definition",
        "What is the 4-bit NormalFloat data type in QLoRA?",
        [("2305.14314", ["4-bit normalfloat"], 2)],
        {},
    ),
    (
        "d10",
        "definition",
        "What is RAPTOR?",
        [("2401.18059", ["recursively embedding, clustering, and summarizing"], 2)],
        {},
    ),
    ("d11", "definition", "What is BitFit?", [("2106.10199", ["bias terms"], 2)], {}),
    (
        "d12",
        "definition",
        "What are reflection tokens in Self-RAG?",
        [("2310.11511", ["reflection tokens"], 2)],
        {},
    ),
    (
        "d13",
        "definition",
        "What is a Differentiable Search Index?",
        [("2202.06991", ["differentiable search index"], 2)],
        {},
    ),
    (
        "d14",
        "definition",
        "Which method does not require relevance labels for zero-shot dense retrieval?",
        [("2212.10496", ["relevance labels"], 2)],
        {"tags": ["negation", "paraphrase"]},
    ),
    (
        "d15",
        "definition",
        "what is low rank adaption",
        [("2106.09685", ["rank decomposition matrices"], 2)],
        {"tags": ["acronym", "paraphrase", "typo"]},
    ),
    (
        "d16",
        "definition",
        "What is the inverse cloze task?",
        [("1906.00300", ["inverse cloze task"], 2)],
        {},
    ),
    # ---------------------------------------------------------------- relationship
    (
        "r01",
        "relationship",
        "Who introduced LoRA?",
        [("2106.09685", ["we propose low-rank adaptation"], 2)],
        {"paths": {"entities": ["method:lora", "paper:2106.09685"], "relations": ["INTRODUCES"]}},
    ),
    (
        "r02",
        "relationship",
        "Which paper introduced ColBERT?",
        [("2004.12832", ["we present colbert"], 2)],
        {
            "paths": {
                "entities": ["method:colbert", "paper:2004.12832"],
                "relations": ["INTRODUCES"],
            }
        },
    ),
    (
        "r03",
        "relationship",
        "Who proposed Dense Passage Retrieval?",
        [("2004.04906", ["dual-encoder"], 2)],
        {"paths": {"entities": ["method:dpr", "paper:2004.04906"], "relations": ["INTRODUCES"]}},
    ),
    (
        "r04",
        "relationship",
        "Which paper introduced the Fusion-in-Decoder approach?",
        [("2007.01282", ["fusion-in-decoder"], 2)],
        {"paths": {"entities": ["method:fid", "paper:2007.01282"], "relations": ["INTRODUCES"]}},
    ),
    (
        "r05",
        "relationship",
        "Who are the authors of REALM?",
        [("2002.08909", ["kelvin guu"], 2)],
        {
            "paths": {
                "entities": ["method:realm", "paper:2002.08909", "author:kelvin guu"],
                "relations": ["INTRODUCES", "AUTHORED_BY"],
            }
        },
    ),
    (
        "r06",
        "relationship",
        "Which paper introduced the HotpotQA dataset?",
        [("1809.09600", ["hotpotqa"], 2)],
        {
            "paths": {
                "entities": ["dataset:hotpotqa", "paper:1809.09600"],
                "relations": ["INTRODUCES"],
            }
        },
    ),
    (
        "r07",
        "relationship",
        "Who introduced the BEIR benchmark?",
        [("2104.08663", ["beir"], 2)],
        {"paths": {"entities": ["dataset:beir", "paper:2104.08663"], "relations": ["INTRODUCES"]}},
    ),
    (
        "r08",
        "relationship",
        "Who introduced prompt tuning?",
        [("2104.08691", ["prompt tuning"], 2)],
        {
            "paths": {
                "entities": ["method:prompt tuning", "paper:2104.08691"],
                "relations": ["INTRODUCES"],
            }
        },
    ),
    (
        "r09",
        "relationship",
        "Which retriever is Atlas based on?",
        [("2208.03299", ["contriever"], 2)],
        {
            "paths": {
                "entities": ["model:atlas", "method:contriever"],
                "relations": ["USES", "BASED_ON"],
            }
        },
    ),
    (
        "r10",
        "relationship",
        "Which retriever does RAG use?",
        [("2005.11401", ["dense passage retriever"], 2)],
        {"paths": {"entities": ["method:rag", "method:dpr"], "relations": ["USES", "BASED_ON"]}},
    ),
    (
        "r11",
        "relationship",
        "Which paper introduced MTEB?",
        [("2210.07316", ["mteb"], 2)],
        {"paths": {"entities": ["dataset:mteb", "paper:2210.07316"], "relations": ["INTRODUCES"]}},
    ),
    (
        "r12",
        "relationship",
        "Who proposed Chain-of-Verification?",
        [("2309.11495", ["chain-of-verification"], 2)],
        {
            "paths": {
                "entities": ["method:chain of verification", "paper:2309.11495"],
                "relations": ["INTRODUCES"],
            }
        },
    ),
    (
        "r13",
        "relationship",
        "Which paper introduced Sentence-BERT siamese networks?",
        [("1908.10084", ["siamese"], 2)],
        {
            "paths": {
                "entities": ["method:sentence bert", "paper:1908.10084"],
                "relations": ["INTRODUCES"],
            }
        },
    ),
    (
        "r14",
        "relationship",
        "Who introduced doc2query document expansion?",
        [("1904.08375", ["doc2query"], 2)],
        {
            "paths": {
                "entities": ["method:doc2query", "paper:1904.08375"],
                "relations": ["INTRODUCES"],
            }
        },
    ),
    # ---------------------------------------------------------------- comparison
    (
        "c01",
        "comparison",
        "How does LoRA differ from adapters?",
        [("2106.09685", ["inference latency"], 2), ("1902.00751", ["adapter modules"], 1)],
        {},
    ),
    (
        "c02",
        "comparison",
        "What is the difference between RAG-Sequence and RAG-Token?",
        [("2005.11401", ["rag-token"], 2)],
        {},
    ),
    (
        "c03",
        "comparison",
        "How does prefix-tuning compare to fine-tuning in trainable parameters?",
        [("2101.00190", ["0.1%"], 2)],
        {},
    ),
    (
        "c04",
        "comparison",
        "How do REALM and RAG differ?",
        [("2005.11401", ["realm"], 2), ("2002.08909", ["knowledge retriever"], 1)],
        {},
    ),
    (
        "c05",
        "comparison",
        "How does BM25 compare to dense retrievers on BEIR in zero-shot settings?",
        [("2104.08663", ["bm25"], 2)],
        {},
    ),
    (
        "c06",
        "comparison",
        "How does prompt tuning compare to GPT-3 prompt design?",
        [("2104.08691", ["prompt design"], 2)],
        {},
    ),
    (
        "c07",
        "comparison",
        "How does ColBERTv2 differ from ColBERT?",
        [("2112.01488", ["residual compression"], 2)],
        {},
    ),
    (
        "c08",
        "comparison",
        "How does Contriever compare to BM25 without supervision?",
        [("2112.09118", ["bm25"], 2)],
        {},
    ),
    (
        "c09",
        "comparison",
        "How does SimCSE versus Sentence-BERT train sentence embeddings?",
        [("2104.08821", ["dropout"], 2), ("1908.10084", ["siamese"], 1)],
        {},
    ),
    (
        "c10",
        "comparison",
        "Compare GraphRAG with naive RAG for global questions",
        [("2404.16130", ["community summaries"], 2)],
        {},
    ),
    (
        "c11",
        "comparison",
        "How does monoT5 compare to monoBERT for passage re-ranking?",
        [("1901.04085", ["re-ranker"], 1), ("2003.06713", ["t5"], 2)],
        {},
    ),
    (
        "c12",
        "comparison",
        "How does LightRAG differ from GraphRAG?",
        [("2410.05779", ["dual-level retrieval"], 2)],
        {},
    ),
    # ---------------------------------------------------------------- multi_hop
    (
        "m01",
        "multi_hop",
        "Which methods based on DPR were evaluated on Natural Questions?",
        [("2005.11401", ["natural questions"], 2), ("2007.01282", ["dpr"], 1)],
        {
            "paths": {
                "entities": ["method:dpr", "dataset:natural questions"],
                "relations": ["BASED_ON", "EVALUATES_ON"],
            }
        },
    ),
    (
        "m02",
        "multi_hop",
        "Which papers that use BERT introduce a ranking method evaluated on MS MARCO?",
        [("2004.12832", ["ms marco"], 2), ("1901.04085", ["ms marco"], 2)],
        {
            "paths": {
                "entities": ["model:bert", "dataset:ms marco"],
                "relations": ["USES", "EVALUATES_ON"],
            }
        },
    ),
    (
        "m03",
        "multi_hop",
        "Which datasets were used to evaluate the method that introduced late interaction?",
        [("2004.12832", ["ms marco"], 2)],
        {
            "paths": {
                "entities": ["method:colbert", "dataset:ms marco"],
                "relations": ["INTRODUCES", "EVALUATES_ON"],
            }
        },
    ),
    (
        "m04",
        "multi_hop",
        "Which parameter-efficient methods that freeze the pretrained model were evaluated on GLUE?",
        [
            ("2106.09685", ["glue"], 2),
            ("1902.00751", ["glue"], 2),
            ("2106.10199", ["glue"], 2),
            ("2110.07602", ["glue"], 1),
        ],
        {"paths": {"entities": ["dataset:glue"], "relations": ["EVALUATES_ON"]}},
    ),
    (
        "m05",
        "multi_hop",
        "Which methods derived from LoRA reduce finetuning memory further?",
        [("2305.14314", ["4-bit normalfloat"], 2), ("2402.09353", ["lora"], 1)],
        {
            "paths": {
                "entities": ["method:lora", "method:qlora"],
                "relations": ["BASED_ON", "EXTENDS", "USES"],
            }
        },
    ),
    (
        "m06",
        "multi_hop",
        "Which retrieval-augmented models that use a DPR retriever pair it with a BART generator?",
        [("2005.11401", ["bart"], 2)],
        {
            "paths": {
                "entities": ["method:rag", "model:bart", "method:dpr"],
                "relations": ["USES", "BASED_ON"],
            }
        },
    ),
    (
        "m07",
        "multi_hop",
        "Which methods that interleave reasoning and retrieval were evaluated on HotpotQA?",
        [("2212.10509", ["hotpotqa"], 2), ("2210.03629", ["hotpotqa"], 2)],
        {"paths": {"entities": ["dataset:hotpotqa"], "relations": ["EVALUATES_ON"]}},
    ),
    (
        "m08",
        "multi_hop",
        "Which embedding models introduced after Sentence-BERT were evaluated on MTEB?",
        [("2212.03533", ["mteb"], 2), ("2309.07597", ["mteb"], 2), ("2210.07316", ["mteb"], 1)],
        {"paths": {"entities": ["dataset:mteb"], "relations": ["EVALUATES_ON"]}},
    ),
    (
        "m09",
        "multi_hop",
        "Which adaptive methods that retrieve only when needed use low-probability tokens as a trigger?",
        [("2305.06983", ["low-probability"], 2), ("2310.11511", ["reflection tokens"], 1)],
        {},
    ),
    (
        "m10",
        "multi_hop",
        "Which methods that extend ColBERT were evaluated on BEIR?",
        [("2112.01488", ["beir"], 2)],
        {
            "paths": {
                "entities": ["method:colbert", "method:colbertv2", "dataset:beir"],
                "relations": ["EXTENDS", "BASED_ON", "EVALUATES_ON"],
            }
        },
    ),
    (
        "m11",
        "multi_hop",
        "Which papers that cite the Transformer paper introduce masked language modeling?",
        [("1810.04805", ["masked language model"], 2)],
        {
            "paths": {
                "entities": ["model:bert", "method:masked language modeling"],
                "relations": ["INTRODUCES", "CITES"],
            }
        },
    ),
    # ---------------------------------------------------------------- experimental
    (
        "e01",
        "experimental",
        "What results show DPR outperforms BM25 on top-20 passage retrieval accuracy?",
        [("2004.04906", ["9%-19%"], 2)],
        {},
    ),
    (
        "e02",
        "experimental",
        "What evidence shows LoRA performs on-par with full fine-tuning?",
        [("2106.09685", ["on-par"], 2)],
        {},
    ),
    (
        "e03",
        "experimental",
        "How much GPU memory does QLoRA need to finetune a 65B model?",
        [("2305.14314", ["48gb"], 2)],
        {},
    ),
    (
        "e04",
        "experimental",
        "How much does chain-of-thought prompting improve accuracy on GSM8K?",
        [("2201.11903", ["gsm8k"], 2)],
        {},
    ),
    (
        "e05",
        "experimental",
        "How does performance change with the position of relevant information in a long context?",
        [("2307.03172", ["u-shaped"], 2)],
        {},
    ),
    (
        "e06",
        "experimental",
        "What evidence shows HyDE improves zero-shot retrieval over Contriever?",
        [("2212.10496", ["contriever"], 2)],
        {},
    ),
    (
        "e07",
        "experimental",
        "How does the number of retrieved passages affect Fusion-in-Decoder accuracy?",
        [("2007.01282", ["number of passages"], 2)],
        {},
    ),
    (
        "e08",
        "experimental",
        "By how much does LoRA reduce the number of trainable parameters compared with GPT-3 fine-tuning?",
        [("2106.09685", ["10,000 times"], 2)],
        {},
    ),
    (
        "e09",
        "experimental",
        "What results show RETRO matches larger models with fewer parameters?",
        [("2112.04426", ["fewer parameters"], 2)],
        {},
    ),
    (
        "e10",
        "experimental",
        "What experiments show Self-RAG outperforms ChatGPT?",
        [("2310.11511", ["chatgpt"], 2)],
        {},
    ),
    (
        "e11",
        "experimental",
        "How does the datastore size affect kNN-LM perplexity?",
        [("1911.00172", ["datastore"], 2)],
        {},
    ),
    (
        "e12",
        "experimental",
        "What power-law relationship links language model loss to model size?",
        [("2001.08361", ["power-law"], 2)],
        {},
    ),
    (
        "e13",
        "experimental",
        "Which parameter-efficient method adds no additional inference latency according to its experiments?",
        [("2106.09685", ["no additional inference latency"], 2)],
        {"tags": ["paraphrase"]},
    ),
    (
        "e14",
        "experimental",
        "How robust are LLMs to noisy retrieved documents in RAG benchmarks?",
        [("2309.01431", ["noise robustness"], 2)],
        {},
    ),
    # ---------------------------------------------------------------- unanswerable (abstention)
    (
        "u01",
        "definition",
        "What is the boiling point of tungsten?",
        [],
        {"answerable": False, "tags": ["unanswerable"]},
    ),
    (
        "u02",
        "relationship",
        "Who won the 2018 FIFA World Cup?",
        [],
        {"answerable": False, "tags": ["unanswerable"]},
    ),
    (
        "u03",
        "comparison",
        "How does LoRA compare to Mixture-of-Depths routing in Gemini?",
        [],
        {"answerable": False, "tags": ["unanswerable", "adversarial"]},
    ),
    (
        "u04",
        "experimental",
        "What exact match score does GPT-7 achieve on SQuAD 5.0?",
        [],
        {"answerable": False, "tags": ["unanswerable", "adversarial"]},
    ),
]


def main() -> None:
    queries, gt, paths = [], {}, {}
    counter: dict[str, int] = defaultdict(int)
    for qid, qtype, text, labels, extra in Q:
        answerable = extra.get("answerable", True)
        key = qtype if answerable else f"u-{qtype}"
        counter[key] += 1
        split = "dev" if counter[key] % 2 == 1 else "test"
        queries.append(
            {
                "id": qid,
                "text": text,
                "type": qtype,
                "split": split,
                "answerable": answerable,
                "tags": extra.get("tags", []),
            }
        )
        gt[qid] = [{"doc_id": d, "anchors": a, "grade": g} for d, a, g in labels]
        if "paths" in extra:
            paths[qid] = extra["paths"]
    out = ROOT / "data" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (
        ("queries.json", queries),
        ("ground_truth.json", gt),
        ("expected_paths.json", paths),
    ):
        (out / name).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    by = defaultdict(lambda: defaultdict(int))
    for q in queries:
        by[q["type"]][q["split"]] += 1
    print(json.dumps({"n": len(queries), "by_type_split": by}, indent=1, default=dict))


if __name__ == "__main__":
    main()
