"""Shared fixtures: a tiny synthetic corpus and a fully built offline index + graph.

Everything here is deterministic and offline: hashing embedder, in-memory vector store,
NetworkX graph, rule extractor, lexical reranker, extractive generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agvrag.config import Settings, load_settings
from agvrag.pipeline.components import ComponentRegistry
from agvrag.pipeline.indexing import build_graph_index, build_index
from agvrag.schemas import Document, Section
from agvrag.utils.io import write_json, write_yaml

GAZETTEER = {
    "entities": [
        {
            "name": "LoRA",
            "type": "Method",
            "aliases": ["Low-Rank Adaptation"],
            "introduced_in": "2106.09685",
            "case_sensitive": True,
        },
        {
            "name": "Adapters",
            "type": "Method",
            "aliases": ["adapter", "adapter modules", "adapter layers"],
            "introduced_in": "1902.00751",
        },
        {
            "name": "DPR",
            "type": "Method",
            "aliases": ["Dense Passage Retrieval"],
            "introduced_in": "2004.04906",
            "case_sensitive": True,
        },
        {
            "name": "RAG",
            "type": "Method",
            "aliases": ["Retrieval-Augmented Generation"],
            "introduced_in": "2005.11401",
            "case_sensitive": True,
        },
        {"name": "Self-RAG", "type": "Method"},
        {"name": "BM25", "type": "Method", "case_sensitive": True},
        {"name": "BERT", "type": "Model", "case_sensitive": True},
        {"name": "BART", "type": "Model", "case_sensitive": True},
        {"name": "GPT-3", "type": "Model", "aliases": ["GPT3"], "case_sensitive": True},
        {"name": "GLUE", "type": "Dataset", "case_sensitive": True},
        {"name": "Natural Questions", "type": "Dataset", "aliases": ["NQ"], "case_sensitive": True},
        {"name": "E2E NLG", "type": "Dataset", "case_sensitive": True},
        {
            "name": "Open-Domain Question Answering",
            "type": "Task",
            "aliases": ["open-domain question answering", "open-domain QA"],
        },
        {"name": "Exact Match", "type": "Metric", "aliases": ["EM"], "case_sensitive": True},
    ]
}


def _doc(
    doc_id: str,
    title: str,
    authors: list[str],
    year: int,
    sections: list[tuple[str, str]],
    refs: list[str],
) -> Document:
    return Document(
        doc_id=doc_id,
        title=title,
        authors=authors,
        year=year,
        sections=[Section(h, 1, t, i + 1, i + 1) for i, (h, t) in enumerate(sections)],
        references=refs,
        source_path=f"{doc_id}.pdf",
        abstract=sections[0][1],
        metadata={"topic": "test"},
    )


def make_corpus() -> list[Document]:
    lora = _doc(
        "2106.09685",
        "LoRA: Low-Rank Adaptation of Large Language Models",
        ["Edward Hu", "Yelong Shen"],
        2021,
        [
            (
                "Abstract",
                "We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and "
                "injects trainable rank decomposition matrices into each layer of the Transformer architecture. "
                "LoRA greatly reduces the number of trainable parameters for downstream tasks.",
            ),
            (
                "1 Introduction",
                "Fine-tuning GPT-3 175B is prohibitively expensive. Adapter layers introduce inference "
                "latency because they are processed sequentially. LoRA outperforms adapter layers while "
                "adding no inference latency. LoRA is based on the observation that weight updates have a "
                "low intrinsic rank.",
            ),
            (
                "5 Experiments",
                "We evaluate LoRA on GLUE and on E2E NLG. On GLUE, LoRA matches or exceeds full fine-tuning "
                "with far fewer trainable parameters. Results on GLUE show that LoRA outperforms Adapters "
                "in accuracy. LoRA uses GPT-3 as a backbone in additional experiments.",
            ),
        ],
        [
            "N. Houlsby, A. Giurgiu, et al. Parameter-Efficient Transfer Learning for NLP. In ICML, 2019."
        ],
    )
    adapters = _doc(
        "1902.00751",
        "Parameter-Efficient Transfer Learning for NLP",
        ["Neil Houlsby", "Andrei Giurgiu"],
        2019,
        [
            (
                "Abstract",
                "We propose transfer with adapter modules. Adapter modules yield a compact and extensible model; "
                "they add only a few trainable parameters per task. Adapters are based on BERT.",
            ),
            (
                "3 Experiments",
                "We evaluate adapters on the GLUE benchmark. Adapters attain within 0.4% of the performance "
                "of full fine-tuning on GLUE while adding only 3.6% parameters per task. We use BERT as the "
                "base model for all GLUE experiments.",
            ),
        ],
        ["J. Devlin et al. BERT: Pre-training of deep bidirectional transformers. 2019."],
    )
    dpr = _doc(
        "2004.04906",
        "Dense Passage Retrieval for Open-Domain Question Answering",
        ["Vladimir Karpukhin", "Barlas Oguz"],
        2020,
        [
            (
                "Abstract",
                "Open-domain question answering relies on efficient passage retrieval. We show that retrieval can "
                "be practically implemented using dense representations alone. Our dense passage retriever, "
                "DPR, uses a dual-encoder framework built on BERT.",
            ),
            (
                "4 Experiments",
                "We evaluate DPR on Natural Questions and TriviaQA. DPR outperforms a strong Lucene BM25 system "
                "by 9%-19% absolute in terms of top-20 passage retrieval accuracy on Natural Questions. "
                "Exact Match scores improve as well.",
            ),
        ],
        [],
    )
    rag = _doc(
        "2005.11401",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        ["Patrick Lewis", "Ethan Perez"],
        2020,
        [
            (
                "Abstract",
                "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models "
                "which combine pre-trained parametric and non-parametric memory for language generation. RAG "
                "uses a DPR retriever and a BART generator.",
            ),
            (
                "2 Methods",
                "The retriever of RAG is based on DPR. The generator is BART-large. RAG outperforms extractive "
                "approaches on open-domain QA.",
            ),
            (
                "3 Experiments",
                "We evaluate RAG on Natural Questions where RAG achieves state-of-the-art Exact Match. "
                "RAG outperforms DPR on Natural Questions.",
            ),
        ],
        [
            "V. Karpukhin et al. Dense Passage Retrieval for Open-Domain Question Answering. In EMNLP, 2020."
        ],
    )
    return [lora, adapters, dpr, rag]


QUERIES = [
    {"id": "t1", "text": "What is LoRA?", "type": "definition", "split": "dev"},
    {"id": "t2", "text": "Who introduced LoRA?", "type": "relationship", "split": "dev"},
    {
        "id": "t3",
        "text": "How does LoRA differ from adapters?",
        "type": "comparison",
        "split": "dev",
    },
    {
        "id": "t4",
        "text": "Which methods based on DPR were evaluated on Natural Questions?",
        "type": "multi_hop",
        "split": "test",
    },
    {
        "id": "t5",
        "text": "What evidence shows DPR outperforms BM25?",
        "type": "experimental",
        "split": "test",
    },
    {
        "id": "t6",
        "text": "What is the capital of France?",
        "type": "definition",
        "split": "test",
        "answerable": False,
    },
]
GROUND_TRUTH = {
    "t1": [
        {"doc_id": "2106.09685", "anchors": ["injects trainable rank decomposition"], "grade": 2}
    ],
    "t2": [{"doc_id": "2106.09685", "anchors": ["We propose Low-Rank Adaptation"], "grade": 2}],
    "t3": [
        {
            "doc_id": "2106.09685",
            "anchors": ["Adapter layers introduce inference latency"],
            "grade": 2,
        },
        {"doc_id": "1902.00751", "anchors": ["adapter modules"], "grade": 1},
    ],
    "t4": [{"doc_id": "2005.11401", "anchors": ["based on DPR"], "grade": 2}],
    "t5": [{"doc_id": "2004.04906", "anchors": ["outperforms a strong Lucene BM25"], "grade": 2}],
    "t6": [],
}
EXPECTED_PATHS = {
    "t2": {"entities": ["method:lora", "paper:2106.09685"], "relations": ["INTRODUCES"]}
}


def make_settings(tmp: Path, **pipeline: Any) -> Settings:
    write_yaml(tmp / "gazetteer.yaml", GAZETTEER)
    overrides: dict[str, Any] = {
        "paths": {
            "gazetteer_file": str(tmp / "gazetteer.yaml"),
            "raw_dir": str(tmp / "raw"),
            "documents_dir": str(tmp / "documents"),
            "index_dir": str(tmp / "index"),
            "cache_dir": str(tmp / "cache"),
            "results_dir": str(tmp / "results"),
            "traces_db": str(tmp / "traces.sqlite"),
            "graph_dir": str(tmp / "graph"),
        },
        "chunking": {"strategy": "structure_aware", "chunk_size": 64, "overlap": 8},
        "embedding": {"provider": "hashing", "hashing_dim": 256},
        "vector_store": {"backend": "memory"},
        "reranker": {"enabled": True, "provider": "lexical", "candidate_pool": 20},
        "retrieval": {"top_k": 5, "candidate_pool": 20},
        "evaluation": {
            "queries_file": str(tmp / "queries.json"),
            "ground_truth_file": str(tmp / "gt.json"),
            "expected_paths_file": str(tmp / "paths.json"),
            "bootstrap_samples": 200,
            "save_traces": True,
        },
        "generation": {"provider": "extractive"},
        "logging": {"level": "WARNING", "json_logs": False},
        "api": {"rate_limit_per_minute": 1000},
    }
    if pipeline:
        overrides["pipeline"] = pipeline
    return load_settings("base", overrides=overrides, environ={})


@pytest.fixture(scope="session")
def corpus() -> list[Document]:
    return make_corpus()


@pytest.fixture(scope="session")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Settings, ComponentRegistry]:
    tmp = tmp_path_factory.mktemp("agv")
    s = make_settings(tmp)
    for d in make_corpus():
        write_json(s.path("documents_dir") / f"{d.doc_id}.json", d.to_dict())
    write_json(tmp / "queries.json", QUERIES)
    write_json(tmp / "gt.json", GROUND_TRUTH)
    write_json(tmp / "paths.json", EXPECTED_PATHS)
    reg = ComponentRegistry()
    build_index(s, reg)
    build_graph_index(s, reg)
    return s, reg


def settings_for(built: tuple[Settings, ComponentRegistry], **pipeline: Any) -> Settings:
    s, _ = built
    tmp = Path(s.paths.index_dir).parent
    out = make_settings(tmp, **pipeline)
    if pipeline.get("router") == "adaptive":
        out = out.model_copy(update={"router": out.router.model_copy(update={"mode": "rule"})})
    return out
