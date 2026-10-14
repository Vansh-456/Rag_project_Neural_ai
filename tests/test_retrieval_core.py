"""BM25 (hand-computed), tokenizer, and fusion tests."""

import math

import pytest

from agvrag.config import BM25Config
from agvrag.retrieval.bm25 import BM25Tokenizer, RankBM25Index, ScratchBM25
from agvrag.retrieval.fusion import rrf_fuse, weighted_fuse
from agvrag.schemas import Chunk, Evidence, EvidenceType


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(f"c{i}", "d", text, None, None, None, "fixed", 3)


CFG = BM25Config(stem=False, remove_stopwords=False, k1=1.5, b=0.75)


def test_bm25_matches_hand_computation():
    texts = ["lora adapter", "lora bert bert", "bert"]
    idx = ScratchBM25([_chunk(i, t) for i, t in enumerate(texts)], texts, CFG)
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected_d0 = idf * 1 * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 2 / 2))
    expected_d1 = idf * 1 * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 3 / 2))
    scores = idx.scores("lora")
    assert scores[0] == pytest.approx(expected_d0)
    assert scores[1] == pytest.approx(expected_d1)
    assert 2 not in scores
    assert [h.chunk.chunk_id for h in idx.search("lora", 5)] == ["c0", "c1"]


def test_bm25_term_frequency_saturates():
    texts = ["bert", "bert bert bert bert bert bert bert bert", "other words here"]
    idx = ScratchBM25([_chunk(i, t) for i, t in enumerate(texts)], texts, CFG)
    s = idx.scores("bert")
    assert s[1] < 8 * s[0]  # k1 saturation: 8x tf gives far less than 8x score


def test_scratch_and_rank_bm25_agree_on_ranking():
    texts = [
        "dense retrieval with dual encoders",
        "sparse lexical retrieval bm25",
        "graph retrieval with knowledge graphs",
        "low rank adaptation lora",
    ]
    chunks = [_chunk(i, t) for i, t in enumerate(texts)]
    cfg = BM25Config(stem=True, remove_stopwords=True)
    a = [
        h.chunk.chunk_id
        for h in ScratchBM25(chunks, texts, cfg).search("lexical bm25 retrieval", 4)
    ]
    b = [
        h.chunk.chunk_id
        for h in RankBM25Index(chunks, texts, cfg).search("lexical bm25 retrieval", 4)
    ]
    assert a[0] == b[0] == "c1"


def test_tokenizer_compounds_and_stems():
    t = BM25Tokenizer(stem=True, remove_stopwords=True)
    toks = t.tokenize("GPT-3 models are fine-tuned on the tasks")
    assert "gpt3" in toks and "gpt" in toks and "3" in toks
    assert "finetun" in toks or "finetuned" in toks
    assert "the" not in toks
    assert "not" in t.tokenize("does not improve")  # negation is kept


def _ev(cid: str, score: float, rank: int, retriever: str) -> Evidence:
    return Evidence(
        f"chunk:{cid}", EvidenceType.TEXT_CHUNK, cid, ["d"], [cid], retriever, score, rank
    )


def test_rrf_hand_example():
    lists = {
        "dense": [_ev("a", 0.9, 1, "dense"), _ev("b", 0.8, 2, "dense")],
        "bm25": [_ev("b", 12.0, 1, "bm25"), _ev("c", 5.0, 2, "bm25")],
    }
    fused = rrf_fuse(lists, k=60)
    assert [e.source_chunk_ids[0] for e in fused] == ["b", "a", "c"]
    assert fused[0].raw_score == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)
    assert fused[0].retriever == "bm25+dense"
    assert set(fused[0].provenance["fusion"]["sources"]) == {"dense", "bm25"}
    assert [e.rank for e in fused] == [1, 2, 3]


def test_weighted_fusion_minmax():
    lists = {
        "dense": [_ev("a", 0.9, 1, "dense"), _ev("b", 0.5, 2, "dense")],
        "bm25": [_ev("b", 10.0, 1, "bm25"), _ev("a", 0.0, 2, "bm25")],
    }
    fused = weighted_fuse(lists, {"dense": 0.3, "bm25": 0.7})
    assert fused[0].source_chunk_ids[0] == "b"  # 0.7*1 + 0.3*0 beats 0.3*1 + 0.7*0
    fused2 = weighted_fuse(lists, {"dense": 0.7, "bm25": 0.3})
    assert fused2[0].source_chunk_ids[0] == "a"
