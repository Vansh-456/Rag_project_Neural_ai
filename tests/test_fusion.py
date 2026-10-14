"""
Unit tests for Reciprocal Rank Fusion (RRF) and Weighted Score Fusion.
"""

from src.schemas import Evidence, EvidenceType
from src.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion


def make_test_evidence(ev_id: str, retriever: str, score: float, rank: int) -> Evidence:
    return Evidence(
        evidence_id=ev_id,
        type=EvidenceType.TEXT_CHUNK,
        content=f"Content for {ev_id}",
        source_doc_ids=["doc_1"],
        source_chunk_ids=[ev_id],
        retriever=retriever,
        raw_score=score,
        rank=rank
    )


def test_reciprocal_rank_fusion():
    # Dense list has item A (rank 1), item B (rank 2)
    dense_list = [
        make_test_evidence("chunk_A", "dense", 0.9, 1),
        make_test_evidence("chunk_B", "dense", 0.8, 2),
    ]
    # Sparse list has item B (rank 1), item C (rank 2)
    sparse_list = [
        make_test_evidence("chunk_B", "bm25", 15.0, 1),
        make_test_evidence("chunk_C", "bm25", 12.0, 2),
    ]

    # chunk_B appears in both lists, so its RRF score is:
    # 1/(60 + 2) + 1/(60 + 1) = 1/62 + 1/61 = 0.0161 + 0.0163 = 0.0324
    # chunk_A only in dense: 1/61 = 0.0163
    # chunk_C only in sparse: 1/62 = 0.0161
    fused = reciprocal_rank_fusion([dense_list, sparse_list], k=60, top_k=3)

    assert len(fused) == 3
    # Item B should be rank 1 due to multi-source consensus
    assert fused[0].source_chunk_ids == ["chunk_B"]
    assert fused[1].source_chunk_ids == ["chunk_A"]
    assert fused[2].source_chunk_ids == ["chunk_C"]


def test_weighted_score_fusion():
    list_a = [
        make_test_evidence("chunk_1", "dense", 0.9, 1),
        make_test_evidence("chunk_2", "dense", 0.3, 2)
    ]
    list_b = [
        make_test_evidence("chunk_1", "bm25", 20.0, 1),
        make_test_evidence("chunk_3", "bm25", 10.0, 2)
    ]

    fused = weighted_score_fusion([(list_a, 0.5), (list_b, 0.5)], top_k=3)
    assert len(fused) == 3
    assert fused[0].source_chunk_ids == ["chunk_1"]
