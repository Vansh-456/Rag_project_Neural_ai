"""
Unit tests for atomic claim verification, evidence support, and contradiction detection.
"""

from src.schemas import Evidence, EvidenceType, VerificationStatus
from src.verification.verifier import ClaimVerifier


def test_claim_verifier_supported():
    verifier = ClaimVerifier()
    evidence = [
        Evidence(
            evidence_id="ev_1",
            type=EvidenceType.TEXT_CHUNK,
            content="LoRA reduces trainable parameters by 10,000x and GPU memory requirement by 3x.",
            source_doc_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
            retriever="dense",
            raw_score=0.9,
            rank=1
        )
    ]

    answer = "LoRA reduces trainable parameters by 10,000x and GPU memory requirement by 3x [E1]."
    res = verifier.verify(answer, evidence, query="What is LoRA?")

    assert res.is_sufficient is True
    assert res.supported_ratio == 1.0
    assert len(res.claims) == 1
    assert res.claims[0].status == VerificationStatus.SUPPORTED


def test_claim_verifier_unsupported():
    verifier = ClaimVerifier()
    evidence = [
        Evidence(
            evidence_id="ev_1",
            type=EvidenceType.TEXT_CHUNK,
            content="BERT uses a masked language model pretraining objective.",
            source_doc_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
            retriever="dense",
            raw_score=0.9,
            rank=1
        )
    ]

    # Claim makes a statement without citations
    answer = "BERT was developed by researchers in Germany."
    res = verifier.verify(answer, evidence, query="Who made BERT?")

    assert res.is_sufficient is False
    assert res.supported_ratio == 0.0
    assert res.claims[0].status == VerificationStatus.UNSUPPORTED
    assert res.retry_recommended is True


def test_claim_verifier_abstention():
    verifier = ClaimVerifier()
    answer = "I cannot answer this question based on the provided evidence."
    res = verifier.verify(answer, [], query="What is quantum gravity?")
    assert res.is_sufficient is True
    assert res.retry_recommended is False
