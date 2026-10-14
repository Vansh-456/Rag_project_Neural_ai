"""
Evidence fusion module implementing Reciprocal Rank Fusion (RRF) and Weighted Score Fusion.
"""

from __future__ import annotations

from typing import Any
from src.schemas import Evidence


def reciprocal_rank_fusion(
    ranked_lists: list[list[Evidence]],
    k: int = 60,
    top_k: int = 5
) -> list[Evidence]:
    """
    Combines multiple ranked evidence lists using Reciprocal Rank Fusion (RRF).
    score(d) = sum_{r in retrievers} (1 / (k + rank_r(d)))
    """
    rrf_scores: dict[str, float] = {}
    evidence_by_content_id: dict[str, Evidence] = {}
    sources_per_item: dict[str, list[dict[str, Any]]] = {}

    for ranked_list in ranked_lists:
        for rank, ev in enumerate(ranked_list, start=1):
            # Key evidence by first chunk ID or content hash for deduplication
            key = ev.source_chunk_ids[0] if ev.source_chunk_ids else ev.content[:60]
            rrf_delta = 1.0 / (k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_delta

            if key not in evidence_by_content_id:
                evidence_by_content_id[key] = ev

            if key not in sources_per_item:
                sources_per_item[key] = []
            sources_per_item[key].append({
                "retriever": ev.retriever,
                "rank": rank,
                "raw_score": ev.raw_score
            })

    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    fused_evidence: list[Evidence] = []
    for rank, key in enumerate(sorted_keys, start=1):
        base_ev = evidence_by_content_id[key]
        fused_evidence.append(Evidence(
            evidence_id=f"ev_fused_{base_ev.evidence_id}",
            type=base_ev.type,
            content=base_ev.content,
            source_doc_ids=base_ev.source_doc_ids,
            source_chunk_ids=base_ev.source_chunk_ids,
            retriever="hybrid_rrf",
            raw_score=round(rrf_scores[key], 5),
            rank=rank,
            provenance={
                **base_ev.provenance,
                "rrf_sources": sources_per_item[key]
            }
        ))

    return fused_evidence


def weighted_score_fusion(
    lists_with_weights: list[tuple[list[Evidence], float]],
    top_k: int = 5
) -> list[Evidence]:
    """Combines evidence lists using min-max normalized weighted score fusion."""
    combined_scores: dict[str, float] = {}
    evidence_map: dict[str, Evidence] = {}

    for ev_list, weight in lists_with_weights:
        if not ev_list:
            continue
        scores = [e.raw_score for e in ev_list]
        min_s = min(scores)
        max_s = max(scores)
        denom = (max_s - min_s) if max_s > min_s else 1.0

        for ev in ev_list:
            key = ev.source_chunk_ids[0] if ev.source_chunk_ids else ev.content[:60]
            norm_score = (ev.raw_score - min_s) / denom
            combined_scores[key] = combined_scores.get(key, 0.0) + (weight * norm_score)
            if key not in evidence_map:
                evidence_map[key] = ev

    sorted_keys = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]

    fused: list[Evidence] = []
    for rank, key in enumerate(sorted_keys, start=1):
        base = evidence_map[key]
        fused.append(Evidence(
            evidence_id=f"ev_weighted_{base.evidence_id}",
            type=base.type,
            content=base.content,
            source_doc_ids=base.source_doc_ids,
            source_chunk_ids=base.source_chunk_ids,
            retriever="hybrid_weighted",
            raw_score=round(combined_scores[key], 4),
            rank=rank,
            provenance=base.provenance
        ))

    return fused
