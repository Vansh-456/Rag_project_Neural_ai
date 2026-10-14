"""Rank fusion over heterogeneous evidence lists.

* Reciprocal Rank Fusion:  score(d) = sum_r w_r / (k + rank_r(d))   (ranks are 1-based)
  Scale-free: only ranks matter, so dense cosine and BM25 scores never need calibrating.
* Weighted score fusion:  score(d) = sum_r w_r * norm_r(s_r(d)), norm = min-max or z-score
  within each list. Sensitive to score distributions; a doc missing from a list gets 0.

Fusion key: evidence is matched across lists by ``fusion_key`` — the chunk id for text
evidence, or the evidence id for graph paths — so the same chunk found by dense and BM25
is merged, and per-source contributions are recorded in ``provenance["fusion"]``.
"""

from __future__ import annotations

import statistics
from dataclasses import replace

from agvrag.schemas import Evidence


def fusion_key(ev: Evidence) -> str:
    if ev.type.value == "text_chunk" and len(ev.source_chunk_ids) == 1:
        return ev.source_chunk_ids[0]
    return ev.evidence_id


def _normalize(scores: list[float], method: str) -> list[float]:
    if not scores:
        return []
    if method == "zscore":
        mu = statistics.fmean(scores)
        sd = statistics.pstdev(scores) or 1.0
        return [(s - mu) / sd for s in scores]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def rrf_fuse(
    lists: dict[str, list[Evidence]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_n: int | None = None,
) -> list[Evidence]:
    """Reciprocal rank fusion. Deterministic tie-break: best single rank, then key."""
    scores: dict[str, float] = {}
    best: dict[str, Evidence] = {}
    contrib: dict[str, dict[str, dict[str, float]]] = {}
    best_rank: dict[str, int] = {}
    for name, evs in lists.items():
        w = (weights or {}).get(name, 1.0)
        for rank, ev in enumerate(evs, start=1):
            key = fusion_key(ev)
            s = w / (k + rank)
            scores[key] = scores.get(key, 0.0) + s
            contrib.setdefault(key, {})[name] = {
                "rank": rank,
                "raw_score": ev.raw_score,
                "rrf": round(s, 6),
            }
            if key not in best or rank < best_rank[key]:
                best[key] = ev
                best_rank[key] = rank
    order = sorted(scores, key=lambda key: (-scores[key], best_rank[key], key))
    if top_n is not None:
        order = order[:top_n]
    out: list[Evidence] = []
    for i, key in enumerate(order, start=1):
        ev = best[key]
        out.append(
            replace(
                ev,
                retriever="+".join(sorted(contrib[key])) if len(contrib[key]) > 1 else ev.retriever,
                raw_score=round(scores[key], 6),
                rank=i,
                provenance={
                    **ev.provenance,
                    "fusion": {"method": "rrf", "k": k, "sources": contrib[key]},
                },
            )
        )
    return out


def weighted_fuse(
    lists: dict[str, list[Evidence]],
    weights: dict[str, float],
    normalization: str = "minmax",
    top_n: int | None = None,
) -> list[Evidence]:
    scores: dict[str, float] = {}
    best: dict[str, Evidence] = {}
    contrib: dict[str, dict[str, dict[str, float]]] = {}
    for name, evs in lists.items():
        w = weights.get(name, 0.0)
        normed = _normalize([e.raw_score for e in evs], normalization)
        for ev, ns in zip(evs, normed, strict=True):
            key = fusion_key(ev)
            scores[key] = scores.get(key, 0.0) + w * ns
            contrib.setdefault(key, {})[name] = {
                "rank": ev.rank,
                "raw_score": ev.raw_score,
                "normalized": round(ns, 6),
                "weight": w,
            }
            best.setdefault(key, ev)
    order = sorted(scores, key=lambda key: (-scores[key], key))
    if top_n is not None:
        order = order[:top_n]
    return [
        replace(
            best[key],
            retriever="+".join(sorted(contrib[key]))
            if len(contrib[key]) > 1
            else best[key].retriever,
            raw_score=round(scores[key], 6),
            rank=i,
            provenance={
                **best[key].provenance,
                "fusion": {
                    "method": "weighted",
                    "normalization": normalization,
                    "sources": contrib[key],
                },
            },
        )
        for i, key in enumerate(order, start=1)
    ]
