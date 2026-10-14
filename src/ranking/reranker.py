"""
Cross-encoder reranking module.
Re-scores candidate evidence items using full query-document cross-attention.
"""

from __future__ import annotations

import re
from typing import Any
from src.schemas import Evidence


class CrossEncoderReranker:
    """Reranks candidate evidence using cross-encoder scoring or heuristic token-interaction scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = None

    def _init_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = "HEURISTIC_FALLBACK"

    def _heuristic_score(self, query: str, content: str) -> float:
        """Deterministic lexical & exact match interaction scoring for fallback."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        if not q_tokens:
            return 0.0

        c_tokens = re.findall(r"\w+", content.lower())
        overlap = sum(1 for t in c_tokens if t in q_tokens)

        # Exact phrase bonus
        phrase_bonus = 0.5 if query.lower() in content.lower() else 0.0
        return round((overlap / max(1, len(c_tokens))) * 10.0 + phrase_bonus, 4)

    def rerank(self, query: str, evidence: list[Evidence], top_k: int = 5) -> list[Evidence]:
        """Reranks candidate evidence and updates rerank_score and rank."""
        if not evidence:
            return []

        self._init_model()

        if self._model == "HEURISTIC_FALLBACK":
            scored = []
            for ev in evidence:
                score = self._heuristic_score(query, ev.content)
                scored.append((ev, score))
        else:
            pairs = [[query, ev.content] for ev in evidence]
            scores = self._model.predict(pairs, batch_size=self.batch_size)
            scored = list(zip(evidence, [float(s) for s in scores]))

        # Sort descending by rerank score
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked_results: list[Evidence] = []
        for new_rank, (ev, score) in enumerate(scored[:top_k], start=1):
            reranked_ev = Evidence(
                evidence_id=ev.evidence_id,
                type=ev.type,
                content=ev.content,
                source_doc_ids=ev.source_doc_ids,
                source_chunk_ids=ev.source_chunk_ids,
                retriever=ev.retriever,
                raw_score=ev.raw_score,
                rank=new_rank,
                rerank_score=round(score, 4),
                provenance={
                    **ev.provenance,
                    "previous_rank": ev.rank,
                    "rerank_model": self.model_name
                }
            )
            reranked_results.append(reranked_ev)

        return reranked_results
