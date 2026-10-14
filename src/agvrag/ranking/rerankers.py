"""Rerankers: re-score a candidate pool and keep the top-k.

* ``CrossEncoderReranker`` — jointly encodes (query, passage) pairs, e.g.
  ``cross-encoder/ms-marco-MiniLM-L-6-v2`` or ``BAAI/bge-reranker-base``. Far more accurate
  than bi-encoder similarity because query and passage tokens attend to each other, but
  costs one forward pass per candidate (latency grows linearly with pool size N).
* ``LexicalReranker`` — query-term coverage; cheap, deterministic baseline/test double.
* ``IdentityReranker`` — keeps the incoming order (ablation control).

Graph-path evidence is reranked on its serialized text like any other passage.
"""

from __future__ import annotations

import re
import threading
from dataclasses import replace

from agvrag.config import RerankerConfig
from agvrag.retrieval.bm25 import STOPWORDS
from agvrag.schemas import Evidence


def _finalize(cands: list[Evidence], scores: list[float], top_k: int) -> list[Evidence]:
    order = sorted(range(len(cands)), key=lambda i: (-scores[i], cands[i].rank))
    return [
        replace(
            cands[i],
            rerank_score=round(float(scores[i]), 6),
            rank=new_rank,
            provenance={**cands[i].provenance, "pre_rerank_rank": cands[i].rank},
        )
        for new_rank, i in enumerate(order[:top_k], start=1)
    ]


class IdentityReranker:
    name = "identity"

    def rerank(self, query: str, candidates: list[Evidence], top_k: int) -> list[Evidence]:
        return _finalize(candidates, [-float(e.rank) for e in candidates], top_k)


class LexicalReranker:
    name = "lexical"

    def rerank(self, query: str, candidates: list[Evidence], top_k: int) -> list[Evidence]:
        q = {t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS}
        scores = []
        for ev in candidates:
            toks = set(re.findall(r"\w+", ev.content.lower()))
            scores.append(len(q & toks) / max(1, len(q)) + 1e-3 / ev.rank)
        return _finalize(candidates, scores, top_k)


class CrossEncoderReranker:
    def __init__(
        self, model: str, batch_size: int = 32, device: str = "cpu", max_chars: int = 2000
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.name = f"cross_encoder:{model}"
        self.model = CrossEncoder(model, device=device)
        self.batch_size = batch_size
        self.max_chars = max_chars
        self._lock = threading.Lock()

    def rerank(self, query: str, candidates: list[Evidence], top_k: int) -> list[Evidence]:
        if not candidates:
            return []
        pairs = [(query, ev.content[: self.max_chars]) for ev in candidates]
        with self._lock:
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return _finalize(candidates, [float(s) for s in scores], top_k)


def build_reranker(cfg: RerankerConfig, device: str = "cpu") -> object:
    if cfg.provider == "identity":
        return IdentityReranker()
    if cfg.provider == "lexical":
        return LexicalReranker()
    return CrossEncoderReranker(cfg.model, cfg.batch_size, device)
