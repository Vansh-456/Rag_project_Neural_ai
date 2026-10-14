"""
Dense semantic retrieval module using vector embeddings.
"""

from __future__ import annotations

from src.schemas import Chunk, Evidence, EvidenceType, RetrievalBudget
from src.embeddings.embedder import BaseEmbedder
from src.vector_store.store import InMemoryVectorStore, QdrantVectorStore


class DenseRetriever:
    """Retrieves top-K evidence chunks by cosine similarity of embeddings."""

    def __init__(self, embedder: BaseEmbedder, vector_store: InMemoryVectorStore | QdrantVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5, budget: RetrievalBudget | None = None) -> list[Evidence]:
        """Embeds query and searches vector store for closest chunks."""
        if budget and not budget.can_call_retrieval():
            return []

        query_vec = self.embedder.embed_query(query)
        scored_chunks = self.vector_store.search(query_vec, top_k=top_k)

        evidence_list: list[Evidence] = []
        for rank, (chunk, score) in enumerate(scored_chunks, start=1):
            evidence_list.append(Evidence(
                evidence_id=f"ev_dense_{chunk.chunk_id}",
                type=EvidenceType.TEXT_CHUNK,
                content=chunk.text,
                source_doc_ids=[chunk.doc_id],
                source_chunk_ids=[chunk.chunk_id],
                retriever="dense",
                raw_score=round(score, 4),
                rank=rank,
                provenance={
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_heading": chunk.section_heading,
                    "doc_title": chunk.metadata.get("doc_title", ""),
                    "token_count": chunk.token_count
                }
            ))

        if budget:
            budget.record_call(candidates=len(evidence_list))

        return evidence_list
