"""
Vector store implementations for dense retrieval.
Supports high-performance in-memory search and Qdrant integration.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from src.schemas import Chunk


class InMemoryVectorStore:
    """Fast, in-memory numpy-based vector store with cosine similarity."""

    def __init__(self, collection_name: str = "rag_corpus", dimension: int = 384):
        self.collection_name = collection_name
        self.dimension = dimension
        self.chunks: list[Chunk] = []
        self.chunk_ids: list[str] = []
        self.vectors: np.ndarray | None = None

    def index_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Indexes chunks with their corresponding dense embeddings."""
        if not chunks:
            return

        new_vecs = np.array(embeddings, dtype=np.float32)
        # Normalize vectors for fast dot-product cosine similarity
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vecs = new_vecs / norms

        if self.vectors is None:
            self.vectors = normalized_vecs
            self.chunks = list(chunks)
            self.chunk_ids = [c.chunk_id for c in chunks]
        else:
            self.vectors = np.vstack([self.vectors, normalized_vecs])
            self.chunks.extend(chunks)
            self.chunk_ids.extend([c.chunk_id for c in chunks])

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Searches for top-K most similar chunks by cosine similarity."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = np.dot(self.vectors, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[Chunk, float]] = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results

    def clear(self):
        self.chunks = []
        self.chunk_ids = []
        self.vectors = None


class QdrantVectorStore:
    """Vector store backed by Qdrant engine."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "rag_corpus",
        dimension: int = 384
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self._client: Any = None
        self._fallback_store: InMemoryVectorStore | None = None

    def _get_client(self):
        if self._client is None and self._fallback_store is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http.models import Distance, VectorParams
                client = QdrantClient(host=self.host, port=self.port, timeout=3.0)
                # Check connection
                collections = client.get_collections().collections
                names = [c.name for c in collections]
                if self.collection_name not in names:
                    client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                    )
                self._client = client
            except Exception:
                # If Qdrant is unreachable, transparently fall back to in-memory store
                self._fallback_store = InMemoryVectorStore(self.collection_name, self.dimension)

    def index_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._get_client()
        if self._fallback_store is not None:
            self._fallback_store.index_chunks(chunks, embeddings)
            return

        from qdrant_client.http.models import PointStruct
        points = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "section_heading": chunk.section_heading,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_count": chunk.token_count,
                "strategy": chunk.strategy,
                "metadata": chunk.metadata
            }
            points.append(PointStruct(id=idx, vector=emb, payload=payload))

        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        self._get_client()
        if self._fallback_store is not None:
            return self._fallback_store.search(query_vector, top_k)

        res = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        results: list[tuple[Chunk, float]] = []
        for scored_point in res:
            p = scored_point.payload or {}
            chunk = Chunk(
                chunk_id=p.get("chunk_id", ""),
                doc_id=p.get("doc_id", ""),
                text=p.get("text", ""),
                section_heading=p.get("section_heading"),
                page_start=p.get("page_start"),
                page_end=p.get("page_end"),
                strategy=p.get("strategy", "dense"),
                token_count=p.get("token_count", 0),
                metadata=p.get("metadata", {})
            )
            results.append((chunk, float(scored_point.score)))
        return results


def get_vector_store(backend: str = "in_memory", host: str = "localhost", port: int = 6333, collection_name: str = "rag_corpus", dimension: int = 384):
    """Factory helper for vector stores."""
    if backend == "qdrant":
        return QdrantVectorStore(host=host, port=port, collection_name=collection_name, dimension=dimension)
    return InMemoryVectorStore(collection_name=collection_name, dimension=dimension)
