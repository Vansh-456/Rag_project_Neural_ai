"""Vector stores.

* ``InMemoryVectorStore`` — exact cosine search with numpy (vectors are L2-normalized, so
  cosine = dot product). Persists to ``vectors.npy`` + ``ids.json``. Used in tests and as a
  zero-dependency backend; exact search is fine up to ~10^5 chunks on CPU.
* ``QdrantVectorStore`` — Qdrant via ``qdrant-client``: either an embedded local store
  (``path=...``, no server needed) or a server (``url=...``, the Docker deployment). Payload
  holds the full chunk (text + metadata) so search results are self-contained.

Both are idempotent under re-upsert because point ids are derived from chunk ids.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from agvrag.interfaces import ScoredChunk
from agvrag.schemas import Chunk
from agvrag.utils.ids import uuid_from_string
from agvrag.utils.logging import get_logger

log = get_logger(__name__)


class InMemoryVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._chunks: dict[str, Chunk] = {}
        self._mat = np.zeros((0, dim), dtype=np.float32)

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if len(chunks) and vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != store dim {self.dim}")
        new_rows: list[np.ndarray] = []
        for c, v in zip(chunks, vectors, strict=True):
            self._chunks[c.chunk_id] = c
            if c.chunk_id in self._pos:
                pos = self._pos[c.chunk_id]
                if pos < len(self._mat):
                    self._mat[pos] = v
                else:  # duplicate id within this same batch
                    new_rows[pos - len(self._mat)] = v
            else:
                self._pos[c.chunk_id] = len(self._ids)
                self._ids.append(c.chunk_id)
                new_rows.append(v)
        if new_rows:
            self._mat = np.vstack([self._mat, np.stack(new_rows).astype(np.float32)])

    def search(
        self, vector: np.ndarray, k: int, doc_filter: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        if not self._ids or k <= 0:
            return []
        scores = self._mat @ vector.astype(np.float32)
        if doc_filter is not None:
            allowed = set(doc_filter)
            mask = np.array([self._chunks[i].doc_id in allowed for i in self._ids])
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            ScoredChunk(self._chunks[self._ids[i]], float(scores[i]))
            for i in top
            if np.isfinite(scores[i])
        ]

    def count(self) -> int:
        return len(self._ids)

    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [self._chunks[c] for c in chunk_ids if c in self._chunks]

    def all_chunks(self) -> list[Chunk]:
        return [self._chunks[i] for i in self._ids]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._mat)
        (directory / "ids.json").write_text(json.dumps(self._ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, chunks: dict[str, Chunk]) -> InMemoryVectorStore:
        mat = np.load(directory / "vectors.npy")
        ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        store = cls(mat.shape[1])
        store.upsert([chunks[i] for i in ids], mat)
        return store


class QdrantVectorStore:
    def __init__(
        self,
        collection: str,
        dim: int,
        path: Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm

        self._qm = qm
        if url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        elif path is not None:
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(location=":memory:")
        self.collection = collection
        self.dim = dim
        self._server = bool(url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        qm = self._qm
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )
            if self._server:  # payload indexes only exist on a Qdrant server
                self.client.create_payload_index(
                    self.collection, field_name="doc_id", field_schema=qm.PayloadSchemaType.KEYWORD
                )
            log.info("created qdrant collection", extra={"collection": self.collection})

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray, batch: int = 256) -> None:
        qm = self._qm
        for i in range(0, len(chunks), batch):
            points = [
                qm.PointStruct(
                    id=uuid_from_string(c.chunk_id), vector=v.tolist(), payload=c.to_dict()
                )
                for c, v in zip(chunks[i : i + batch], vectors[i : i + batch], strict=True)
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def search(
        self, vector: np.ndarray, k: int, doc_filter: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        qm = self._qm
        flt = None
        if doc_filter is not None:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=list(doc_filter)))]
            )
        res = self.client.query_points(
            self.collection, query=vector.tolist(), limit=k, query_filter=flt, with_payload=True
        )
        return [
            ScoredChunk(Chunk.from_dict(_payload(p.payload)), float(p.score)) for p in res.points
        ]

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        pts = self.client.retrieve(
            self.collection, ids=[uuid_from_string(c) for c in chunk_ids], with_payload=True
        )
        by_id = {
            p.payload["chunk_id"]: Chunk.from_dict(_payload(p.payload)) for p in pts if p.payload
        }
        return [by_id[c] for c in chunk_ids if c in by_id]

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection)

    def close(self) -> None:
        self.client.close()


def _payload(p: dict[str, Any] | None) -> dict[str, Any]:
    if p is None:
        raise ValueError("qdrant point without payload")
    return dict(p)
