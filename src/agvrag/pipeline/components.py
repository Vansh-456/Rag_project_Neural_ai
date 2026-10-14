"""Loading and sharing heavy components (embedder, indexes, graph, models).

``ComponentRegistry`` caches components by identity (index name, model name), so the API can
serve eight pipelines while loading each embedder / index / graph exactly once.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agvrag.chunking.chunkers import chunk_embed_text
from agvrag.config import Secrets, Settings
from agvrag.embeddings.embedders import build_embedder
from agvrag.graph.gazetteer import Gazetteer
from agvrag.graph.linking import EntityLinker
from agvrag.graph.store import Neo4jGraphStore, NetworkXGraphStore
from agvrag.retrieval.bm25 import build_bm25
from agvrag.schemas import Chunk
from agvrag.utils.io import read_jsonl
from agvrag.utils.logging import get_logger
from agvrag.vector_store.stores import InMemoryVectorStore, QdrantVectorStore

log = get_logger(__name__)


class IndexNotBuiltError(FileNotFoundError):
    pass


def index_dir(settings: Settings) -> Path:
    return settings.path("index_dir") / settings.index_name


def load_chunks(settings: Settings) -> dict[str, Chunk]:
    path = index_dir(settings) / "chunks.jsonl"
    if not path.exists():
        raise IndexNotBuiltError(
            f"index '{settings.index_name}' not built ({path} missing). Run: agvrag index"
        )
    return {d["chunk_id"]: Chunk.from_dict(d) for d in read_jsonl(path)}


def collection_name(settings: Settings) -> str:
    return settings.vector_store.collection or settings.index_name


def open_vector_store(settings: Settings, dim: int, chunks: dict[str, Chunk] | None = None) -> Any:
    vs = settings.vector_store
    idx = index_dir(settings)
    if vs.backend == "memory":
        if chunks is not None and (idx / "vectors.npy").exists():
            return InMemoryVectorStore.load(idx, chunks)
        return InMemoryVectorStore(dim)
    if vs.backend == "qdrant":
        return QdrantVectorStore(collection_name(settings), dim, url=vs.url)
    return QdrantVectorStore(collection_name(settings), dim, path=idx / "qdrant")


@dataclass
class IndexBundle:
    chunks: dict[str, Chunk]
    vector_store: Any
    bm25: Any


class ComponentRegistry:
    def __init__(self, secrets: Secrets | None = None) -> None:
        self.secrets = secrets or Secrets()
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, ...], Any] = {}

    def _get(self, key: tuple[str, ...], factory: Any) -> Any:
        with self._lock:
            if key not in self._cache:
                self._cache[key] = factory()
            return self._cache[key]

    def embedder(self, s: Settings) -> Any:
        e = s.embedding
        key = (
            "embedder",
            e.provider,
            e.model,
            str(e.hashing_dim),
            e.query_prefix,
            e.document_prefix,
        )
        return self._get(key, lambda: build_embedder(e, s.path("cache_dir")))

    def index(self, s: Settings) -> IndexBundle:
        b = s.bm25
        key = (
            "index",
            s.index_name,
            s.vector_store.backend,
            b.implementation,
            str(b.k1),
            str(b.b),
            str(b.stem),
            str(b.remove_stopwords),
        )

        def make() -> IndexBundle:
            chunks = load_chunks(s)
            emb = self.embedder(s)
            store = self._get(
                ("vstore", s.index_name, s.vector_store.backend),
                lambda: open_vector_store(s, emb.dim, chunks),
            )
            ordered = list(chunks.values())
            bm25 = build_bm25(ordered, [chunk_embed_text(c) for c in ordered], b)
            log.info("index loaded", extra={"index": s.index_name, "chunks": len(chunks)})
            return IndexBundle(chunks, store, bm25)

        return self._get(key, make)

    def gazetteer(self, s: Settings) -> Gazetteer:
        path = s.path("gazetteer_file")
        return self._get(("gazetteer", str(path)), lambda: Gazetteer.load(path))

    def graph(self, s: Settings) -> Any:
        key = ("graph", s.index_name, s.graph.backend)

        def make() -> Any:
            if s.graph.backend == "neo4j":
                if not self.secrets.neo4j_password:
                    raise ValueError("NEO4J_PASSWORD is required for the neo4j graph backend")
                return Neo4jGraphStore(
                    s.graph.neo4j_uri, s.graph.neo4j_user, self.secrets.neo4j_password
                )
            path = index_dir(s) / "graph.json"
            if not path.exists():
                raise IndexNotBuiltError(f"graph not built ({path} missing). Run: agvrag graph")
            return NetworkXGraphStore.load(path)

        return self._get(key, make)

    def linker(self, s: Settings) -> EntityLinker:
        key = ("linker", s.index_name, s.graph.backend)
        return self._get(
            key,
            lambda: EntityLinker(
                self.graph(s),
                s.graph.fuzzy_link_threshold,
                self.embedder(s),
                s.graph.embedding_link_threshold,
            ),
        )

    def reranker(self, s: Settings) -> Any:
        from agvrag.ranking.rerankers import build_reranker

        r = s.reranker
        return self._get(
            ("reranker", r.provider, r.model), lambda: build_reranker(r, s.embedding.device)
        )

    def llm(self, s: Settings, effort: str | None = None) -> Any:
        from agvrag.generation.llm import build_llm

        g = s.generation
        key = ("llm", g.provider, g.model, effort or g.effort)
        return self._get(key, lambda: build_llm(g, self.secrets, s.path("cache_dir"), effort))

    def invalidate_index(self, s: Settings) -> None:
        """Drop cached index/graph/linker objects after (re)indexing or ingesting."""
        with self._lock:
            for k in list(self._cache):
                if len(k) > 1 and k[1] == s.index_name and k[0] in {"index", "graph", "linker"}:
                    del self._cache[k]
