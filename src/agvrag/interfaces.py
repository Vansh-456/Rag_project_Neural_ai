"""Component interfaces.

Each Protocol has at least one real and one fake/test implementation. A ``Pipeline`` is a
configuration of these components, so vector-only, hybrid, graph-only, adaptive and verified
systems are different YAML files over the same interfaces.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from agvrag.schemas import (
    Chunk,
    Document,
    Entity,
    Evidence,
    GraphPath,
    QueryAnalysis,
    Relation,
    RoutingDecision,
)


@runtime_checkable
class Parser(Protocol):
    def parse(self, path: Path, metadata: dict[str, Any] | None = None) -> Document:
        """Parse a file into a structured Document. Raises ``ValueError`` if unreadable."""
        ...


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a document. Output chunks are non-empty and have deterministic ids."""
        ...


@runtime_checkable
class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalized vectors."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 L2-normalized vector (query prefix applied if needed)."""
        ...


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Idempotent: re-upserting the same chunk ids overwrites, never duplicates."""
        ...

    def search(
        self, vector: np.ndarray, k: int, doc_filter: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        """Top-k by cosine similarity, descending. Optional restriction to ``doc_filter``."""
        ...

    def count(self) -> int: ...

    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]: ...


@runtime_checkable
class SparseIndex(Protocol):
    def search(self, query: str, k: int) -> list[ScoredChunk]: ...


@runtime_checkable
class GraphStore(Protocol):
    def add_entities(self, entities: Sequence[Entity]) -> None: ...

    def add_relations(self, relations: Sequence[Relation]) -> None: ...

    def get_entity(self, entity_id: str) -> Entity | None: ...

    def entities(self) -> list[Entity]: ...

    def relations(self) -> list[Relation]: ...

    def neighbors(
        self, entity_id: str, relation_types: Sequence[str] | None = None
    ) -> list[tuple[Relation, str, str]]:
        """Return (relation, neighbor_id, direction) for all incident edges.

        ``direction`` is "out" when the relation's source is ``entity_id``.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, analysis: QueryAnalysis, k: int) -> list[Evidence]: ...


@runtime_checkable
class QueryAnalyzerP(Protocol):
    def analyze(self, query: str) -> QueryAnalysis: ...


@runtime_checkable
class Router(Protocol):
    def route(self, analysis: QueryAnalysis) -> RoutingDecision: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[Evidence], top_k: int) -> list[Evidence]:
        """Return at most ``top_k`` evidence items with ``rerank_score`` set, re-ranked."""
        ...


@dataclass
class BuiltContext:
    text: str
    evidence_ids: dict[str, Evidence]  # citation label (E1/G1) -> evidence
    token_count: int
    dropped: list[str] = field(default_factory=list)


@runtime_checkable
class ContextBuilder(Protocol):
    def build(
        self, query: str, analysis: QueryAnalysis, evidence: list[Evidence]
    ) -> BuiltContext: ...


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def complete(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse: ...

    def stream(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> Iterator[str]: ...


@runtime_checkable
class Verifier(Protocol):
    def verify(
        self, answer: str, citations: list[dict[str, Any]], context: BuiltContext
    ) -> dict[str, Any]: ...


@runtime_checkable
class GraphTraverser(Protocol):
    def paths(
        self, seeds: dict[str, float], max_hops: int, relation_types: Sequence[str] | None
    ) -> list[GraphPath]: ...
