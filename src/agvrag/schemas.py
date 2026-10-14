"""Typed data contracts shared by every component.

All components communicate through these objects so that pipelines can be composed and
ablated by configuration alone. Every class round-trips through ``to_dict``/``from_dict``
(JSON-safe) so artifacts and traces can be persisted and reloaded losslessly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


@dataclass
class Section:
    """A contiguous region of a document under one heading.

    Invariant: ``page_start <= page_end`` (1-indexed pages).
    """

    heading: str
    level: int
    text: str
    page_start: int
    page_end: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Section:
        return cls(**d)


@dataclass
class Document:
    """A parsed paper. ``doc_id`` is stable across runs (arXiv id when known)."""

    doc_id: str
    title: str
    authors: list[str]
    year: int | None
    sections: list[Section]
    references: list[str]
    source_path: str
    abstract: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        d = dict(d)
        d["sections"] = [Section.from_dict(s) for s in d.get("sections", [])]
        return cls(**d)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


@dataclass
class Chunk:
    """The unit of retrieval.

    ``chunk_id`` is deterministic: it depends only on (doc_id, strategy, params, ordinal),
    so re-chunking with the same config yields the same ids (idempotent indexing).
    """

    chunk_id: str
    doc_id: str
    text: str
    section_heading: str | None
    page_start: int | None
    page_end: int | None
    strategy: str  # fixed | recursive | semantic | structure_aware
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)


class EvidenceType(StrEnum):
    TEXT_CHUNK = "text_chunk"
    GRAPH_PATH = "graph_path"
    GRAPH_FACT = "graph_fact"


@dataclass
class Evidence:
    """A retrieved candidate. ``rank`` is 1-based within the list that produced it."""

    evidence_id: str
    type: EvidenceType
    content: str
    source_doc_ids: list[str]
    source_chunk_ids: list[str]
    retriever: str
    raw_score: float
    rank: int
    rerank_score: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evidence:
        d = dict(d)
        d["type"] = EvidenceType(d["type"])
        return cls(**d)


@dataclass
class QueryAnalysis:
    query: str
    intent: str  # definition | relationship | comparison | multi_hop | experimental
    entities: list[str]
    complexity: float  # 0..1
    hops_estimate: int
    rationale: str
    linked_entities: list[dict[str, Any]] = field(default_factory=list)
    relation_cues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoutingDecision:
    strategy: str  # vector | bm25 | hybrid | graph | graph_vector | multi_hop
    params: dict[str, Any]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineTrace:
    """Everything the UI and the evaluator need to inspect one query execution."""

    trace_id: str
    pipeline: str
    query_analysis: QueryAnalysis
    routing: RoutingDecision
    retrieved: list[Evidence]
    reranked: list[Evidence]
    context: str
    answer: str | None
    citations: list[dict[str, Any]]
    verification: dict[str, Any] | None
    costs: dict[str, float]
    retrieval_lists: dict[str, list[Evidence]] = field(default_factory=dict)
    evidence_pool: list[Evidence] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""

    def ranked_chunk_ids(self) -> list[str]:
        """Final ranking flattened to unique chunk ids (used by retrieval metrics)."""
        source = self.reranked or self.evidence_pool or self.retrieved
        out: list[str] = []
        seen: set[str] = set()
        for ev in source:
            for cid in ev.source_chunk_ids:
                if cid not in seen:
                    seen.add(cid)
                    out.append(cid)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "pipeline": self.pipeline,
            "query_analysis": self.query_analysis.to_dict(),
            "routing": self.routing.to_dict(),
            "retrieved": [e.to_dict() for e in self.retrieved],
            "reranked": [e.to_dict() for e in self.reranked],
            "context": self.context,
            "answer": self.answer,
            "citations": self.citations,
            "verification": self.verification,
            "costs": self.costs,
            "retrieval_lists": {
                k: [e.to_dict() for e in v] for k, v in self.retrieval_lists.items()
            },
            "evidence_pool": [e.to_dict() for e in self.evidence_pool],
            "iterations": self.iterations,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineTrace:
        return cls(
            trace_id=d["trace_id"],
            pipeline=d["pipeline"],
            query_analysis=QueryAnalysis(**d["query_analysis"]),
            routing=RoutingDecision(**d["routing"]),
            retrieved=[Evidence.from_dict(e) for e in d["retrieved"]],
            reranked=[Evidence.from_dict(e) for e in d["reranked"]],
            context=d["context"],
            answer=d.get("answer"),
            citations=d.get("citations", []),
            verification=d.get("verification"),
            costs=d.get("costs", {}),
            retrieval_lists={
                k: [Evidence.from_dict(e) for e in v]
                for k, v in d.get("retrieval_lists", {}).items()
            },
            evidence_pool=[Evidence.from_dict(e) for e in d.get("evidence_pool", [])],
            iterations=d.get("iterations", []),
            created_at=d.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Knowledge-graph contracts
# ---------------------------------------------------------------------------

ENTITY_TYPES: tuple[str, ...] = (
    "Paper",
    "Author",
    "Method",
    "Model",
    "Dataset",
    "Metric",
    "Task",
    "Concept",
)

RELATION_TYPES: tuple[str, ...] = (
    "AUTHORED_BY",
    "CITES",
    "INTRODUCES",
    "USES",
    "EVALUATES_ON",
    "IMPROVES",
    "EXTENDS",
    "BASED_ON",
    "APPLIED_TO",
    "INFLUENCES",
)


@dataclass
class Entity:
    """A canonical graph node. ``entity_id`` = ``<type-lower>:<normalized-name>``."""

    entity_id: str
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Entity:
        return cls(**d)


@dataclass
class Relation:
    """A typed, directed edge with mandatory provenance.

    Invariant: ``source_chunk_ids`` is non-empty for text-derived edges; metadata-derived
    edges (e.g. AUTHORED_BY from arXiv metadata) carry ``properties["origin"]``.
    """

    source: str
    target: str
    type: str
    confidence: float
    source_chunk_ids: list[str]
    evidence_text: str = ""
    extractor: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.type, self.target)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Relation:
        return cls(**d)


@dataclass
class GraphPath:
    """An ordered walk ``nodes[0] -edges[0]-> nodes[1] ...``.

    ``directions[i]`` is ``"out"`` if edges[i] points from nodes[i] to nodes[i+1], else ``"in"``.
    """

    nodes: list[str]
    edges: list[Relation]
    directions: list[str]
    score: float

    @property
    def hops(self) -> int:
        return len(self.edges)
