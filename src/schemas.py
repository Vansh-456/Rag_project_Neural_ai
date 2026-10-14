"""
Core architectural schemas and Protocol interfaces for the Adaptive Graph + Vector RAG system.
All components communicate through these typed objects to ensure composability,
clean testing, and trustworthy ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvidenceType(str, Enum):
    TEXT_CHUNK = "text_chunk"
    GRAPH_PATH = "graph_path"
    GRAPH_FACT = "graph_fact"


class QueryIntent(str, Enum):
    DEFINITION = "definition"
    RELATIONSHIP = "relationship"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    EXPERIMENTAL = "experimental"


class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    GRAPH = "graph"
    MULTI_HOP = "multi_hop"
    ADAPTIVE = "adaptive"


class EntityType(str, Enum):
    PAPER = "Paper"
    AUTHOR = "Author"
    METHOD = "Method"
    MODEL = "Model"
    DATASET = "Dataset"
    METRIC = "Metric"
    TASK = "Task"
    CONCEPT = "Concept"


class RelationType(str, Enum):
    AUTHORED_BY = "AUTHORED_BY"
    CITES = "CITES"
    INTRODUCES = "INTRODUCES"
    USES = "USES"
    EVALUATES_ON = "EVALUATES_ON"
    IMPROVES = "IMPROVES"
    EXTENDS = "EXTENDS"
    BASED_ON = "BASED_ON"
    APPLIED_TO = "APPLIED_TO"
    INFLUENCES = "INFLUENCES"


class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


# ---------------------------------------------------------------------------
# Document & Chunk Schemas
# ---------------------------------------------------------------------------

@dataclass
class Section:
    heading: str
    level: int
    text: str
    page_start: int = 1
    page_end: int = 1


@dataclass
class Document:
    doc_id: str
    title: str
    authors: list[str]
    year: int | None
    sections: list[Section]
    references: list[str]
    source_path: str
    abstract: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section_heading: str | None
    page_start: int | None
    page_end: int | None
    strategy: str  # fixed | recursive | semantic | structure_aware
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Knowledge Graph Schemas
# ---------------------------------------------------------------------------

@dataclass
class GraphEntity:
    entity_id: str
    name: str
    entity_type: EntityType
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    source_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelation:
    source_id: str
    target_id: str
    relation_type: RelationType
    source_doc_ids: list[str] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphPath:
    nodes: list[GraphEntity]
    relations: list[GraphRelation]
    path_score: float
    description: str  # Serialized textual description (e.g., LoRA -INTRODUCES-> ...)


# ---------------------------------------------------------------------------
# Evidence & Query Analysis Schemas
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    evidence_id: str
    type: EvidenceType
    content: str  # text passage or serialized path/triple
    source_doc_ids: list[str]
    source_chunk_ids: list[str]
    retriever: str  # dense | bm25 | graph | hybrid | ...
    raw_score: float
    rank: int
    rerank_score: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)  # path nodes/edges, pages, etc.


@dataclass
class QueryAnalysis:
    query: str
    intent: str  # definition | relationship | comparison | multi_hop | experimental
    entities: list[str]
    complexity: float  # 0.0 to 1.0
    hops_estimate: int
    rationale: str  # human-readable explanation of router inputs


@dataclass
class RoutingDecision:
    strategy: str  # vector | graph | hybrid | multi_hop
    params: dict[str, Any]  # k, hops, candidate pool size, etc.
    rationale: str


# ---------------------------------------------------------------------------
# Verification & Budget Schemas
# ---------------------------------------------------------------------------

@dataclass
class ClaimVerification:
    claim: str
    status: VerificationStatus
    cited_evidence_ids: list[str]
    reasoning: str
    confidence: float = 1.0


@dataclass
class VerificationResult:
    is_sufficient: bool
    claims: list[ClaimVerification]
    supported_ratio: float
    contradiction_found: bool
    retry_recommended: bool
    reformulated_query: str | None = None


@dataclass
class RetrievalBudget:
    max_retrieval_calls: int = 5
    max_graph_hops: int = 3
    max_candidate_pool: int = 50
    max_latency_ms: float = 5000.0
    max_token_budget: int = 4096
    cost_per_1k_prompt_tokens: float = 0.00015
    cost_per_1k_completion_tokens: float = 0.0006

    calls_used: int = 0
    hops_used: int = 0
    candidates_considered: int = 0
    elapsed_ms: float = 0.0
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0

    def can_call_retrieval(self) -> bool:
        return self.calls_used < self.max_retrieval_calls and self.elapsed_ms < self.max_latency_ms

    def can_traverse_graph(self, hops: int = 1) -> bool:
        return (self.hops_used + hops) <= self.max_graph_hops and self.elapsed_ms < self.max_latency_ms

    def record_call(self, hops: int = 0, candidates: int = 0, elapsed_ms: float = 0.0):
        self.calls_used += 1
        self.hops_used += hops
        self.candidates_considered += candidates
        self.elapsed_ms += elapsed_ms

    def record_tokens(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens_used += prompt_tokens
        self.completion_tokens_used += completion_tokens

    def total_cost_usd(self) -> float:
        prompt_cost = (self.prompt_tokens_used / 1000.0) * self.cost_per_1k_prompt_tokens
        completion_cost = (self.completion_tokens_used / 1000.0) * self.cost_per_1k_completion_tokens
        return round(prompt_cost + completion_cost, 6)


# ---------------------------------------------------------------------------
# Pipeline Trace
# ---------------------------------------------------------------------------

@dataclass
class PipelineTrace:
    query_analysis: QueryAnalysis
    routing: RoutingDecision
    retrieved: list[Evidence]
    reranked: list[Evidence]
    context: str
    answer: str | None
    citations: list[dict[str, Any]]
    verification: dict[str, Any] | None
    costs: dict[str, Any]  # latency_ms per stage, tokens in/out, retrieval calls, graph hops, cost_usd


# ---------------------------------------------------------------------------
# Pydantic Request / Response Models for API
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    pipeline: str = Field(default="adaptive", description="vector | bm25 | hybrid | graph | multi_hop | adaptive | adaptive_verified")
    top_k: int = Field(default=5, ge=1, le=20)
    enable_rerank: bool = True
    enable_verification: bool = True
    llm_provider: str | None = None


class CompareRequest(BaseModel):
    query: str
    pipeline_a: str = Field(default="vector")
    pipeline_b: str = Field(default="adaptive_verified")
    top_k: int = Field(default=5)


class QueryResponse(BaseModel):
    trace_id: str
    query: str
    answer: str
    pipeline: str
    strategy_chosen: str
    citations: list[dict[str, Any]]
    evidence_used: list[dict[str, Any]]
    verification: dict[str, Any] | None
    metrics: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    vector_store_status: str
    graph_store_status: str
    document_count: int
    chunk_count: int
    entity_count: int
    relation_count: int


# ---------------------------------------------------------------------------
# Protocol Interfaces
# ---------------------------------------------------------------------------

@runtime_checkable
class Parser(Protocol):
    def parse(self, file_path: str) -> Document:
        ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]:
        ...


@runtime_checkable
class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...


@runtime_checkable
class VectorStore(Protocol):
    def index_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        ...


@runtime_checkable
class SparseIndex(Protocol):
    def index_chunks(self, chunks: list[Chunk]) -> None:
        ...

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        ...


@runtime_checkable
class GraphStore(Protocol):
    def add_entity(self, entity: GraphEntity) -> None:
        ...

    def add_relation(self, relation: GraphRelation) -> None:
        ...

    def get_neighbors(self, entity_id: str, depth: int = 1) -> list[GraphPath]:
        ...

    def find_paths_between(self, source_id: str, target_id: str, max_depth: int = 3) -> list[GraphPath]:
        ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5, budget: RetrievalBudget | None = None) -> list[Evidence]:
        ...


@runtime_checkable
class Router(Protocol):
    def analyze_query(self, query: str) -> QueryAnalysis:
        ...

    def route(self, analysis: QueryAnalysis) -> RoutingDecision:
        ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, evidence: list[Evidence], top_k: int = 5) -> list[Evidence]:
        ...


@runtime_checkable
class ContextBuilder(Protocol):
    def build_context(self, evidence: list[Evidence], query: str, max_tokens: int = 2048) -> tuple[str, list[Evidence]]:
        ...


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, prompt: str, system_prompt: str | None = None) -> tuple[str, dict[str, int]]:
        ...


@runtime_checkable
class Verifier(Protocol):
    def verify(self, answer: str, evidence: list[Evidence], query: str) -> VerificationResult:
        ...


@runtime_checkable
class Pipeline(Protocol):
    def run(self, query: str, top_k: int = 5, budget: RetrievalBudget | None = None) -> PipelineTrace:
        ...
