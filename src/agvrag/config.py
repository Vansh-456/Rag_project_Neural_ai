"""Configuration: YAML files (with ``extends`` inheritance) + environment overrides.

Resolution order (later wins):
  1. ``configs/base.yaml``
  2. each file in the ``extends`` chain of the requested config
  3. the requested config file
  4. environment variables ``AGVRAG__SECTION__KEY=value`` (YAML-parsed values)

Secrets never live in YAML; they are read from the environment / ``.env`` via ``Secrets``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Section):
    corpus_file: str = "data/corpus/papers.yaml"
    gazetteer_file: str = "data/corpus/gazetteer.yaml"
    raw_dir: str = "data/raw/papers"
    documents_dir: str = "data/processed/documents"
    chunks_dir: str = "data/processed/chunks"
    entities_dir: str = "data/processed/entities"
    graph_dir: str = "data/processed/graph"
    index_dir: str = "data/index"
    cache_dir: str = "data/cache"
    results_dir: str = "results"
    traces_db: str = "data/traces/traces.sqlite"
    evaluation_dir: str = "data/evaluation"


class IngestionConfig(_Section):
    parser: Literal["pymupdf", "pdfplumber"] = "pymupdf"
    arxiv_delay_seconds: float = 3.0
    min_section_chars: int = 40
    max_heading_words: int = 12


class ChunkingConfig(_Section):
    strategy: Literal["fixed", "recursive", "semantic", "structure_aware"] = "structure_aware"
    chunk_size: int = 256
    overlap: int = 32
    semantic_threshold: float = 0.55
    semantic_min_tokens: int = 64
    include_title_context: bool = True


class EmbeddingConfig(_Section):
    provider: Literal["sentence_transformers", "hashing"] = "sentence_transformers"
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    document_prefix: str = ""
    batch_size: int = 32
    normalize: bool = True
    hashing_dim: int = 512
    device: str = "cpu"


class VectorStoreConfig(_Section):
    backend: Literal["qdrant_local", "qdrant", "memory"] = "qdrant_local"
    url: str = "http://localhost:6333"
    collection: str = ""  # default: derived from chunking + embedding config


class BM25Config(_Section):
    implementation: Literal["scratch", "rank_bm25"] = "scratch"
    k1: float = 1.5
    b: float = 0.75
    stem: bool = True
    remove_stopwords: bool = True
    keep_acronym_case: bool = False


class GraphConfig(_Section):
    backend: Literal["networkx", "neo4j"] = "networkx"
    extractor: Literal["rule", "llm", "rule+llm"] = "rule"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    min_edge_confidence: float = 0.3
    fuzzy_link_threshold: float = 0.88
    embedding_link_threshold: float = 0.80
    resolution_similarity: float = 0.92
    max_llm_extraction_chunks: int = 400


class GraphRetrievalConfig(_Section):
    max_hops: int = 2
    hop_decay: float = 0.6
    max_paths: int = 20
    max_expansions: int = 2000
    relation_cue_bonus: float = 1.5
    chunks_per_path: int = 2


class RetrievalConfig(_Section):
    top_k: int = 10
    candidate_pool: int = 50
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    weights: dict[str, float] = Field(
        default_factory=lambda: {"dense": 0.5, "bm25": 0.3, "graph": 0.2}
    )
    normalization: Literal["minmax", "zscore"] = "minmax"
    graph: GraphRetrievalConfig = Field(default_factory=GraphRetrievalConfig)


class RerankerConfig(_Section):
    enabled: bool = False
    provider: Literal["cross_encoder", "lexical", "identity"] = "cross_encoder"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    candidate_pool: int = 50


class RouterConfig(_Section):
    mode: Literal["fixed", "rule"] = "fixed"
    fixed_strategy: Literal["vector", "bm25", "hybrid", "graph", "graph_vector", "multi_hop"] = (
        "vector"
    )
    definition_cues: list[str] = Field(
        default_factory=lambda: [
            "what is",
            "what are",
            "define",
            "definition of",
            "explain",
            "describe",
            "what does",
        ]
    )
    relationship_cues: list[str] = Field(
        default_factory=lambda: [
            "who introduced",
            "who proposed",
            "who authored",
            "who wrote",
            "authored",
            "cites",
            "cite",
            "based on",
            "derived from",
            "introduced",
            "proposed by",
            "which paper",
        ]
    )
    comparison_cues: list[str] = Field(
        default_factory=lambda: [
            "compare",
            "comparison",
            "differ",
            "difference",
            "versus",
            " vs",
            "vs.",
            "contrast",
            "better than",
        ]
    )
    multi_hop_cues: list[str] = Field(
        default_factory=lambda: [
            "which methods",
            "which models",
            "that improved",
            "that use",
            "derived from",
            "built on",
            "papers that",
            "methods that",
            "which papers",
        ]
    )
    experimental_cues: list[str] = Field(
        default_factory=lambda: [
            "evidence",
            "results",
            "ablation",
            "experiment",
            "improvement over",
            "how much",
            "outperform",
            "benchmark",
            "score",
            "accuracy",
        ]
    )
    intent_strategy: dict[str, str] = Field(
        default_factory=lambda: {
            "definition": "vector",
            "relationship": "graph",
            "comparison": "graph_vector",
            "multi_hop": "multi_hop",
            "experimental": "vector",
        }
    )
    rerank_intents: list[str] = Field(default_factory=lambda: ["experimental"])
    experimental_candidate_pool: int = 100
    fallback_strategy: Literal["vector", "hybrid", "graph_vector"] = "hybrid"


class ContextConfig(_Section):
    max_tokens: int = 2500
    ordering: Literal["relevance", "edges"] = "relevance"
    dedup_jaccard: float = 0.8
    mmr_lambda: float | None = None
    max_graph_paths: int = 6


class GenerationConfig(_Section):
    enabled: bool = False
    provider: Literal["extractive", "anthropic", "openai", "ollama", "fake"] = "extractive"
    model: str = ""
    max_tokens: int = 700
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    max_retries: int = 3
    ollama_url: str = "http://localhost:11434"
    prompt_version: str = "grounded_v1"
    cache_responses: bool = True
    # Anthropic: effort is the depth/cost control on current models (temperature is rejected).
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    extraction_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    # Server-side refusal fallback (Claude API only; Opus/Sonnet 5.5-class models).
    anthropic_fallbacks: bool = True


class VerificationConfig(_Section):
    enabled: bool = False
    method: Literal["lexical", "nli", "llm"] = "lexical"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    support_threshold: float = 0.6
    sufficiency_threshold: float = 0.7
    max_iterations: int = 2
    check_contradictions: bool = True


class BudgetConfig(_Section):
    max_retrieval_calls: int = 8
    max_graph_hops: int = 3
    max_candidates: int = 300
    max_tokens: int = 12000
    max_latency_ms: float = 30000.0
    lambda_cost: float = 0.01
    action_costs: dict[str, float] = Field(
        default_factory=lambda: {
            "dense": 1.0,
            "bm25": 0.3,
            "graph_hop": 0.5,
            "rerank_candidate": 0.02,
            "llm_1k_tokens": 2.0,
        }
    )
    # Self-correction policy (pipeline.use_budget): start cheap, escalate on low confidence.
    escalation_ladder: list[str] = Field(
        default_factory=lambda: ["vector", "hybrid", "graph_vector", "multi_hop"]
    )
    min_confidence: float = 0.62


class EvaluationConfig(_Section):
    queries_file: str = "data/evaluation/queries.json"
    ground_truth_file: str = "data/evaluation/ground_truth.json"
    expected_paths_file: str = "data/evaluation/expected_paths.json"
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    primary_k: int = 5
    bootstrap_samples: int = 1000
    confidence: float = 0.95
    split: Literal["dev", "test", "all"] = "dev"
    save_traces: bool = True


class PipelineConfig(_Section):
    name: str = "vector_only"
    description: str = ""
    router: Literal["fixed", "adaptive"] = "fixed"
    strategy: Literal["vector", "bm25", "hybrid", "graph", "graph_vector", "multi_hop"] = "vector"
    rerank: bool = False
    generate: bool = False
    verify: bool = False
    use_budget: bool = False


class ApiConfig(_Section):
    auth_enabled: bool = False
    rate_limit_per_minute: int = 60
    default_pipeline: str = "graph_vector"
    pipelines: list[str] = Field(
        default_factory=lambda: [
            "vector_only",
            "bm25_only",
            "hybrid",
            "hybrid_rerank",
            "graph_only",
            "graph_vector",
            "adaptive",
            "adaptive_tuned",
            "adaptive_budget",
            "adaptive_verified",
        ]
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    retrieval_cache_size: int = 512
    max_upload_mb: int = 40


class LoggingConfig(_Section):
    level: str = "INFO"
    json_logs: bool = True


class Settings(_Section):
    """Fully resolved configuration for one run."""

    seed: int = 42
    env: str = "development"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    root: str = Field(default_factory=lambda: str(PROJECT_ROOT))

    def path(self, name: str) -> Path:
        """Resolve a ``paths.<name>`` entry against the project root."""
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else Path(self.root) / p

    def resolve(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else Path(self.root) / p

    @property
    def index_name(self) -> str:
        """Collection/index identity: changes whenever chunking or embedding changes."""
        c, e = self.chunking, self.embedding
        model = e.model if e.provider == "sentence_transformers" else f"hashing{e.hashing_dim}"
        model_slug = model.split("/")[-1].replace(".", "_").replace("-", "_").lower()
        return f"{c.strategy}_{c.chunk_size}_{c.overlap}__{model_slug}"


class Secrets(BaseSettings):
    """Secrets from env / .env. Never written to results or logs."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    neo4j_password: str | None = None
    api_username: str = "admin"
    api_password: str | None = None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (dicts merge, others replace)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _find_config(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p.resolve()
    candidates = [
        CONFIG_DIR / f"{name_or_path}.yaml",
        CONFIG_DIR / "experiments" / f"{name_or_path}.yaml",
        PROJECT_ROOT / name_or_path,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"config not found: {name_or_path}")


def _load_chain(path: Path, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f"circular 'extends' in {path}")
    seen.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parents = data.pop("extends", None)
    merged: dict[str, Any] = {}
    if parents:
        for parent in [parents] if isinstance(parents, str) else parents:
            parent_path = (path.parent / parent).resolve()
            if not parent_path.exists():
                parent_path = _find_config(parent)
            merged = deep_merge(merged, _load_chain(parent_path, seen))
    return deep_merge(merged, data)


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith("AGVRAG__"):
            continue
        parts = [p.lower() for p in key.split("__")[1:] if p]
        if not parts:
            continue
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(raw)
    return out


def load_settings(
    name_or_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> Settings:
    """Load and validate a configuration.

    ``name_or_path`` may be an experiment name (``hybrid``), an env name (``production``)
    or a YAML path. ``None`` loads ``AGVRAG_CONFIG`` or ``base``.
    """
    environ = dict(os.environ) if environ is None else environ
    name = name_or_path or environ.get("AGVRAG_CONFIG") or "base"
    base_path = CONFIG_DIR / "base.yaml"
    data: dict[str, Any] = {}
    if base_path.exists():
        data = _load_chain(base_path.resolve(), set())
    target = _find_config(name)
    if target != base_path.resolve():
        data = deep_merge(data, _load_chain(target, set()))
    env_layer = environ.get("AGVRAG_ENV")
    if env_layer and env_layer != name:
        env_path = CONFIG_DIR / f"{env_layer}.yaml"
        if env_path.exists():
            data = deep_merge(data, _load_chain(env_path.resolve(), set()))
    data = deep_merge(data, _env_overrides(environ))
    if overrides:
        data = deep_merge(data, overrides)
    return Settings.model_validate(data)


def experiment_overlay(name_or_path: str) -> dict[str, Any]:
    """The overrides an experiment config adds on top of ``base.yaml`` (base itself excluded).

    Used by the API to apply a pipeline definition on top of the *running* deployment's
    settings, so environment choices (backends, embedder, paths) are never lost.
    """
    path = _find_config(name_or_path)
    base = (CONFIG_DIR / "base.yaml").resolve()

    def chain(p: Path) -> dict[str, Any]:
        if p == base:
            return {}
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        parents = data.pop("extends", None) or []
        merged: dict[str, Any] = {}
        for parent in [parents] if isinstance(parents, str) else parents:
            pp = (p.parent / parent).resolve()
            merged = deep_merge(merged, chain(pp if pp.exists() else _find_config(parent)))
        return deep_merge(merged, data)

    return chain(path)


def apply_overlay(settings: Settings, overlay: dict[str, Any]) -> Settings:
    return Settings.model_validate(deep_merge(settings.model_dump(), overlay))


def dump_settings(settings: Settings) -> str:
    return yaml.safe_dump(settings.model_dump(), sort_keys=False)
