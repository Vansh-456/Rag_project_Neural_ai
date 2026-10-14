"""
Configuration management module for Adaptive Graph + Vector RAG.
Loads base YAML, environment overrides, and experiment overrides with validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    name: str = "Adaptive Graph + Vector RAG"
    environment: str = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000


class CorpusConfig(BaseModel):
    raw_dir: str = "data/raw/papers"
    processed_dir: str = "data/processed"
    documents_file: str = "data/processed/documents/corpus_documents.json"
    chunks_file: str = "data/processed/chunks/corpus_chunks.json"
    graph_file: str = "data/processed/entities/corpus_graph.json"


class ChunkingConfig(BaseModel):
    strategy: str = "structure_aware"
    chunk_size: int = 512
    chunk_overlap: int = 64
    respect_sections: bool = True
    attach_heading_context: bool = True


class EmbeddingsConfig(BaseModel):
    provider: str = "sentence-transformers"
    model_name: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True
    cache_dir: str = "data/cache/embeddings"


class VectorStoreConfig(BaseModel):
    backend: str = "in_memory"
    collection_name: str = "rag_corpus"
    host: str = "localhost"
    port: int = 6333
    distance_metric: str = "cosine"


class SparseRetrievalConfig(BaseModel):
    backend: str = "rank_bm25"
    k1: float = 1.5
    b: float = 0.75
    lowercase: bool = True
    remove_stopwords: bool = True


class GraphConfig(BaseModel):
    backend: str = "networkx"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "adaptive_rag_password"
    max_traversal_depth: int = 3
    decay_factor: float = 0.85
    min_edge_confidence: float = 0.7


class RetrievalConfig(BaseModel):
    default_strategy: str = "adaptive"
    dense_top_k: int = 10
    sparse_top_k: int = 10
    graph_top_k: int = 10
    final_top_k: int = 5
    fusion_method: str = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    graph_weight: float = 0.5


class RerankingConfig(BaseModel):
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_pool_size: int = 20
    batch_size: int = 16


class RoutingConfig(BaseModel):
    strategy: str = "rule_based"
    confidence_threshold: float = 0.75


class GenerationConfig(BaseModel):
    provider: str = "mock"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    enforce_citations: bool = True
    abstain_when_uncertain: bool = True


class VerificationConfig(BaseModel):
    enabled: bool = True
    max_retries: int = 2
    contradiction_threshold: float = 0.8
    min_supported_claims_ratio: float = 0.8


class BudgetConfig(BaseModel):
    max_retrieval_calls: int = 5
    max_graph_hops: int = 3
    max_candidate_pool: int = 50
    max_latency_ms: float = 5000.0
    max_token_budget: int = 4096
    cost_per_1k_prompt_tokens: float = 0.00015
    cost_per_1k_completion_tokens: float = 0.0006


class SystemSettings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    sparse_retrieval: SparseRetrievalConfig = Field(default_factory=SparseRetrievalConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    # Environment variables
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def load_config(
    base_path: str | Path = "configs/base.yaml",
    env_override_path: str | Path | None = None,
    experiment_override_path: str | Path | None = None,
) -> SystemSettings:
    """Loads configuration by merging base YAML, environment YAML, and experiment overrides."""
    base_file = Path(base_path)
    config_dict: dict[str, Any] = {}

    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

    if env_override_path:
        env_file = Path(env_override_path)
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                env_dict = yaml.safe_load(f) or {}
                _deep_merge(config_dict, env_dict)

    if experiment_override_path:
        exp_file = Path(experiment_override_path)
        if exp_file.exists():
            with open(exp_file, "r", encoding="utf-8") as f:
                exp_dict = yaml.safe_load(f) or {}
                _deep_merge(config_dict, exp_dict)

    return SystemSettings(**config_dict)
