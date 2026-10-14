from pathlib import Path

import pytest
from pydantic import ValidationError

from agvrag.config import deep_merge, load_settings


def test_experiment_extends_base_and_overrides():
    s = load_settings("hybrid_rerank", environ={})
    assert s.pipeline.name == "hybrid_rerank"
    assert s.pipeline.rerank is True
    assert s.embedding.model == "BAAI/bge-small-en-v1.5"  # inherited from base


def test_env_overrides_are_yaml_parsed():
    s = load_settings(
        "base", environ={"AGVRAG__RETRIEVAL__TOP_K": "7", "AGVRAG__RERANKER__ENABLED": "false"}
    )
    assert s.retrieval.top_k == 7
    assert s.reranker.enabled is False


def test_env_layer_production_switches_backends():
    s = load_settings("hybrid", environ={"AGVRAG_ENV": "production"})
    assert s.vector_store.backend == "qdrant"
    assert s.graph.backend == "neo4j"
    assert s.pipeline.name == "hybrid"


def test_index_name_tracks_chunking_and_embedding():
    a = load_settings("base", environ={})
    b = load_settings("base", overrides={"chunking": {"chunk_size": 512}}, environ={})
    assert a.index_name != b.index_name


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        load_settings("base", overrides={"retrieval": {"topk": 3}}, environ={})


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}} and base["a"]["b"] == 1


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_settings("does_not_exist", environ={})


def test_path_resolution_is_project_relative():
    s = load_settings("base", environ={})
    assert s.path("raw_dir").is_absolute()
    assert Path(s.root) in s.path("raw_dir").parents
