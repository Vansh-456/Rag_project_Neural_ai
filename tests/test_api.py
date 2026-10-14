"""
Integration tests for FastAPI endpoints (/health, /query, /trace, /compare, /graph/visualization).
"""

from fastapi.testclient import TestClient
from src.api.app import app


def test_api_dashboard_endpoint():
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "Adaptive Graph + Vector RAG" in res.text


def test_api_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["chunk_count"] > 0
        assert data["entity_count"] > 0


def test_api_query_endpoint():
    with TestClient(app) as client:
        payload = {
            "query": "What is LoRA?",
            "pipeline": "adaptive",
            "top_k": 3
        }
        res = client.post("/query", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "trace_id" in data
        assert len(data["answer"]) > 0
        assert len(data["evidence_used"]) > 0
        assert "timings_ms" in data["metrics"]

        # Verify trace retrieval endpoint
        trace_res = client.get(f"/trace/{data['trace_id']}")
        assert trace_res.status_code == 200
        trace_data = trace_res.json()
        assert trace_data["trace_id"] == data["trace_id"]


def test_api_compare_endpoint():
    with TestClient(app) as client:
        payload = {
            "query": "What is LoRA?",
            "pipeline_a": "vector",
            "pipeline_b": "hybrid",
            "top_k": 3
        }
        res = client.post("/compare", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "pipeline_a" in data
        assert "pipeline_b" in data
        assert data["pipeline_a"]["name"] == "vector"
        assert data["pipeline_b"]["name"] == "hybrid"


def test_api_graph_visualization():
    with TestClient(app) as client:
        res = client.get("/graph/visualization")
        assert res.status_code == 200
        data = res.json()
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) > 0
        assert len(data["links"]) > 0
