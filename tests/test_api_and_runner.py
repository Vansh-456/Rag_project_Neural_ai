"""API (FastAPI TestClient) and experiment-runner tests on the fixture index."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agvrag.api.app import create_app
from agvrag.api.support import LRUCache, RateLimiter
from agvrag.config import Secrets
from agvrag.evaluation.runner import list_experiments, run_experiment

from .conftest import settings_for


@pytest.fixture()
def client(built):
    s = settings_for(built)
    s = s.model_copy(
        update={
            "api": s.api.model_copy(
                update={
                    "pipelines": [
                        "vector_only",
                        "hybrid",
                        "graph_only",
                        "adaptive",
                        "adaptive_verified",
                    ],
                    "default_pipeline": "adaptive",
                }
            )
        }
    )
    app = create_app(s, Secrets(_env_file=None))
    app.state.agv.registry = built[1]
    return TestClient(app)


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["index_built"] and r["graph_built"]


def test_query_trace_roundtrip_and_cache(client):
    r = client.post("/query", json={"query": "Who introduced LoRA?", "pipeline": "adaptive"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["routing"]["strategy"] == "graph" and body["answer"]
    assert body["cached"] is False
    again = client.post(
        "/query", json={"query": "Who introduced LoRA?", "pipeline": "adaptive"}
    ).json()
    assert again["cached"] is True
    t = client.get(f"/trace/{body['trace_id']}").json()
    assert t["trace_id"] == body["trace_id"]
    assert client.get("/traces").json()[0]["trace_id"] == body["trace_id"]
    assert client.get("/trace/nope").status_code == 404


def test_compare_and_pipelines(client):
    r = client.post(
        "/compare", json={"query": "What is LoRA?", "pipelines": ["vector_only", "hybrid"]}
    ).json()
    assert set(r["results"]) == {"vector_only", "hybrid"}
    names = [p["name"] for p in client.get("/pipelines").json()]
    assert "adaptive" in names
    assert client.post("/query", json={"query": "x y", "pipeline": "nope"}).status_code == 404


def test_stream_sse(client):
    with client.stream(
        "POST", "/query/stream", json={"query": "What is LoRA?", "pipeline": "vector_only"}
    ) as r:
        text = "".join(r.iter_text())
    assert "event: analysis" in text and "event: done" in text


def test_graph_and_chunk_endpoints(client):
    hits = client.get("/graph/search", params={"q": "LoRA"}).json()
    assert hits and hits[0]["entity_id"] == "method:lora"
    nb = client.get("/graph/neighborhood/method:lora").json()
    assert any(e["type"] == "INTRODUCES" for e in nb["edges"])
    trace = client.post("/query", json={"query": "What is LoRA?", "pipeline": "vector_only"}).json()
    cid = trace["evidence_pool"][0]["source_chunk_ids"][0]
    assert client.get(f"/chunks/{cid}").json()["chunk_id"] == cid
    assert client.get("/graph/stats").status_code == 200


def test_metrics_endpoint(client):
    client.get("/health")
    m = client.get("/metrics").json()
    assert "requests" in m and "cache" in m


def test_auth_required_when_enabled(built):
    s = settings_for(built)
    s = s.model_copy(update={"api": s.api.model_copy(update={"auth_enabled": True})})
    with pytest.raises(RuntimeError):
        create_app(s, Secrets(_env_file=None, api_password=None))
    app = create_app(s, Secrets(_env_file=None, api_password="pw"))
    c = TestClient(app)
    assert c.get("/pipelines").status_code == 401
    assert c.get("/pipelines", auth=("admin", "wrong")).status_code == 401
    assert c.get("/pipelines", auth=("admin", "pw")).status_code == 200
    assert c.get("/health").status_code == 200  # health stays open for probes


def test_rate_limiter_and_lru():
    rl = RateLimiter(2, window=10)
    assert rl.allow("a", 0)[0] and rl.allow("a", 1)[0]
    ok, retry = rl.allow("a", 2)
    assert not ok and retry > 0
    assert rl.allow("a", 11)[0]
    c = LRUCache(2)
    c.put(1, "a"), c.put(2, "b"), c.get(1), c.put(3, "c")
    assert c.get(2) is None and c.get(1) == "a" and c.get(3) == "c"


def test_run_experiment_writes_registry(built):
    s = settings_for(built, name="hybrid", strategy="hybrid", generate=True)
    res = run_experiment(s, "test", "all", built[1])
    d = Path(res["dir"])
    for f in (
        "config.yaml",
        "git_commit.txt",
        "metrics.json",
        "per_query.csv",
        "failures.json",
        "traces.jsonl",
    ):
        assert (d / f).exists(), f
    m = json.loads((d / "metrics.json").read_text())
    assert m["n_queries"] == 6 and m["label_coverage"]["n_unmapped"] == 0
    assert "ndcg@5" in m["overall"] and "ci_low" in m["overall"]["ndcg@5"]
    assert set(m["per_type"]) >= {
        "definition",
        "relationship",
        "comparison",
        "multi_hop",
        "experimental",
    }
    assert m["unanswerable"]["abstention_correct"]["mean"] == 1.0
    assert list_experiments(s.path("results_dir"))[0]["exp_id"] == res["exp_id"]


def test_path_metrics_recorded_for_graph_queries(built):
    s = settings_for(built, name="graph_only", strategy="graph")
    res = run_experiment(s, "test", "dev", built[1], save_traces=False)
    row = next(r for r in res["rows"] if r["query_id"] == "t2")
    assert row["path_entity_recall"] == 1.0 and row["path_relation_recall"] == 1.0
