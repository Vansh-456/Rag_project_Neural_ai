"""End-to-end pipeline tests on the fixture index (offline)."""

import pytest

from agvrag.pipeline.pipeline import RAGPipeline, _next_strategy, _reformulate
from agvrag.schemas import EvidenceType, PipelineTrace, QueryAnalysis

from .conftest import settings_for


@pytest.mark.parametrize(
    "strategy", ["vector", "bm25", "hybrid", "graph", "graph_vector", "multi_hop"]
)
def test_every_strategy_runs_and_traces(built, strategy):
    s = settings_for(built, name=strategy, strategy=strategy)
    trace = RAGPipeline(s, built[1]).run("How does LoRA differ from adapters?")
    assert trace.routing.strategy == strategy
    assert trace.costs["latency_ms.total"] > 0
    if strategy != "graph":
        assert trace.ranked_chunk_ids(), "no chunks retrieved"
    assert PipelineTrace.from_dict(trace.to_dict()).to_dict() == trace.to_dict()  # lossless


def test_dense_finds_lora_definition(built):
    s = settings_for(built, name="v", strategy="hybrid")
    trace = RAGPipeline(s, built[1]).run("trainable rank decomposition matrices")
    top = built[1].index(s).chunks[trace.ranked_chunk_ids()[0]]
    assert top.doc_id == "2106.09685"


def test_graph_answers_relationship_with_grounded_path(built):
    s = settings_for(built, name="g", strategy="graph")
    trace = RAGPipeline(s, built[1]).run("Who introduced LoRA?")
    paths = [e for e in trace.evidence_pool if e.type != EvidenceType.TEXT_CHUNK]
    assert paths
    assert any(
        "INTRODUCES" in p.content and "paper:2106.09685" in p.provenance["nodes"] for p in paths
    )
    assert all(p.source_chunk_ids for p in paths)


def test_adaptive_routes_by_intent(built):
    s = settings_for(built, name="adaptive", router="adaptive", strategy="hybrid")
    pipe = RAGPipeline(s, built[1])
    assert pipe.run("What is LoRA?").routing.strategy == "vector"
    assert pipe.run("Who introduced LoRA?").routing.strategy == "graph"
    t = pipe.run("What results show DPR outperforms BM25 on Natural Questions?")
    assert t.reranked and t.reranked[0].rerank_score is not None


def test_rerank_pipeline_records_before_after(built):
    s = settings_for(built, name="hr", strategy="hybrid", rerank=True)
    t = RAGPipeline(s, built[1]).run("DPR outperforms BM25")
    assert t.reranked and all("pre_rerank_rank" in e.provenance for e in t.reranked)
    assert "latency_ms.rerank" in t.costs


def test_generation_and_bounded_verification_loop(built):
    s = settings_for(
        built, name="av", router="adaptive", strategy="hybrid", generate=True, verify=True
    )
    s = s.model_copy(
        update={
            "verification": s.verification.model_copy(
                update={"sufficiency_threshold": 1.01, "max_iterations": 2}
            )
        }
    )  # never sufficient -> must stop
    t = RAGPipeline(s, built[1]).run("How does LoRA differ from adapters?")
    assert t.answer is not None and t.verification is not None
    assert len(t.iterations) <= 1 + 2  # initial + max_iterations
    assert t.verification["iterations"] == len(t.iterations) - 1
    assert "citation_validation" in t.verification


def test_unanswerable_query_abstains(built):
    s = settings_for(built, name="v", strategy="vector", generate=True)
    t = RAGPipeline(s, built[1]).run("What is the capital of France?")
    assert t.answer is not None and t.answer.startswith("INSUFFICIENT_EVIDENCE")


def test_budget_pipeline_escalates_and_respects_limits(built):
    s = settings_for(built, name="ab", router="adaptive", strategy="hybrid", use_budget=True)
    s = s.model_copy(
        update={
            "budget": s.budget.model_copy(update={"min_confidence": 2.0, "max_retrieval_calls": 3})
        }
    )
    t = RAGPipeline(s, built[1]).run("What is LoRA?")
    steps = [i.get("step") for i in t.iterations]
    assert steps[0] == "vector" and len(steps) >= 2  # escalated because confidence never suffices
    assert t.costs["budget.retrieval_calls"] <= 3


def test_graph_citations_name_their_source_paper(built):
    s = settings_for(built, name="g", strategy="graph")
    t = RAGPipeline(s, built[1]).run("Who introduced LoRA?", generate=True)
    graph_cites = [c for c in t.citations if c["type"] != "text_chunk"]
    assert graph_cites and all(c["doc_title"] for c in graph_cites)


def test_streaming_events(built):
    s = settings_for(built, name="v", strategy="vector", generate=True)
    events = [e["event"] for e in RAGPipeline(s, built[1]).stream("What is LoRA?")]
    assert events[:3] == ["analysis", "routing", "evidence"] and events[-1] == "done"
    assert "token" in events


def test_helpers():
    ladder = ["vector", "hybrid", "graph_vector", "multi_hop"]
    assert _next_strategy("vector", ladder, True) == "hybrid"
    assert _next_strategy("hybrid", ladder, False) is None
    a = QueryAnalysis(
        "what is it",
        "definition",
        ["LoRA"],
        0,
        0,
        "",
        linked_entities=[{"name": "LoRA", "entity_id": "method:lora"}],
    )
    assert "LoRA" in _reformulate("what is it", a, ["Adapters introduce inference latency"])
