"""Router, context builder, generation/citations, verifier and budget tests."""

import pytest

from agvrag.config import (
    BudgetConfig,
    ContextConfig,
    RetrievalConfig,
    RouterConfig,
    VerificationConfig,
)
from agvrag.context.builder import StructuredContextBuilder, dedup_evidence, edges_order, mmr_select
from agvrag.generation.generator import (
    ABSTAIN,
    ExtractiveGenerator,
    LLMGenerator,
    parse_citations,
    validate_citations,
)
from agvrag.generation.llm import CachedLLMClient, FakeLLMClient, estimate_cost
from agvrag.graph.linking import EntityLinker
from agvrag.graph.store import NetworkXGraphStore
from agvrag.pipeline.budget import BudgetExceeded, BudgetTracker, utility
from agvrag.routing.router import FixedRouter, RuleQueryAnalyzer, RuleRouter
from agvrag.schemas import Entity, Evidence, EvidenceType, QueryAnalysis
from agvrag.verification.verifier import ClaimVerifier, extract_claims


def _linker() -> EntityLinker:
    s = NetworkXGraphStore()
    s.add_entities(
        [
            Entity("method:lora", "LoRA", "Method"),
            Entity("method:adapters", "Adapters", "Method", ["adapter"]),
            Entity("method:dpr", "DPR", "Method", properties={"case_sensitive": True}),
            Entity("dataset:nq", "Natural Questions", "Dataset"),
        ]
    )
    return EntityLinker(s)


@pytest.mark.parametrize(
    "query,intent,strategy",
    [
        ("What is LoRA?", "definition", "vector"),
        ("Who introduced LoRA?", "relationship", "graph"),
        ("How does LoRA differ from adapters?", "comparison", "graph_vector"),
        (
            "Which methods derived from DPR improved results on Natural Questions?",
            "multi_hop",
            "multi_hop",
        ),
        (
            "What evidence supports the improvement of LoRA over adapters in the experiments?",
            "experimental",
            "vector",
        ),
        ("tell me something", "unknown", "hybrid"),
    ],
)
def test_router_taxonomy(query, intent, strategy):
    a = RuleQueryAnalyzer(RouterConfig(), _linker()).analyze(query)
    d = RuleRouter(RouterConfig(), RetrievalConfig()).route(a)
    assert a.intent == intent, a.rationale
    assert d.strategy == strategy, d.rationale
    assert d.rationale and 0.0 <= a.complexity <= 1.0


def test_experimental_route_requests_rerank_and_larger_pool():
    a = RuleQueryAnalyzer(RouterConfig(), _linker()).analyze(
        "What results show LoRA outperforms adapters?"
    )
    d = RuleRouter(RouterConfig(), RetrievalConfig()).route(a)
    assert d.params["rerank"] is True and d.params["candidate_pool"] == 100


def test_fixed_router():
    a = QueryAnalysis("q", "definition", [], 0.1, 0, "")
    assert FixedRouter("bm25", RetrievalConfig()).route(a).strategy == "bm25"


def _text_ev(i: int, text: str, doc: str = "d") -> Evidence:
    return Evidence(
        f"chunk:c{i}",
        EvidenceType.TEXT_CHUNK,
        text,
        [doc],
        [f"c{i}"],
        "dense",
        1.0 / i,
        i,
        provenance={"doc_title": f"Doc {doc}", "section": "S", "page_start": 1, "page_end": 2},
    )


def _path_ev() -> Evidence:
    return Evidence(
        "path:1",
        EvidenceType.GRAPH_FACT,
        "LoRA [Method] <-INTRODUCES- LoRA paper [Paper]",
        ["d"],
        ["c1"],
        "graph",
        0.9,
        1,
        provenance={"nodes": ["method:lora", "paper:d"], "edges": []},
    )


def test_dedup_and_edges_ordering():
    evs = [
        _text_ev(1, "alpha beta gamma delta epsilon"),
        _text_ev(1, "dup chunk id"),
        _text_ev(2, "alpha beta gamma delta epsilon"),
        _text_ev(3, "something else entirely here"),
    ]
    kept, dropped = dedup_evidence(evs, 0.8)
    assert [e.source_chunk_ids[0] for e in kept] == ["c1", "c3"] and len(dropped) == 2
    assert [x for x in edges_order([1, 2, 3, 4, 5])] == [1, 3, 5, 4, 2]


def test_mmr_prefers_diverse():
    evs = [
        _text_ev(1, "lora low rank adaptation matrices"),
        _text_ev(2, "lora low rank adaptation matrices too"),
        _text_ev(3, "graph retrieval of entities"),
    ]
    picked = mmr_select(evs, 0.3, 2)
    assert [e.source_chunk_ids[0] for e in picked] == ["c1", "c3"]


def test_context_labels_budget_and_determinism():
    a = QueryAnalysis("What is LoRA?", "definition", ["LoRA"], 0.1, 0, "")
    evs = [_path_ev()] + [
        _text_ev(i, f"passage number {i} about low rank adaptation " * 5) for i in range(1, 8)
    ]
    b = StructuredContextBuilder(ContextConfig(max_tokens=200))
    ctx = b.build("What is LoRA?", a, evs)
    assert "G1" in ctx.evidence_ids and "E1" in ctx.evidence_ids
    assert ctx.token_count <= 200 + 20 and ctx.dropped  # budget forced drops
    assert ctx.text == b.build("What is LoRA?", a, evs).text
    assert "DETECTED ENTITIES: LoRA" in ctx.text


def _ctx():
    a = QueryAnalysis("What is LoRA?", "definition", ["LoRA"], 0.1, 0, "")
    evs = [
        _text_ev(
            1,
            "LoRA freezes the pretrained weights and injects trainable rank decomposition matrices.",
        ),
        _text_ev(2, "Adapters add bottleneck layers between transformer blocks."),
    ]
    return a, StructuredContextBuilder(ContextConfig()).build("What is LoRA?", a, evs)


def test_citation_parsing_and_validation():
    _a, ctx = _ctx()
    assert parse_citations("A claim here [E1]. Another [E1, E2].")[1][2] == ["E1", "E2"]
    cits, val = validate_citations(
        "LoRA freezes the weights [E1]. It is great for everyone always. Bad [E9].", ctx
    )
    assert val["invalid_ids"] == ["E9"]
    assert val["uncited_claims"] == ["It is great for everyone always."]
    assert val["valid"] is False and cits[0]["evidence_id"] == "chunk:c1"


def test_llm_generator_with_fake_llm_and_cache(tmp_path):
    a, ctx = _ctx()
    fake = FakeLLMClient(
        lambda s, p: "LoRA injects trainable rank decomposition matrices into frozen models [E1]."
    )
    gen = LLMGenerator(CachedLLMClient(fake, tmp_path / "c.sqlite"))
    r1 = gen.generate("What is LoRA?", a, ctx)
    r2 = gen.generate("What is LoRA?", a, ctx)
    assert r1.validation["valid"] and not r1.abstained
    assert len(fake.calls) == 1 and r2.usage["llm_cached"] == 1.0
    assert "[E1]" in r1.answer
    assert "".join(gen.stream("What is LoRA?", a, ctx)) == r1.answer


def test_extractive_generator_answers_and_abstains():
    a, ctx = _ctx()
    r = ExtractiveGenerator().generate("What is LoRA?", a, ctx)
    assert not r.abstained and r.validation["valid"] and "[E1]" in r.answer
    q = QueryAnalysis("capital of France?", "definition", [], 0, 0, "")
    r2 = ExtractiveGenerator().generate("What is the capital of France?", q, ctx)
    assert r2.abstained and r2.answer.startswith(ABSTAIN)


def test_verifier_lexical_labels():
    _a, ctx = _ctx()
    v = ClaimVerifier(
        VerificationConfig(method="lexical", support_threshold=0.6, sufficiency_threshold=0.5)
    )
    res = v.verify(
        "LoRA injects trainable rank decomposition matrices [E1]. "
        "LoRA was invented on the moon by astronauts [E2]. Uncited claim about everything here.",
        [],
        ctx,
    )
    labels = [c["label"] for c in res["claims"]]
    assert labels == ["supported", "unsupported", "unsupported"]
    assert res["support_rate"] == pytest.approx(1 / 3, abs=1e-3) and res["sufficient"] is False
    assert extract_claims(f"{ABSTAIN} nothing found.") == []


def test_verifier_llm_judge():
    _a, ctx = _ctx()
    llm = FakeLLMClient(lambda s, p: '{"label": "contradicted", "confidence": 0.8}')
    v = ClaimVerifier(VerificationConfig(method="llm"), llm)
    assert v.verify("LoRA freezes weights [E1].", [], ctx)["claims"][0]["label"] == "contradicted"


def test_budget_enforced():
    b = BudgetTracker(BudgetConfig(max_retrieval_calls=2, max_graph_hops=1))
    b.charge("dense")
    b.charge("bm25")
    assert not b.can_afford("dense")
    with pytest.raises(BudgetExceeded):
        b.charge("dense")
    with pytest.raises(BudgetExceeded):
        b.charge("graph_hop", 2)
    assert b.retrieval_calls == 2 and b.cost_units == pytest.approx(1.0 + 0.3)
    free = BudgetTracker(BudgetConfig(max_retrieval_calls=1), enforce=False)
    free.charge("dense")
    free.charge("dense")
    assert free.retrieval_calls == 2
    assert utility(0.8, 2.0, 0.1) == pytest.approx(0.6)


def test_cost_estimate():
    assert estimate_cost("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert estimate_cost("unknown-model", 1000, 1000) == 0.0
