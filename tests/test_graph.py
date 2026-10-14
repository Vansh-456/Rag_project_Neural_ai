"""Gazetteer, resolution, store, traversal, linking and extraction tests."""

import pytest

from agvrag.generation.llm import FakeLLMClient
from agvrag.graph.extraction import LLMExtractor, RuleExtractor, parse_json_object
from agvrag.graph.gazetteer import GazEntry, Gazetteer
from agvrag.graph.linking import EntityLinker
from agvrag.graph.resolution import EntityResolver, acronym_matches, mine_acronyms
from agvrag.graph.store import NetworkXGraphStore
from agvrag.graph.traversal import GraphTraverser, relation_cues_for, serialize_path
from agvrag.schemas import Chunk, Entity, Relation

from .conftest import GAZETTEER


@pytest.fixture()
def gaz() -> Gazetteer:
    return Gazetteer(
        [
            GazEntry(
                e["name"],
                e["type"],
                e.get("aliases", []),
                e.get("introduced_in"),
                e.get("case_sensitive", False),
            )
            for e in GAZETTEER["entities"]
        ]
    )


def test_gazetteer_boundaries_and_case(gaz):
    ms = gaz.find_mentions(
        "Self-RAG improves on RAG, unlike rag tags. Low-Rank Adaptation (LoRA) helps."
    )
    ids = [m.entity_id for m in ms]
    assert ids.count("method:rag") == 1  # not inside Self-RAG, not lowercase 'rag'
    assert "method:self rag" in ids
    assert ids.count("method:lora") == 2  # alias + acronym


def test_gazetteer_rejects_unknown_type():
    with pytest.raises(ValueError):
        Gazetteer([GazEntry("X", "Thing")])


def test_acronyms():
    assert acronym_matches("LoRA", "Low-Rank Adaptation")
    assert not acronym_matches("BERT", "Low-Rank Adaptation")
    assert ("Dense Passage Retrieval", "DPR") in mine_acronyms(
        "we propose Dense Passage Retrieval (DPR) for QA"
    )


def test_resolver_alias_plural_acronym_fuzzy(gaz):
    r = EntityResolver(gaz)
    assert r.resolve("adapter modules", "Method") == "method:adapters"
    assert r.resolve("Adapter", "Method") == "method:adapters"
    assert r.resolve("Low Rank Adaptation", "Method") == "method:lora"
    new = r.resolve("Prefix Tuning", "Method")
    assert new == "method:prefix tuning"
    assert r.resolve("Prefix-Tuning", "Method") == new  # punctuation-insensitive
    assert r.resolve("Prefix Tunings", "Method") == new  # plural


def test_resolver_llm_adjudication(gaz):
    calls = []

    def adj(a: str, b: str, t: str) -> bool:
        calls.append((a, b))
        return True

    r = EntityResolver(gaz, merge_threshold=0.99, ambiguous_low=0.7, adjudicator=adj)
    eid = r.resolve("Natural Quest", "Dataset")
    assert eid == "dataset:natural questions"
    assert calls and r.audit[-1]["method"] in {"llm_adjudication"}


def _store() -> NetworkXGraphStore:
    s = NetworkXGraphStore()
    s.add_entities(
        [
            Entity("paper:1", "LoRA paper", "Paper", properties={"year": 2021}),
            Entity("method:lora", "LoRA", "Method"),
            Entity("author:hu", "Edward Hu", "Author"),
            Entity("dataset:glue", "GLUE", "Dataset"),
            Entity("method:adapters", "Adapters", "Method"),
        ]
    )
    s.add_relations(
        [
            Relation("paper:1", "method:lora", "INTRODUCES", 0.9, ["c1"]),
            Relation("paper:1", "author:hu", "AUTHORED_BY", 0.99, ["c0"]),
            Relation("method:lora", "dataset:glue", "EVALUATES_ON", 0.6, ["c2"]),
            Relation("method:lora", "method:adapters", "IMPROVES", 0.5, ["c3"]),
        ]
    )
    return s


def test_store_merges_duplicate_relations():
    s = _store()
    s.add_relations([Relation("method:lora", "dataset:glue", "EVALUATES_ON", 0.5, ["c9"])])
    r = next(r for r in s.relations() if r.type == "EVALUATES_ON")
    assert r.confidence == pytest.approx(1 - 0.4 * 0.5)
    assert r.source_chunk_ids == ["c2", "c9"]
    with pytest.raises(KeyError):
        s.add_relations([Relation("method:lora", "missing", "USES", 0.5, [])])
    with pytest.raises(ValueError):
        s.add_relations([Relation("method:lora", "dataset:glue", "LIKES", 0.5, [])])


def test_store_roundtrip(tmp_path):
    s = _store()
    s.save(tmp_path / "g.json")
    s2 = NetworkXGraphStore.load(tmp_path / "g.json")
    assert len(s2.relations()) == len(s.relations()) and len(s2.entities()) == len(s.entities())


def test_traversal_scoring_decay_and_cues():
    s = _store()
    t = GraphTraverser(s, hop_decay=0.5, cue_bonus=2.0)
    paths = t.paths({"method:lora": 1.0}, max_hops=2, cue_relations=["AUTHORED_BY", "INTRODUCES"])
    by_nodes = {tuple(p.nodes): p for p in paths}
    one = by_nodes[("method:lora", "paper:1")]
    assert one.score == pytest.approx(0.9 * 2.0)
    two = by_nodes[("method:lora", "paper:1", "author:hu")]
    assert two.score == pytest.approx(0.9 * 2.0 * 0.99 * 2.0 * 0.5)
    assert all(len(set(p.nodes)) == len(p.nodes) for p in paths)  # no cycles
    text = serialize_path(two, s)
    assert "<-INTRODUCES-" in text and "-AUTHORED_BY->" in text and "[Paper, 2021]" in text


def test_traversal_respects_expansion_budget():
    t = GraphTraverser(_store(), max_expansions=1)
    t.paths({"method:lora": 1.0}, max_hops=3)
    assert t.last_expansions <= 3  # stops expanding once the budget is hit


def test_connecting_paths_get_bonus():
    s = _store()
    paths = GraphTraverser(s).paths({"method:lora": 1.0, "method:adapters": 1.0}, max_hops=1)
    top = paths[0]
    assert set(top.nodes) == {"method:lora", "method:adapters"}


def test_relation_cues():
    assert "INTRODUCES" in relation_cues_for("Who introduced LoRA?")
    assert "EVALUATES_ON" in relation_cues_for("Which datasets was it evaluated on?")


def test_linker_exact_alias_fuzzy_and_case():
    s = _store()
    s.add_entities(
        [
            Entity(
                "method:rag",
                "RAG",
                "Method",
                ["Retrieval-Augmented Generation"],
                {"case_sensitive": True},
            )
        ]
    )
    linker = EntityLinker(s)
    names = [e["name"] for e in linker.link("How does LoRA differ from Adapters?")]
    assert names == ["LoRA", "Adapters"]
    assert [e["name"] for e in linker.link("what is retrieval-augmented generation")] == ["RAG"]
    assert linker.link("put the rag away") == []
    fuzzy = linker.link("results of Adapterss")
    assert fuzzy and fuzzy[0]["name"] == "Adapters"


def test_rule_extractor_grounds_edges(corpus, gaz):
    titles = {d.doc_id: d.title for d in corpus}
    from agvrag.chunking.chunkers import build_chunker
    from agvrag.config import ChunkingConfig

    ch = build_chunker(ChunkingConfig(chunk_size=64, overlap=8))
    lora = corpus[0]
    res = RuleExtractor(gaz, EntityResolver(gaz), titles).extract(lora, ch.chunk(lora))
    rels = {(r.source, r.type, r.target): r for r in res.relations}
    intro = rels[("paper:2106.09685", "INTRODUCES", "method:lora")]
    assert intro.source_chunk_ids and "Low-Rank Adaptation" in intro.evidence_text
    assert ("method:lora", "IMPROVES", "method:adapters") in rels
    assert ("paper:2106.09685", "CITES", "paper:1902.00751") in rels
    assert ("paper:2106.09685", "AUTHORED_BY", "author:edward hu") in rels
    for r in res.relations:
        assert r.source_chunk_ids or r.properties.get("origin"), f"untraceable edge {r}"


def test_llm_extractor_validates_and_rejects_hallucinated_evidence(gaz):
    chunk = Chunk(
        "c1", "2106.09685", "LoRA outperforms adapter layers on GLUE.", "Intro", 1, 1, "fixed", 9
    )
    from agvrag.schemas import Document

    doc = Document("2106.09685", "LoRA", [], 2021, [], [], "x")
    payload = (
        '```json\n{"entities": [{"name": "LoRA", "type": "Method"}], "relations": ['
        '{"source": "LoRA", "source_type": "Method", "relation": "IMPROVES", "target": "adapter layers",'
        ' "target_type": "Method", "evidence": "LoRA outperforms adapter layers"},'
        '{"source": "LoRA", "source_type": "Method", "relation": "EVALUATES_ON", "target": "SQuAD",'
        ' "target_type": "Dataset", "evidence": "evaluated on SQuAD with great results"}]}\n```'
    )
    ex = LLMExtractor(FakeLLMClient(lambda s, p: payload), EntityResolver(gaz))
    res = ex.extract_chunk(doc, chunk)
    assert [(r.source, r.type, r.target) for r in res.relations] == [
        ("method:lora", "IMPROVES", "method:adapters")
    ]
    assert any(x["reason"] == "evidence not in text" for x in res.rejected)
    bad = LLMExtractor(FakeLLMClient(lambda s, p: "not json"), EntityResolver(gaz)).extract_chunk(
        doc, chunk
    )
    assert not bad.relations and bad.rejected


def test_parse_json_object():
    assert parse_json_object('text {"a": 1} more') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no json here")
