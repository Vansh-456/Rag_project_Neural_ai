"""Knowledge-graph construction: extraction -> resolution -> store -> statistics + audit."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import networkx as nx

from agvrag.graph.extraction import LLMExtractor, RuleExtractor
from agvrag.graph.gazetteer import Gazetteer
from agvrag.graph.resolution import EntityResolver
from agvrag.graph.store import NetworkXGraphStore
from agvrag.interfaces import LLMClient
from agvrag.schemas import Chunk, Document, Entity, Relation
from agvrag.utils.logging import get_logger

log = get_logger(__name__)


def build_graph(
    docs: list[Document],
    chunks: list[Chunk],
    gazetteer: Gazetteer,
    extractor: str = "rule",
    llm: LLMClient | None = None,
    min_confidence: float = 0.3,
    max_llm_chunks: int = 400,
    seed: int = 42,
) -> tuple[NetworkXGraphStore, dict[str, Any]]:
    """Build the graph. Returns the store and a report (resolution audit, rejections, usage).

    Only entities that participate in at least one relation are kept, so gazetteer entries
    never mentioned in the corpus do not appear as orphan nodes.
    """
    resolver = EntityResolver(gazetteer)
    titles = {d.doc_id: d.title for d in docs}
    rule = RuleExtractor(gazetteer, resolver, titles)
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    entities: dict[str, Entity] = {}
    relations: list[Relation] = []
    rejected: list[dict[str, Any]] = []
    mention_counts: Counter[str] = Counter()
    for d in docs:
        entities[f"paper:{d.doc_id}"] = rule._paper_entity(d)
    if extractor in {"rule", "rule+llm"}:
        for d in docs:
            res = rule.extract(d, by_doc.get(d.doc_id, []))
            entities.update(res.entities)
            relations.extend(res.relations)
            rejected.extend(res.rejected)
            for cnt in res.mentions.values():
                mention_counts.update(cnt)
    llm_usage: dict[str, Any] = {}
    if extractor in {"llm", "rule+llm"}:
        if llm is None:
            raise ValueError("LLM extraction requested but no LLM client configured")
        llm_ex = LLMExtractor(llm, resolver)
        # Sample chunks deterministically to bound cost; prefer early (abstract/intro) chunks.
        rng = random.Random(seed)
        pool = sorted(chunks, key=lambda c: (c.metadata.get("ordinal", 0), c.chunk_id))
        early = [c for c in pool if c.metadata.get("ordinal", 99) < 6]
        rest = [c for c in pool if c.metadata.get("ordinal", 99) >= 6]
        rng.shuffle(rest)
        selected = (early + rest)[:max_llm_chunks]
        doc_by_id = {d.doc_id: d for d in docs}
        for i, c in enumerate(selected):
            res = llm_ex.extract_chunk(doc_by_id[c.doc_id], c)
            entities.update(res.entities)
            relations.extend(res.relations)
            rejected.extend(res.rejected)
            if (i + 1) % 25 == 0:
                log.info("llm extraction progress", extra={"done": i + 1, "total": len(selected)})
        llm_usage = llm_ex.usage

    for eid in {r.source for r in relations} | {r.target for r in relations}:
        if eid not in entities and eid in resolver.entities:
            entities[eid] = resolver.entities[eid]
    relations = [r for r in relations if r.source in entities and r.target in entities]
    # Canonical entity objects carry aliases learned during resolution.
    for eid, _ent in list(entities.items()):
        if eid in resolver.entities:
            entities[eid] = resolver.entities[eid]
        entities[eid].properties["mentions"] = int(mention_counts.get(eid, 0))

    store = NetworkXGraphStore()
    store.add_entities(list(entities.values()))
    store.add_relations(relations)
    dropped = store.remove_low_confidence(min_confidence)
    orphans = [
        n
        for n in store.g.nodes
        if store.g.degree(n) == 0 and store.g.nodes[n]["entity"].type != "Paper"
    ]
    store.g.remove_nodes_from(orphans)
    report = {
        "resolution_audit": resolver.audit,
        "rejected": rejected,
        "dropped_low_confidence": dropped,
        "removed_orphans": len(orphans),
        "llm_usage": llm_usage,
    }
    return store, report


def graph_statistics(store: NetworkXGraphStore) -> dict[str, Any]:
    g = store.g
    ents = store.entities()
    rels = store.relations()
    und = nx.Graph(g)
    comps = sorted((len(c) for c in nx.connected_components(und)), reverse=True) if len(und) else []
    degrees = sorted((d for _, d in g.degree()), reverse=True)
    grounded = sum(1 for r in rels if r.source_chunk_ids)
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "nodes_by_type": dict(Counter(e.type for e in ents).most_common()),
        "edges_by_type": dict(Counter(r.type for r in rels).most_common()),
        "edges_by_extractor": dict(Counter(r.extractor.split(":")[0] for r in rels).most_common()),
        "pct_edges_chunk_grounded": round(100 * grounded / max(1, len(rels)), 2),
        "pct_edges_traceable": round(
            100
            * sum(1 for r in rels if r.source_chunk_ids or r.properties.get("origin"))
            / max(1, len(rels)),
            2,
        ),
        "degree_max": degrees[0] if degrees else 0,
        "degree_mean": round(sum(degrees) / max(1, len(degrees)), 3),
        "degree_top10": [(n, d) for n, d in sorted(g.degree(), key=lambda t: -t[1])[:10]],
        "connected_components": len(comps),
        "largest_component": comps[0] if comps else 0,
        "orphan_nodes": sum(1 for _, d in g.degree() if d == 0),
        "mean_edge_confidence": round(sum(r.confidence for r in rels) / max(1, len(rels)), 4),
    }


def sample_edges_for_audit(
    store: NetworkXGraphStore, n: int = 50, seed: int = 42, exclude_metadata: bool = True
) -> list[dict[str, Any]]:
    """Random edges for the manual precision audit (annotator fills ``correct``)."""
    rels = [r for r in store.relations() if not (exclude_metadata and r.extractor in {"metadata"})]
    rng = random.Random(seed)
    rng.shuffle(rels)
    out = []
    for r in sorted(rels[:n], key=lambda r: (r.type, r.source)):
        s, t = store.get_entity(r.source), store.get_entity(r.target)
        out.append(
            {
                "source": s.name if s else r.source,
                "relation": r.type,
                "target": t.name if t else r.target,
                "confidence": r.confidence,
                "extractor": r.extractor,
                "evidence": r.evidence_text,
                "chunks": r.source_chunk_ids[:3],
                "correct": None,
                "notes": "",
            }
        )
    return out
