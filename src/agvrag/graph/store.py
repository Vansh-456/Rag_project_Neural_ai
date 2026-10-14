"""Graph stores.

* ``NetworkXGraphStore`` — in-memory ``MultiDiGraph``; JSON persistence. Used for tests and
  as the default single-process backend.
* ``Neo4jGraphStore`` — Neo4j via the official driver, with a uniqueness constraint on
  ``Entity.id``. Every node gets the ``:Entity`` label plus its type label; relationship types
  are the relation names. Edge provenance (confidence, chunk ids, evidence) lives on the
  relationship. ``neighbors`` results are memoised because traversal calls it repeatedly.

Duplicate relations (same source, type, target) are merged: chunk ids are unioned and
confidence becomes ``1 - prod(1 - c_i)`` (noisy-OR over independent supporting mentions).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx

from agvrag.schemas import RELATION_TYPES, Entity, Relation
from agvrag.utils.io import read_json, write_json


def merge_relation(a: Relation, b: Relation, max_chunks: int = 8) -> Relation:
    conf = 1 - (1 - a.confidence) * (1 - b.confidence)
    chunks = list(dict.fromkeys(a.source_chunk_ids + b.source_chunk_ids))[:max_chunks]
    extractors = sorted(set(a.extractor.split("+")) | set(b.extractor.split("+")) - {""})
    support = int(a.properties.get("support", 1)) + int(b.properties.get("support", 1))
    return Relation(
        source=a.source,
        target=a.target,
        type=a.type,
        confidence=round(min(conf, 0.99), 4),
        source_chunk_ids=chunks,
        evidence_text=a.evidence_text or b.evidence_text,
        extractor="+".join(extractors),
        properties={**b.properties, **a.properties, "support": support},
    )


class NetworkXGraphStore:
    def __init__(self) -> None:
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_entities(self, entities: Sequence[Entity]) -> None:
        for e in entities:
            if self.g.has_node(e.entity_id):
                cur: Entity = self.g.nodes[e.entity_id]["entity"]
                cur.aliases = list(dict.fromkeys(cur.aliases + e.aliases))
                cur.properties = {**e.properties, **cur.properties}
            else:
                self.g.add_node(e.entity_id, entity=e)

    def add_relations(self, relations: Sequence[Relation]) -> None:
        for r in relations:
            if r.type not in RELATION_TYPES:
                raise ValueError(f"unknown relation type {r.type}")
            for node in (r.source, r.target):
                if not self.g.has_node(node):
                    raise KeyError(f"relation endpoint {node} is not an entity")
            if self.g.has_edge(r.source, r.target, key=r.type):
                old: Relation = self.g.edges[r.source, r.target, r.type]["relation"]
                self.g.edges[r.source, r.target, r.type]["relation"] = merge_relation(old, r)
            else:
                self.g.add_edge(r.source, r.target, key=r.type, relation=r)

    def get_entity(self, entity_id: str) -> Entity | None:
        if not self.g.has_node(entity_id):
            return None
        ent: Entity = self.g.nodes[entity_id]["entity"]
        return ent

    def entities(self) -> list[Entity]:
        return [d["entity"] for _, d in self.g.nodes(data=True)]

    def relations(self) -> list[Relation]:
        return [d["relation"] for _, _, d in self.g.edges(data=True)]

    def neighbors(
        self, entity_id: str, relation_types: Sequence[str] | None = None
    ) -> list[tuple[Relation, str, str]]:
        if not self.g.has_node(entity_id):
            return []
        allowed = set(relation_types) if relation_types else None
        out: list[tuple[Relation, str, str]] = []
        for _, tgt, d in self.g.out_edges(entity_id, data=True):
            r: Relation = d["relation"]
            if allowed is None or r.type in allowed:
                out.append((r, tgt, "out"))
        for src, _, d in self.g.in_edges(entity_id, data=True):
            r = d["relation"]
            if allowed is None or r.type in allowed:
                out.append((r, src, "in"))
        out.sort(key=lambda t: (-t[0].confidence, t[1], t[0].type))
        return out

    def degree(self, entity_id: str) -> int:
        return int(self.g.degree(entity_id)) if self.g.has_node(entity_id) else 0

    def remove_low_confidence(self, min_conf: float) -> int:
        drop = [
            (u, v, k)
            for u, v, k, d in self.g.edges(keys=True, data=True)
            if d["relation"].confidence < min_conf
        ]
        self.g.remove_edges_from(drop)
        return len(drop)

    def save(self, path: Path) -> None:
        write_json(
            path,
            {
                "entities": [e.to_dict() for e in self.entities()],
                "relations": [r.to_dict() for r in self.relations()],
            },
            indent=None,
        )

    @classmethod
    def load(cls, path: Path) -> NetworkXGraphStore:
        data = read_json(path)
        store = cls()
        store.add_entities([Entity.from_dict(e) for e in data["entities"]])
        store.add_relations([Relation.from_dict(r) for r in data["relations"]])
        return store


class Neo4jGraphStore:
    """Neo4j-backed store. Requires the ``neo4j`` extra and a running server."""

    def __init__(self, uri: str, user: str, password: str, database: str | None = None) -> None:
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.driver.verify_connectivity()
        self._run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        self._run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
        self._cached_neighbors = lru_cache(maxsize=20000)(self._neighbors_uncached)

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as s:
            return [dict(r) for r in s.run(query, **params)]

    def close(self) -> None:
        self.driver.close()

    def clear(self) -> None:
        self._run("MATCH (n:Entity) DETACH DELETE n")
        self._cached_neighbors.cache_clear()

    def add_entities(self, entities: Sequence[Entity]) -> None:
        by_type: dict[str, list[dict[str, Any]]] = {}
        for e in entities:
            by_type.setdefault(e.type, []).append(
                {
                    "id": e.entity_id,
                    "name": e.name,
                    "aliases": e.aliases,
                    "props": {
                        k: v
                        for k, v in e.properties.items()
                        if isinstance(v, str | int | float | bool)
                    },
                }
            )
        for etype, rows in by_type.items():
            # Labels cannot be parameterised; etype is validated against ENTITY_TYPES upstream.
            self._run(
                f"UNWIND $rows AS row MERGE (e:Entity {{id: row.id}}) SET e:{etype}, "
                "e.name = row.name, e.aliases = row.aliases, e.type = $etype, e += row.props",
                rows=rows,
                etype=etype,
            )
        self._cached_neighbors.cache_clear()

    def add_relations(self, relations: Sequence[Relation]) -> None:
        by_type: dict[str, list[dict[str, Any]]] = {}
        for r in relations:
            if r.type not in RELATION_TYPES:
                raise ValueError(f"unknown relation type {r.type}")
            by_type.setdefault(r.type, []).append(
                {
                    "s": r.source,
                    "t": r.target,
                    "confidence": r.confidence,
                    "chunks": r.source_chunk_ids,
                    "evidence": r.evidence_text,
                    "extractor": r.extractor,
                }
            )
        for rtype, rows in by_type.items():
            self._run(
                f"UNWIND $rows AS row MATCH (a:Entity {{id: row.s}}), (b:Entity {{id: row.t}}) "
                f"MERGE (a)-[r:{rtype}]->(b) "
                "ON CREATE SET r.confidence = row.confidence, r.chunks = row.chunks, "
                "  r.evidence = row.evidence, r.extractor = row.extractor, r.support = 1 "
                "ON MATCH SET r.confidence = 1 - (1 - r.confidence) * (1 - row.confidence), "
                "  r.chunks = r.chunks + [c IN row.chunks WHERE NOT c IN r.chunks], "
                "  r.support = r.support + 1",
                rows=rows,
            )
        self._cached_neighbors.cache_clear()

    @staticmethod
    def _entity(rec: dict[str, Any]) -> Entity:
        props = dict(rec)
        return Entity(
            entity_id=props.pop("id"),
            name=props.pop("name", ""),
            type=props.pop("type", "Concept"),
            aliases=list(props.pop("aliases", []) or []),
            properties=props,
        )

    def get_entity(self, entity_id: str) -> Entity | None:
        rows = self._run("MATCH (e:Entity {id: $id}) RETURN properties(e) AS p", id=entity_id)
        return self._entity(rows[0]["p"]) if rows else None

    def entities(self) -> list[Entity]:
        return [
            self._entity(r["p"]) for r in self._run("MATCH (e:Entity) RETURN properties(e) AS p")
        ]

    @staticmethod
    def _relation(s: str, t: str, rtype: str, p: dict[str, Any]) -> Relation:
        return Relation(
            source=s,
            target=t,
            type=rtype,
            confidence=float(p.get("confidence", 0.5)),
            source_chunk_ids=list(p.get("chunks", []) or []),
            evidence_text=p.get("evidence", "") or "",
            extractor=p.get("extractor", "") or "",
            properties={"support": p.get("support", 1)},
        )

    def relations(self) -> list[Relation]:
        rows = self._run(
            "MATCH (a:Entity)-[r]->(b:Entity) RETURN a.id AS s, b.id AS t, "
            "type(r) AS rt, properties(r) AS p"
        )
        return [self._relation(r["s"], r["t"], r["rt"], r["p"]) for r in rows]

    def _neighbors_uncached(
        self, entity_id: str, rel_key: tuple[str, ...] | None
    ) -> tuple[Any, ...]:
        rows = self._run(
            "MATCH (a:Entity {id: $id})-[r]-(b:Entity) "
            "WHERE $types IS NULL OR type(r) IN $types "
            "RETURN startNode(r).id AS s, endNode(r).id AS t, b.id AS other, type(r) AS rt, "
            "properties(r) AS p",
            id=entity_id,
            types=list(rel_key) if rel_key else None,
        )
        out = [
            (
                self._relation(r["s"], r["t"], r["rt"], r["p"]),
                r["other"],
                "out" if r["s"] == entity_id else "in",
            )
            for r in rows
        ]
        out.sort(key=lambda t: (-t[0].confidence, t[1], t[0].type))
        return tuple(out)

    def neighbors(
        self, entity_id: str, relation_types: Sequence[str] | None = None
    ) -> list[tuple[Relation, str, str]]:
        key = tuple(sorted(relation_types)) if relation_types else None
        return list(self._cached_neighbors(entity_id, key))

    def cypher(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Run an arbitrary read query (used by the API's graph explorer)."""
        return self._run(query, **params)
