"""
Knowledge Graph store implementations for relational retrieval.
Includes NetworkX in-memory multigraph and Neo4j driver integration.
"""

from __future__ import annotations

import math
from typing import Any
import networkx as nx
from src.schemas import EntityType, GraphEntity, GraphPath, GraphRelation, RelationType


class NetworkXGraphStore:
    """In-memory property graph store powered by NetworkX."""

    def __init__(self, decay_factor: float = 0.85, min_confidence: float = 0.7):
        self.decay_factor = decay_factor
        self.min_confidence = min_confidence
        self.graph = nx.MultiDiGraph()
        self.entities: dict[str, GraphEntity] = {}
        self.alias_to_id: dict[str, str] = {}

    def add_entity(self, entity: GraphEntity) -> None:
        """Adds or updates an entity node in the graph with alias mapping."""
        self.entities[entity.entity_id] = entity
        self.alias_to_id[entity.name.lower()] = entity.entity_id
        for alias in entity.aliases:
            self.alias_to_id[alias.lower()] = entity.entity_id

        self.graph.add_node(
            entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type.value if isinstance(entity.entity_type, EntityType) else str(entity.entity_type),
            description=entity.description,
            source_doc_ids=entity.source_doc_ids,
        )

    def add_relation(self, relation: GraphRelation) -> None:
        """Adds a directed relation edge between two entities with provenance."""
        if relation.confidence < self.min_confidence:
            return

        rel_type = relation.relation_type.value if isinstance(relation.relation_type, RelationType) else str(relation.relation_type)

        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=rel_type,
            confidence=relation.confidence,
            weight=relation.weight,
            source_doc_ids=relation.source_doc_ids,
            source_chunk_ids=relation.source_chunk_ids,
        )

    def find_entity_by_name(self, name: str) -> GraphEntity | None:
        """Resolves entity by name or alias (case-insensitive)."""
        clean = name.strip().lower()
        if clean in self.alias_to_id:
            return self.entities[self.alias_to_id[clean]]
        # Partial containment check
        for alias, ent_id in self.alias_to_id.items():
            if clean in alias or alias in clean:
                return self.entities[ent_id]
        return None

    def get_neighbors(self, entity_id: str, depth: int = 1) -> list[GraphPath]:
        """Traverses outgoing and incoming edges up to specified depth with hop decay."""
        if entity_id not in self.graph:
            return []

        paths: list[GraphPath] = []
        visited = {entity_id}

        def _traverse(current_id: str, current_nodes: list[GraphEntity], current_rels: list[GraphRelation], current_score: float, current_depth: int):
            if current_depth > depth:
                return

            # Outgoing edges
            for _, target_id, edge_data in self.graph.out_edges(current_id, data=True):
                if target_id in visited:
                    continue
                visited.add(target_id)

                target_entity = self.entities.get(target_id)
                if not target_entity:
                    continue

                rel = GraphRelation(
                    source_id=current_id,
                    target_id=target_id,
                    relation_type=RelationType(edge_data.get("relation_type", RelationType.USES.value)),
                    source_doc_ids=edge_data.get("source_doc_ids", []),
                    source_chunk_ids=edge_data.get("source_chunk_ids", []),
                    confidence=edge_data.get("confidence", 1.0),
                    weight=edge_data.get("weight", 1.0)
                )

                decayed_score = current_score * self.decay_factor * rel.confidence
                new_nodes = current_nodes + [target_entity]
                new_rels = current_rels + [rel]

                # Format human readable path description
                desc = " -> ".join([f"({n.name}: {n.entity_type.value if hasattr(n.entity_type, 'value') else n.entity_type})" for n in new_nodes])
                paths.append(GraphPath(
                    nodes=new_nodes,
                    relations=new_rels,
                    path_score=round(decayed_score, 4),
                    description=desc
                ))

                _traverse(target_id, new_nodes, new_rels, decayed_score, current_depth + 1)
                visited.remove(target_id)

            # Incoming edges
            for source_id, _, edge_data in self.graph.in_edges(current_id, data=True):
                if source_id in visited:
                    continue
                visited.add(source_id)

                src_entity = self.entities.get(source_id)
                if not src_entity:
                    continue

                rel = GraphRelation(
                    source_id=source_id,
                    target_id=current_id,
                    relation_type=RelationType(edge_data.get("relation_type", RelationType.USES.value)),
                    source_doc_ids=edge_data.get("source_doc_ids", []),
                    source_chunk_ids=edge_data.get("source_chunk_ids", []),
                    confidence=edge_data.get("confidence", 1.0),
                    weight=edge_data.get("weight", 1.0)
                )

                decayed_score = current_score * self.decay_factor * rel.confidence
                new_nodes = current_nodes + [src_entity]
                new_rels = current_rels + [rel]

                desc = f"({src_entity.name}) <-[{rel.relation_type.value}]- ({self.entities[current_id].name})"
                paths.append(GraphPath(
                    nodes=new_nodes,
                    relations=new_rels,
                    path_score=round(decayed_score, 4),
                    description=desc
                ))

                _traverse(source_id, new_nodes, new_rels, decayed_score, current_depth + 1)
                visited.remove(source_id)

        root_entity = self.entities.get(entity_id)
        if root_entity:
            _traverse(entity_id, [root_entity], [], 1.0, 1)

        # Sort paths by path score descending
        paths.sort(key=lambda p: p.path_score, reverse=True)
        return paths

    def find_paths_between(self, source_id: str, target_id: str, max_depth: int = 3) -> list[GraphPath]:
        """Finds directed and undirected connecting paths between two entities."""
        if source_id not in self.graph or target_id not in self.graph:
            return []

        try:
            nx_paths = list(nx.all_simple_paths(self.graph.to_undirected(), source=source_id, target=target_id, cutoff=max_depth))
        except Exception:
            return []

        results: list[GraphPath] = []
        for p in nx_paths:
            nodes = [self.entities[nid] for nid in p if nid in self.entities]
            hops = len(nodes) - 1
            path_score = round(1.0 * (self.decay_factor ** hops), 4)
            desc = " <-> ".join([f"{n.name} [{n.entity_type.value if hasattr(n.entity_type, 'value') else n.entity_type}]" for n in nodes])
            results.append(GraphPath(
                nodes=nodes,
                relations=[],
                path_score=path_score,
                description=desc
            ))

        results.sort(key=lambda x: x.path_score, reverse=True)
        return results

    def get_graph_statistics(self) -> dict[str, Any]:
        """Computes comprehensive graph topology metrics."""
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()

        # Count by entity type
        types_count: dict[str, int] = {}
        for ent in self.entities.values():
            t = ent.entity_type.value if isinstance(ent.entity_type, EntityType) else str(ent.entity_type)
            types_count[t] = types_count.get(t, 0) + 1

        # Count by relation type
        rel_count: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            r = data.get("relation_type", "UNKNOWN")
            rel_count[r] = rel_count.get(r, 0) + 1

        degrees = [d for _, d in self.graph.degree()]
        avg_degree = round(sum(degrees) / max(1, len(degrees)), 2)
        isolated_nodes = sum(1 for d in degrees if d == 0)

        # Connected components on undirected projection
        undirected = self.graph.to_undirected()
        num_components = nx.number_connected_components(undirected) if node_count > 0 else 0

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "average_degree": avg_degree,
            "isolated_nodes": isolated_nodes,
            "connected_components": num_components,
            "entity_types": types_count,
            "relation_types": rel_count
        }

    def export_for_visualization(self) -> dict[str, Any]:
        """Returns nodes and links formatted for D3 / Cytoscape / React force-graph."""
        nodes = []
        for ent_id, ent in self.entities.items():
            nodes.append({
                "id": ent_id,
                "label": ent.name,
                "type": ent.entity_type.value if isinstance(ent.entity_type, EntityType) else str(ent.entity_type),
                "description": ent.description
            })

        links = []
        for u, v, data in self.graph.edges(data=True):
            links.append({
                "source": u,
                "target": v,
                "label": data.get("relation_type", "RELATED_TO"),
                "confidence": data.get("confidence", 1.0)
            })

        return {"nodes": nodes, "links": links}


class Neo4jGraphStore:
    """Graph store backed by Neo4j graph database."""

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "adaptive_rag_password"):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: Any = None
        self._fallback_store: NetworkXGraphStore | None = None

    def _get_driver(self):
        if self._driver is None and self._fallback_store is None:
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                driver.verify_connectivity()
                self._driver = driver
            except Exception:
                # Fall back to in-memory graph
                self._fallback_store = NetworkXGraphStore()

    def add_entity(self, entity: GraphEntity) -> None:
        self._get_driver()
        if self._fallback_store is not None:
            self._fallback_store.add_entity(entity)
            return

        with self._driver.session() as session:
            cypher = """
            MERGE (e:Entity {id: $id})
            SET e.name = $name,
                e.type = $type,
                e.description = $description,
                e.aliases = $aliases
            """
            session.run(
                cypher,
                id=entity.entity_id,
                name=entity.name,
                type=entity.entity_type.value,
                description=entity.description,
                aliases=entity.aliases
            )

    def add_relation(self, relation: GraphRelation) -> None:
        self._get_driver()
        if self._fallback_store is not None:
            self._fallback_store.add_relation(relation)
            return

        with self._driver.session() as session:
            rel_type = relation.relation_type.value
            cypher = f"""
            MATCH (s:Entity {{id: $source_id}}), (t:Entity {{id: $target_id}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r.confidence = $confidence,
                r.weight = $weight
            """
            session.run(
                cypher,
                source_id=relation.source_id,
                target_id=relation.target_id,
                confidence=relation.confidence,
                weight=relation.weight
            )

    def get_neighbors(self, entity_id: str, depth: int = 1) -> list[GraphPath]:
        self._get_driver()
        if self._fallback_store is not None:
            return self._fallback_store.get_neighbors(entity_id, depth)
        return []

    def find_paths_between(self, source_id: str, target_id: str, max_depth: int = 3) -> list[GraphPath]:
        self._get_driver()
        if self._fallback_store is not None:
            return self._fallback_store.find_paths_between(source_id, target_id, max_depth)
        return []

    def get_graph_statistics(self) -> dict[str, Any]:
        self._get_driver()
        if self._fallback_store is not None:
            return self._fallback_store.get_graph_statistics()
        return {"backend": "neo4j", "status": "connected"}

    def export_for_visualization(self) -> dict[str, Any]:
        self._get_driver()
        if self._fallback_store is not None:
            return self._fallback_store.export_for_visualization()
        return {"nodes": [], "links": []}


def get_graph_store(backend: str = "networkx", neo4j_uri: str = "bolt://localhost:7687", neo4j_user: str = "neo4j", neo4j_password: str = "adaptive_rag_password") -> NetworkXGraphStore | Neo4jGraphStore:
    """Factory helper for graph stores."""
    if backend == "neo4j":
        return Neo4jGraphStore(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    return NetworkXGraphStore()
