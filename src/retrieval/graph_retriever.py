"""
Graph retrieval module for structured relational evidence extraction.
Traverses neighborhoods and multi-hop paths and serializes them into textual evidence.
"""

from __future__ import annotations

from src.schemas import Evidence, EvidenceType, RetrievalBudget
from src.graph.store import NetworkXGraphStore, Neo4jGraphStore
from src.graph.extraction import EntityLinker


class GraphRetriever:
    """Retrieves relational evidence paths from the knowledge graph."""

    def __init__(self, graph_store: NetworkXGraphStore | Neo4jGraphStore, entity_linker: EntityLinker | None = None, max_depth: int = 2):
        self.graph_store = graph_store
        self.entity_linker = entity_linker or EntityLinker()
        self.max_depth = max_depth

    def retrieve(self, query: str, top_k: int = 5, budget: RetrievalBudget | None = None) -> list[Evidence]:
        """Links entities in query, extracts graph paths, and formats them into Evidence objects."""
        if budget and not budget.can_traverse_graph():
            return []

        entity_ids = self.entity_linker.link_query(query)
        if not entity_ids:
            return []

        all_paths = []
        hops_taken = 0

        # If two or more entities linked, find connecting paths between them
        if len(entity_ids) >= 2:
            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    paths = self.graph_store.find_paths_between(entity_ids[i], entity_ids[j], max_depth=self.max_depth)
                    all_paths.extend(paths)
                    hops_taken += 1

        # Also retrieve 1-hop / 2-hop neighborhoods for each detected entity
        for ent_id in entity_ids:
            paths = self.graph_store.get_neighbors(ent_id, depth=min(self.max_depth, 2))
            all_paths.extend(paths)
            hops_taken += 1

        all_paths.sort(key=lambda p: p.path_score, reverse=True)

        # Convert to Evidence objects
        evidence_list: list[Evidence] = []
        seen_descriptions = set()

        for rank, path in enumerate(all_paths, start=1):
            if path.description in seen_descriptions:
                continue
            seen_descriptions.add(path.description)

            # Collect source document and chunk IDs from entities and relations
            doc_ids = []
            chunk_ids = []
            for n in path.nodes:
                doc_ids.extend(n.source_doc_ids)
            for r in path.relations:
                doc_ids.extend(r.source_doc_ids)
                chunk_ids.extend(r.source_chunk_ids)

            evidence_list.append(Evidence(
                evidence_id=f"ev_graph_{rank:03d}",
                type=EvidenceType.GRAPH_PATH,
                content=f"Knowledge Graph Relation Path: {path.description}",
                source_doc_ids=list(set(doc_ids)),
                source_chunk_ids=list(set(chunk_ids)),
                retriever="graph",
                raw_score=path.path_score,
                rank=rank,
                provenance={
                    "nodes": [n.name for n in path.nodes],
                    "node_types": [n.entity_type.value if hasattr(n.entity_type, "value") else str(n.entity_type) for n in path.nodes],
                    "path_score": path.path_score,
                    "depth": len(path.nodes) - 1
                }
            ))

            if len(evidence_list) >= top_k:
                break

        if budget:
            budget.record_call(hops=hops_taken, candidates=len(evidence_list))

        return evidence_list
