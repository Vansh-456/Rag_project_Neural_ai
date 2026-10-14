"""
Multi-hop retrieval module.
Coordinates Graph-Guided Vector retrieval and Vector-Guided Graph retrieval.
"""

from __future__ import annotations

from src.schemas import Evidence, RetrievalBudget
from src.retrieval.dense import DenseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.fusion import reciprocal_rank_fusion


class MultiHopRetriever:
    """Orchestrates multi-hop graph traversal chained into vector evidence expansion."""

    def __init__(self, dense_retriever: DenseRetriever, graph_retriever: GraphRetriever):
        self.dense_retriever = dense_retriever
        self.graph_retriever = graph_retriever

    def retrieve(self, query: str, top_k: int = 5, budget: RetrievalBudget | None = None) -> list[Evidence]:
        """
        1. Retrieve relational graph paths for query entities.
        2. Extract connected target entities from paths.
        3. Retrieve dense chunks for original query AND graph-discovered entities.
        4. Fuse heterogeneous evidence into unified candidate pool.
        """
        # Step 1: Graph traversal
        graph_evidence = self.graph_retriever.retrieve(query, top_k=top_k, budget=budget)

        # Step 2: Extract discovered entities
        discovered_entities: set[str] = set()
        for ev in graph_evidence:
            nodes = ev.provenance.get("nodes", [])
            for node in nodes:
                discovered_entities.add(str(node))

        # Step 3: Base dense retrieval
        dense_evidence = self.dense_retriever.retrieve(query, top_k=top_k, budget=budget)

        # Step 4: Secondary dense expansion if entities discovered and budget permits
        expanded_dense: list[Evidence] = []
        if discovered_entities and (budget is None or budget.can_call_retrieval()):
            expansion_query = f"{query} {' '.join(list(discovered_entities)[:3])}"
            expanded_dense = self.dense_retriever.retrieve(expansion_query, top_k=top_k, budget=budget)

        # Step 5: Reciprocal Rank Fusion of heterogeneous lists
        candidate_lists = [dense_evidence, graph_evidence]
        if expanded_dense:
            candidate_lists.append(expanded_dense)

        fused = reciprocal_rank_fusion(candidate_lists, k=60, top_k=top_k)
        return fused
