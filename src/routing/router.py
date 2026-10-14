"""
Query analysis and adaptive retrieval routing module.
Analyzes query intent, complexity, and linked entities to select the optimal retrieval strategy.
"""

from __future__ import annotations

import re
from typing import Any
from src.schemas import QueryAnalysis, QueryIntent, RetrievalStrategy, RoutingDecision
from src.graph.extraction import EntityLinker


class QueryAnalyzer:
    """Extracts semantic features, intent, entity mentions, and complexity from raw queries."""

    def __init__(self, entity_linker: EntityLinker | None = None):
        self.entity_linker = entity_linker or EntityLinker()

    def analyze(self, query: str) -> QueryAnalysis:
        q_lower = query.lower().strip()
        linked_entities = self.entity_linker.link_query(query)

        # 1. Intent Detection Heuristics
        intent = QueryIntent.DEFINITION
        rationale_parts = []

        if re.search(r"\b(who (introduced|authored|created|proposed|wrote)|cites|authors? of)\b", q_lower):
            intent = QueryIntent.RELATIONSHIP
            rationale_parts.append("Matched relationship question cues ('who introduced/authored/cites')")
        elif re.search(r"\b(which methods?|what techniques?).*(derived from|improved|based on)\b", q_lower) or \
             re.search(r"\b(how did .* lead to|path between|multi-hop)\b", q_lower):
            intent = QueryIntent.MULTI_HOP
            rationale_parts.append("Matched multi-hop relational progression pattern")
        elif re.search(r"\b(compare|differs? from|versus|vs\.?|difference between)\b", q_lower):
            intent = QueryIntent.COMPARISON
            rationale_parts.append("Matched comparative cues ('compare/differs/versus')")
        elif re.search(r"\b(evidence|results?|ablation|experiments?|benchmark scores?|quantitative|numbers?|improvement|improvements? over|bleu|rouge|accuracy|f1)\b", q_lower):
            intent = QueryIntent.EXPERIMENTAL
            rationale_parts.append("Matched experimental and quantitative metric cues")
        elif re.search(r"\b(what is|define|explain|meaning of|overview of)\b", q_lower):
            intent = QueryIntent.DEFINITION
            rationale_parts.append("Matched definitional cues ('what is/define/explain')")
        else:
            # Fallback based on entity counts
            if len(linked_entities) >= 2:
                intent = QueryIntent.COMPARISON
                rationale_parts.append("Multiple entities detected without explicit definitional pattern")
            elif len(linked_entities) == 1:
                intent = QueryIntent.RELATIONSHIP
                rationale_parts.append("Single discrete entity detected in relational context")
            else:
                intent = QueryIntent.DEFINITION
                rationale_parts.append("Standard semantic inquiry")

        # 2. Complexity & Hops Estimation
        words = q_lower.split()
        base_complexity = min(1.0, len(words) / 25.0)
        entity_complexity = min(0.4, len(linked_entities) * 0.15)
        relational_bump = 0.2 if intent in (QueryIntent.MULTI_HOP, QueryIntent.COMPARISON) else 0.0

        complexity = round(min(1.0, base_complexity * 0.4 + entity_complexity + relational_bump), 2)
        hops_estimate = 2 if intent == QueryIntent.MULTI_HOP else (1 if len(linked_entities) > 0 else 0)

        rationale = f"{'; '.join(rationale_parts)}. Linked entities: {len(linked_entities)}, estimated complexity: {complexity}."

        return QueryAnalysis(
            query=query,
            intent=intent.value,
            entities=linked_entities,
            complexity=complexity,
            hops_estimate=hops_estimate,
            rationale=rationale
        )


class AdaptiveRouter:
    """Selects the retrieval strategy based on QueryAnalysis according to the project taxonomy."""

    def __init__(self, analyzer: QueryAnalyzer | None = None):
        self.analyzer = analyzer or QueryAnalyzer()

    def route(self, analysis: QueryAnalysis) -> RoutingDecision:
        intent = analysis.intent
        entities = analysis.entities

        if intent == QueryIntent.DEFINITION.value and len(entities) <= 1:
            return RoutingDecision(
                strategy=RetrievalStrategy.VECTOR.value,
                params={"dense_top_k": 5, "candidate_pool": 10},
                rationale="Definitional query with ≤1 entity: dense vector search provides optimal semantic passage coverage."
            )

        if intent == QueryIntent.RELATIONSHIP.value and len(entities) >= 1:
            return RoutingDecision(
                strategy=RetrievalStrategy.GRAPH.value,
                params={"graph_top_k": 5, "hops": 1},
                rationale="Relationship query with linked entity: direct knowledge graph traversal isolates provenanced relations."
            )

        if intent == QueryIntent.COMPARISON.value:
            return RoutingDecision(
                strategy=RetrievalStrategy.HYBRID.value,
                params={"dense_top_k": 8, "sparse_top_k": 8, "final_top_k": 5},
                rationale="Comparison query across entities: hybrid fusion (dense + BM25) covers dual perspectives without bias."
            )

        if intent == QueryIntent.MULTI_HOP.value or analysis.hops_estimate >= 2:
            return RoutingDecision(
                strategy=RetrievalStrategy.MULTI_HOP.value,
                params={"graph_top_k": 5, "dense_top_k": 8, "hops": 2},
                rationale="Multi-hop query: graph traversal isolates entity progression path, followed by vector expansion for evidence."
            )

        if intent == QueryIntent.EXPERIMENTAL.value:
            return RoutingDecision(
                strategy=RetrievalStrategy.HYBRID.value,
                params={"dense_top_k": 10, "sparse_top_k": 10, "rerank": True, "final_top_k": 5},
                rationale="Experimental evidence query: hybrid dense + lexical retrieval with cross-encoder reranking isolates specific metric tables."
            )

        # Fallback
        return RoutingDecision(
            strategy=RetrievalStrategy.HYBRID.value,
            params={"dense_top_k": 5, "sparse_top_k": 5, "final_top_k": 5},
            rationale="General query: robust hybrid fusion provides balanced lexical and semantic coverage."
        )
