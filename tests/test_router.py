"""
Unit tests for QueryAnalyzer taxonomy classification and AdaptiveRouter strategy selection.
"""

from src.routing.router import AdaptiveRouter, QueryAnalyzer
from src.schemas import QueryIntent, RetrievalStrategy


def test_query_analyzer_intent_classification():
    analyzer = QueryAnalyzer()

    # Definitional
    res = analyzer.analyze("What is LoRA?")
    assert res.intent == QueryIntent.DEFINITION.value
    assert "ent_lora" in res.entities

    # Relational
    res = analyzer.analyze("Who introduced LoRA?")
    assert res.intent == QueryIntent.RELATIONSHIP.value

    # Comparative
    res = analyzer.analyze("Compare LoRA and adapters for fine-tuning")
    assert res.intent == QueryIntent.COMPARISON.value

    # Multi-hop
    res = analyzer.analyze("Which methods derived from Transformers improved GLUE?")
    assert res.intent == QueryIntent.MULTI_HOP.value
    assert res.hops_estimate >= 2

    # Experimental
    res = analyzer.analyze("What BLEU improvement did Transformer achieve on WMT 2014?")
    assert res.intent == QueryIntent.EXPERIMENTAL.value


def test_adaptive_router_decisions():
    router = AdaptiveRouter()

    # Definitional -> Vector
    analysis = router.analyzer.analyze("What is LoRA?")
    dec = router.route(analysis)
    assert dec.strategy == RetrievalStrategy.VECTOR.value

    # Relational -> Graph
    analysis = router.analyzer.analyze("Who introduced LoRA?")
    dec = router.route(analysis)
    assert dec.strategy == RetrievalStrategy.GRAPH.value

    # Comparative -> Hybrid
    analysis = router.analyzer.analyze("How does LoRA differ from adapters?")
    dec = router.route(analysis)
    assert dec.strategy == RetrievalStrategy.HYBRID.value

    # Multi-hop -> Multi_Hop
    analysis = router.analyzer.analyze("Which methods derived from Transformers improved GLUE?")
    dec = router.route(analysis)
    assert dec.strategy == RetrievalStrategy.MULTI_HOP.value
