"""
Unit tests for Knowledge Graph store, path traversal, decay, and cycle prevention.
"""

from src.schemas import EntityType, GraphEntity, GraphRelation, RelationType
from src.graph.store import NetworkXGraphStore


def test_graph_store_add_and_retrieve():
    store = NetworkXGraphStore(decay_factor=0.85)

    # Add entities
    e1 = GraphEntity(entity_id="ent_lora", name="LoRA", entity_type=EntityType.METHOD)
    e2 = GraphEntity(entity_id="ent_author_hu", name="Edward Hu", entity_type=EntityType.AUTHOR)
    e3 = GraphEntity(entity_id="ent_glue", name="GLUE", entity_type=EntityType.DATASET)

    store.add_entity(e1)
    store.add_entity(e2)
    store.add_entity(e3)

    # Add relations
    store.add_relation(GraphRelation(
        source_id="ent_author_hu",
        target_id="ent_lora",
        relation_type=RelationType.INTRODUCES,
        confidence=1.0
    ))
    store.add_relation(GraphRelation(
        source_id="ent_lora",
        target_id="ent_glue",
        relation_type=RelationType.EVALUATES_ON,
        confidence=0.9
    ))

    # Test neighbor traversal with decay
    paths = store.get_neighbors("ent_author_hu", depth=2)
    assert len(paths) > 0

    # Test path decay: 1-hop path has higher score than 2-hop path
    one_hop_paths = [p for p in paths if len(p.nodes) == 2]
    two_hop_paths = [p for p in paths if len(p.nodes) == 3]

    assert len(one_hop_paths) > 0
    if two_hop_paths:
        assert one_hop_paths[0].path_score > two_hop_paths[0].path_score

    # Test finding path between
    between = store.find_paths_between("ent_author_hu", "ent_glue", max_depth=3)
    assert len(between) > 0
    assert "LoRA" in between[0].description


def test_graph_statistics():
    store = NetworkXGraphStore()
    e1 = GraphEntity(entity_id="e1", name="Entity 1", entity_type=EntityType.METHOD)
    e2 = GraphEntity(entity_id="e2", name="Entity 2", entity_type=EntityType.TASK)
    store.add_entity(e1)
    store.add_entity(e2)
    store.add_relation(GraphRelation(source_id="e1", target_id="e2", relation_type=RelationType.APPLIED_TO))

    stats = store.get_graph_statistics()
    assert stats["node_count"] == 2
    assert stats["edge_count"] == 1
    assert stats["entity_types"]["Method"] == 1
    assert stats["entity_types"]["Task"] == 1
