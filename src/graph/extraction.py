"""
Entity and relation extraction and resolution pipeline.
Performs alias normalization, schema validation, and entity linking.
"""

from __future__ import annotations

import re
from typing import Any
from src.schemas import Chunk, EntityType, GraphEntity, GraphRelation, RelationType


# Canonical alias dictionary mapping variant surface forms to canonical entity IDs
CANONICAL_ALIASES = {
    # Methods & Models
    "lora": "ent_lora",
    "low-rank adaptation": "ent_lora",
    "low rank adaptation": "ent_lora",
    "qlora": "ent_qlora",
    "adalora": "ent_adalora",
    "prefix-tuning": "ent_prefix_tuning",
    "prefix tuning": "ent_prefix_tuning",
    "prompt tuning": "ent_prompt_tuning",
    "adapter": "ent_adapters",
    "adapters": "ent_adapters",
    "bert": "ent_bert",
    "roberta": "ent_roberta",
    "t5": "ent_t5",
    "transformer": "ent_transformer",
    "transformers": "ent_transformer",
    "dpr": "ent_dpr",
    "dense passage retrieval": "ent_dpr",
    "realm": "ent_realm",
    "rag": "ent_rag",
    "fid": "ent_fid",
    "fusion-in-decoder": "ent_fid",
    "colbert": "ent_colbert",
    "retro": "ent_retro",
    "hyde": "ent_hyde",
    "graphrag": "ent_graphrag",
    "self-rag": "ent_self_rag",
    "crag": "ent_crag",
    "bge": "ent_bge",

    # Authors
    "edward hu": "ent_author_edward_hu",
    "hu et al.": "ent_author_edward_hu",
    "vaswani": "ent_author_vaswani",
    "ashish vaswani": "ent_author_vaswani",
    "devlin": "ent_author_devlin",
    "jacob devlin": "ent_author_devlin",
    "karpukhin": "ent_author_karpukhin",
    "vladimir karpukhin": "ent_author_karpukhin",
    "patrick lewis": "ent_author_patrick_lewis",
    "khattab": "ent_author_khattab",
    "omar khattab": "ent_author_khattab",
    "dettmers": "ent_author_dettmers",
    "tim dettmers": "ent_author_dettmers",

    # Benchmarks & Datasets
    "glue": "ent_benchmark_glue",
    "squad": "ent_benchmark_squad",
    "ms marco": "ent_benchmark_ms_marco",
    "ms-marco": "ent_benchmark_ms_marco",
    "natural questions": "ent_benchmark_nq",
    "triviaqa": "ent_benchmark_triviaqa",
}


class EntityExtractor:
    """Extracts entities and relations from text chunks with alias normalization."""

    def __init__(self):
        self.resolution_audit_log: list[dict[str, Any]] = []

    def resolve_entity(self, surface_form: str) -> str | None:
        """Resolves surface string to canonical entity ID."""
        clean = surface_form.strip().lower()
        clean = re.sub(r"[^\w\s-]", "", clean)

        if clean in CANONICAL_ALIASES:
            canonical_id = CANONICAL_ALIASES[clean]
            self.resolution_audit_log.append({
                "surface_form": surface_form,
                "canonical_id": canonical_id,
                "method": "exact_alias_match"
            })
            return canonical_id

        # Partial token match
        for alias, can_id in CANONICAL_ALIASES.items():
            if alias in clean or clean in alias:
                self.resolution_audit_log.append({
                    "surface_form": surface_form,
                    "canonical_id": can_id,
                    "method": "partial_alias_match"
                })
                return can_id

        return None

    def extract_from_chunk(self, chunk: Chunk) -> tuple[list[GraphEntity], list[GraphRelation]]:
        """Extracts candidate entities and relations from chunk text using pattern matching."""
        text = chunk.text
        entities: list[GraphEntity] = []
        relations: list[GraphRelation] = []

        found_entities: set[str] = set()

        for alias, ent_id in CANONICAL_ALIASES.items():
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                if ent_id not in found_entities:
                    found_entities.add(ent_id)
                    # Determine coarse entity type
                    if "author" in ent_id:
                        etype = EntityType.AUTHOR
                    elif "benchmark" in ent_id:
                        etype = EntityType.DATASET
                    else:
                        etype = EntityType.METHOD

                    entities.append(GraphEntity(
                        entity_id=ent_id,
                        name=alias.title(),
                        entity_type=etype,
                        aliases=[alias],
                        source_doc_ids=[chunk.doc_id],
                        metadata={"source_chunk": chunk.chunk_id}
                    ))

        # Heuristic relations between co-occurring entities in same chunk
        ent_list = list(found_entities)
        for i in range(len(ent_list)):
            for j in range(i + 1, len(ent_list)):
                e1, e2 = ent_list[i], ent_list[j]
                if "author" in e1 and "author" not in e2:
                    relations.append(GraphRelation(
                        source_id=e1,
                        target_id=e2,
                        relation_type=RelationType.INTRODUCES,
                        source_doc_ids=[chunk.doc_id],
                        source_chunk_ids=[chunk.chunk_id],
                        confidence=0.9
                    ))
                elif "benchmark" in e2 and "benchmark" not in e1:
                    relations.append(GraphRelation(
                        source_id=e1,
                        target_id=e2,
                        relation_type=RelationType.EVALUATES_ON,
                        source_doc_ids=[chunk.doc_id],
                        source_chunk_ids=[chunk.chunk_id],
                        confidence=0.85
                    ))
                elif "lora" in e1 and "adapter" in e2:
                    relations.append(GraphRelation(
                        source_id=e1,
                        target_id=e2,
                        relation_type=RelationType.IMPROVES,
                        source_doc_ids=[chunk.doc_id],
                        source_chunk_ids=[chunk.chunk_id],
                        confidence=0.95
                    ))
                elif "transformer" in e2:
                    relations.append(GraphRelation(
                        source_id=e1,
                        target_id=e2,
                        relation_type=RelationType.BASED_ON,
                        source_doc_ids=[chunk.doc_id],
                        source_chunk_ids=[chunk.chunk_id],
                        confidence=0.9
                    ))

        return entities, relations


class EntityLinker:
    """Links entities mentioned in user queries to GraphEntity IDs."""

    def __init__(self, extractor: EntityExtractor | None = None):
        self.extractor = extractor or EntityExtractor()

    def link_query(self, query: str) -> list[str]:
        """Detects mentioned entities in query and returns list of canonical entity IDs."""
        linked_ids: set[str] = set()
        clean = query.lower()

        for alias, can_id in CANONICAL_ALIASES.items():
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, clean, re.IGNORECASE):
                linked_ids.add(can_id)

        return sorted(list(linked_ids))
