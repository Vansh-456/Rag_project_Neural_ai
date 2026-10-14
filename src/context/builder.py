"""
Context construction module.
Formats structured, token-budgeted, deduplicated context with explicit citation IDs [E1], [E2].
Includes lost-in-the-middle attention-aware evidence ordering.
"""

from __future__ import annotations

from typing import Any
from src.schemas import Evidence, EvidenceType
from src.utils.metrics import estimate_tokens


class ContextBuilder:
    """Constructs structured evidence context for generation with token budgeting and citation tracking."""

    def __init__(self, max_tokens: int = 2048, ordering: str = "lost_in_the_middle"):
        self.max_tokens = max_tokens
        self.ordering = ordering  # "lost_in_the_middle" | "relevance_first"

    def _order_evidence(self, evidence: list[Evidence]) -> list[Evidence]:
        """
        Orders evidence items. 'lost_in_the_middle' places the top items at the
        start and end of the context where transformer attention is strongest.
        """
        if self.ordering != "lost_in_the_middle" or len(evidence) <= 2:
            return evidence

        # Alternating sandwich: 1st at start, 2nd at end, 3rd near start, 4th near end...
        sorted_ev = sorted(evidence, key=lambda e: e.rank)
        reordered: list[Evidence] = [None] * len(sorted_ev)  # type: ignore

        left = 0
        right = len(sorted_ev) - 1
        for i, ev in enumerate(sorted_ev):
            if i % 2 == 0:
                reordered[left] = ev
                left += 1
            else:
                reordered[right] = ev
                right -= 1

        return [e for e in reordered if e is not None]

    def build_context(
        self,
        evidence: list[Evidence],
        query: str,
        detected_entities: list[str] | None = None,
        max_tokens: int | None = None
    ) -> tuple[str, list[Evidence]]:
        """
        Deduplicates evidence, assigns citation IDs [E1], [E2], packs within token budget,
        and constructs the structured markdown context.
        """
        limit = max_tokens or self.max_tokens

        # Deduplicate evidence by source chunk ID or content hash
        seen_keys: set[str] = set()
        deduped: list[Evidence] = []
        for ev in evidence:
            key = ev.source_chunk_ids[0] if ev.source_chunk_ids else ev.content[:60]
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(ev)

        # Order evidence
        ordered_ev = self._order_evidence(deduped)

        # Build context blocks and enforce token budget
        packed_evidence: list[Evidence] = []
        evidence_blocks: list[str] = []
        current_tokens = estimate_tokens(query) + 100

        for idx, ev in enumerate(ordered_ev, start=1):
            citation_id = f"E{idx}"
            title = ev.provenance.get("doc_title", "")
            page = ev.provenance.get("page_start", "")
            header = f"[{citation_id}] (Source: {ev.retriever.upper()}"
            if title:
                header += f" | Document: {title}"
            if page:
                header += f" | Page: {page}"
            header += ")"

            block = f"{header}\n{ev.content}\n"
            block_tokens = estimate_tokens(block)

            if current_tokens + block_tokens > limit:
                break

            current_tokens += block_tokens
            evidence_blocks.append(block)
            packed_evidence.append(ev)

        # Assemble full prompt context
        entities_str = ", ".join(detected_entities) if detected_entities else "None"
        context_body = "\n".join(evidence_blocks) if evidence_blocks else "No relevant evidence retrieved."

        full_context = (
            f"# USER QUERY:\n{query}\n\n"
            f"# DETECTED ENTITIES:\n{entities_str}\n\n"
            f"# RETRIEVED EVIDENCE:\n{context_body}\n"
        )

        return full_context, packed_evidence
