"""Context construction: dedup -> (optional MMR) -> ordering -> token-budgeted packing.

Output is deterministic for the same evidence list and config. Every evidence item gets a
citation label: ``G1..Gn`` for graph paths/facts and ``E1..En`` for text chunks, so the
generator can cite by id and the validator can check the ids exist.

Ordering:
* ``relevance`` — best first.
* ``edges``     — best items at the start *and* end, weakest in the middle, mitigating the
                  "lost in the middle" effect (Liu et al., 2023) where models under-use the
                  middle of long contexts.
"""

from __future__ import annotations

from agvrag.config import ContextConfig
from agvrag.interfaces import BuiltContext
from agvrag.schemas import Evidence, EvidenceType, QueryAnalysis
from agvrag.utils.text import count_tokens, jaccard, shingles


def dedup_evidence(
    evidence: list[Evidence], jaccard_threshold: float
) -> tuple[list[Evidence], list[str]]:
    """Remove exact duplicates (same chunk / same evidence id) and near-duplicates by shingles."""
    kept: list[Evidence] = []
    kept_sh: list[set[tuple[str, ...]]] = []
    seen: set[str] = set()
    dropped: list[str] = []
    for ev in evidence:
        key = (
            ev.source_chunk_ids[0]
            if ev.type == EvidenceType.TEXT_CHUNK and ev.source_chunk_ids
            else ev.evidence_id
        )
        if key in seen:
            dropped.append(ev.evidence_id)
            continue
        sh = shingles(ev.content)
        if ev.type == EvidenceType.TEXT_CHUNK and any(
            jaccard(sh, k) >= jaccard_threshold for k in kept_sh
        ):
            dropped.append(ev.evidence_id)
            continue
        seen.add(key)
        kept.append(ev)
        kept_sh.append(sh)
    return kept, dropped


def mmr_select(evidence: list[Evidence], lam: float, k: int) -> list[Evidence]:
    """Maximal Marginal Relevance with shingle-Jaccard as the redundancy measure.

    MMR(d) = lam * rel(d) - (1 - lam) * max_{s in S} sim(d, s); rel = 1/rank (scale-free).
    """
    pool = list(evidence)
    sh = {e.evidence_id: shingles(e.content) for e in pool}
    selected: list[Evidence] = []
    while pool and len(selected) < k:

        def score(e: Evidence) -> float:
            red = max(
                (jaccard(sh[e.evidence_id], sh[s.evidence_id]) for s in selected), default=0.0
            )
            return lam * (1.0 / e.rank) - (1 - lam) * red

        best = max(pool, key=lambda e: (score(e), -e.rank))
        selected.append(best)
        pool.remove(best)
    return selected


def edges_order(items: list[Evidence]) -> list[Evidence]:
    """[1,2,3,4,5] -> [1,3,5,4,2]: strongest items at both ends."""
    front = items[0::2]
    back = items[1::2]
    return front + back[::-1]


class StructuredContextBuilder:
    def __init__(self, cfg: ContextConfig) -> None:
        self.cfg = cfg

    def build(self, query: str, analysis: QueryAnalysis, evidence: list[Evidence]) -> BuiltContext:
        deduped, dropped = dedup_evidence(evidence, self.cfg.dedup_jaccard)
        paths = [e for e in deduped if e.type != EvidenceType.TEXT_CHUNK][
            : self.cfg.max_graph_paths
        ]
        texts = [e for e in deduped if e.type == EvidenceType.TEXT_CHUNK]
        if self.cfg.mmr_lambda is not None:
            texts = mmr_select(texts, self.cfg.mmr_lambda, len(texts))

        header = [f"QUESTION: {query}"]
        if analysis.entities:
            header.append("DETECTED ENTITIES: " + ", ".join(analysis.entities))
        used = count_tokens("\n".join(header))
        budget = self.cfg.max_tokens
        labels: dict[str, Evidence] = {}
        path_lines: list[str] = []
        for i, ev in enumerate(paths, start=1):
            line = f"[G{i}] {ev.content}"
            t = count_tokens(line)
            if used + t > budget:
                dropped.append(ev.evidence_id)
                continue
            used += t
            labels[f"G{i}"] = ev
            path_lines.append(line)
        text_blocks: list[tuple[str, Evidence]] = []
        n = 0
        for ev in texts:
            p = ev.provenance
            pages = f"pp. {p.get('page_start')}-{p.get('page_end')}" if p.get("page_start") else ""
            src = f"{p.get('doc_title') or ev.source_doc_ids[0]} | {p.get('section') or ''} | {pages}".strip(
                " |"
            )
            block_body = f"(source: {src})\n{ev.content.strip()}"
            t = count_tokens(block_body) + 4
            if used + t > budget:
                dropped.append(ev.evidence_id)
                continue
            used += t
            n += 1
            text_blocks.append((f"E{n}", ev))
            labels[f"E{n}"] = ev
        ordered = text_blocks
        if self.cfg.ordering == "edges":
            ordered_evs = edges_order([ev for _, ev in text_blocks])
            label_of = {id(ev): lab for lab, ev in text_blocks}
            ordered = [(label_of[id(ev)], ev) for ev in ordered_evs]
        parts = header[:]
        if path_lines:
            parts.append("\nGRAPH EVIDENCE (relations extracted from the corpus):")
            parts.extend(path_lines)
        if ordered:
            parts.append("\nTEXT EVIDENCE:")
            for lab, ev in ordered:
                p = ev.provenance
                pages = (
                    f"pp. {p.get('page_start')}-{p.get('page_end')}" if p.get("page_start") else ""
                )
                src = f"{p.get('doc_title') or ev.source_doc_ids[0]} | {p.get('section') or ''} | {pages}".strip(
                    " |"
                )
                parts.append(f"[{lab}] (source: {src})\n{ev.content.strip()}\n")
        text = "\n".join(parts).strip()
        return BuiltContext(
            text=text, evidence_ids=labels, token_count=count_tokens(text), dropped=dropped
        )
