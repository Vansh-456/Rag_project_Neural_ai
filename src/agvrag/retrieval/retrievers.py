"""Retrievers. Each returns a ranked ``list[Evidence]``; none of them generate text.

* ``DenseRetriever``       — query embedding -> vector store top-k.
* ``SparseRetriever``      — BM25 top-k.
* ``HybridRetriever``      — dense + BM25 fused (RRF or weighted).
* ``GraphRetriever``       — entity linking -> bounded multi-hop paths -> serialized path
                             evidence grounded to source chunks.
* ``GraphVectorRetriever`` — unified pool: dense + BM25 + graph-grounded chunks + graph paths,
                             with *vector-guided graph* seeding (entities mentioned in the top
                             dense chunks become extra low-weight seeds).
* ``MultiHopRetriever``    — *graph-guided vector*: traverse the graph first, then run a dense
                             search restricted to documents on the paths with the query
                             expanded by path entity names, and fuse with path evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agvrag.config import RetrievalConfig
from agvrag.graph.gazetteer import Gazetteer
from agvrag.graph.linking import EntityLinker
from agvrag.graph.traversal import GraphTraverser, serialize_path
from agvrag.interfaces import GraphStore, ScoredChunk
from agvrag.retrieval.fusion import rrf_fuse, weighted_fuse
from agvrag.schemas import Chunk, Evidence, EvidenceType, GraphPath, QueryAnalysis
from agvrag.utils.ids import stable_hash

CostHook = Callable[[str, float], None]  # (action, amount) -> charge the budget


def _noop(action: str, amount: float) -> None:
    return None


def chunk_evidence(sc: ScoredChunk, retriever: str, rank: int) -> Evidence:
    c = sc.chunk
    return Evidence(
        evidence_id=f"chunk:{c.chunk_id}",
        type=EvidenceType.TEXT_CHUNK,
        content=c.text,
        source_doc_ids=[c.doc_id],
        source_chunk_ids=[c.chunk_id],
        retriever=retriever,
        raw_score=round(float(sc.score), 6),
        rank=rank,
        provenance={
            "doc_title": c.metadata.get("doc_title"),
            "section": c.section_heading,
            "page_start": c.page_start,
            "page_end": c.page_end,
        },
    )


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: Any, store: Any, cost: CostHook = _noop) -> None:
        self.embedder = embedder
        self.store = store
        self.cost = cost

    def search(self, text: str, k: int, doc_filter: list[str] | None = None) -> list[Evidence]:
        self.cost("dense", 1.0)
        vec = self.embedder.embed_query(text)
        hits = self.store.search(vec, k, doc_filter=doc_filter)
        return [chunk_evidence(h, self.name, i) for i, h in enumerate(hits, start=1)]

    def retrieve(self, query: str, analysis: QueryAnalysis, k: int) -> list[Evidence]:
        return self.search(query, k)


class SparseRetriever:
    name = "bm25"

    def __init__(self, index: Any, cost: CostHook = _noop) -> None:
        self.index = index
        self.cost = cost

    def retrieve(self, query: str, analysis: QueryAnalysis, k: int) -> list[Evidence]:
        self.cost("bm25", 1.0)
        hits = self.index.search(query, k)
        return [chunk_evidence(h, self.name, i) for i, h in enumerate(hits, start=1)]


def fuse(lists: dict[str, list[Evidence]], cfg: RetrievalConfig, top_n: int) -> list[Evidence]:
    lists = {k: v for k, v in lists.items() if v}
    if cfg.fusion == "weighted":
        return weighted_fuse(lists, cfg.weights, cfg.normalization, top_n)
    return rrf_fuse(lists, cfg.rrf_k, None, top_n)


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self, dense: DenseRetriever, sparse: SparseRetriever, cfg: RetrievalConfig
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.cfg = cfg
        self.last_lists: dict[str, list[Evidence]] = {}

    def retrieve(self, query: str, analysis: QueryAnalysis, k: int) -> list[Evidence]:
        pool = max(k, self.cfg.candidate_pool)
        self.last_lists = {
            "dense": self.dense.retrieve(query, analysis, pool),
            "bm25": self.sparse.retrieve(query, analysis, pool),
        }
        return fuse(self.last_lists, self.cfg, k)


class GraphRetriever:
    """Graph-only retrieval. Evidence = serialized paths, grounded to source chunks."""

    name = "graph"

    def __init__(
        self,
        store: GraphStore,
        linker: EntityLinker,
        chunks: dict[str, Chunk],
        cfg: RetrievalConfig,
        cost: CostHook = _noop,
    ) -> None:
        self.store = store
        self.linker = linker
        self.chunks = chunks
        self.cfg = cfg
        self.gcfg = cfg.graph
        self.cost = cost
        self.traverser = GraphTraverser(
            store, self.gcfg.hop_decay, self.gcfg.relation_cue_bonus, self.gcfg.max_expansions
        )
        self.last_paths: list[GraphPath] = []

    def seeds_for(self, analysis: QueryAnalysis) -> dict[str, float]:
        return {e["entity_id"]: float(e.get("score", 1.0)) for e in analysis.linked_entities}

    def paths(
        self,
        analysis: QueryAnalysis,
        extra_seeds: dict[str, float] | None = None,
        max_hops: int | None = None,
    ) -> list[GraphPath]:
        seeds = self.seeds_for(analysis)
        for k, v in (extra_seeds or {}).items():
            seeds.setdefault(k, v)
        if not seeds:
            self.last_paths = []
            return []
        hops = (
            max_hops
            if max_hops is not None
            else max(1, min(self.gcfg.max_hops, analysis.hops_estimate or 1))
        )
        paths = self.traverser.paths(seeds, hops, None, analysis.relation_cues, self.gcfg.max_paths)
        self.cost("graph_hop", float(hops))
        self.last_paths = paths
        return paths

    def path_evidence(self, paths: list[GraphPath]) -> list[Evidence]:
        out: list[Evidence] = []
        for i, p in enumerate(paths, start=1):
            chunk_ids: list[str] = []
            for e in p.edges:
                for cid in e.source_chunk_ids[: self.gcfg.chunks_per_path]:
                    if cid in self.chunks and cid not in chunk_ids:
                        chunk_ids.append(cid)
            doc_ids = list(dict.fromkeys(self.chunks[c].doc_id for c in chunk_ids))
            for n in p.nodes:
                if n.startswith("paper:") and n[6:] not in doc_ids:
                    doc_ids.append(n[6:])
            out.append(
                Evidence(
                    evidence_id=f"path:{stable_hash([e.key for e in p.edges], length=12)}",
                    type=EvidenceType.GRAPH_PATH if p.hops > 1 else EvidenceType.GRAPH_FACT,
                    content=serialize_path(p, self.store),
                    source_doc_ids=doc_ids,
                    source_chunk_ids=chunk_ids,
                    retriever=self.name,
                    raw_score=round(p.score, 6),
                    rank=i,
                    provenance={
                        "nodes": p.nodes,
                        "directions": p.directions,
                        "hops": p.hops,
                        "edges": [
                            {
                                "source": e.source,
                                "type": e.type,
                                "target": e.target,
                                "confidence": e.confidence,
                                "evidence": e.evidence_text[:300],
                                "extractor": e.extractor,
                                "chunks": e.source_chunk_ids[:3],
                            }
                            for e in p.edges
                        ],
                    },
                )
            )
        return out

    def grounded_chunks(self, path_evs: list[Evidence]) -> list[Evidence]:
        """Flatten path evidence into a ranked chunk list (best path first)."""
        out: list[Evidence] = []
        seen: set[str] = set()
        for ev in path_evs:
            for cid in ev.source_chunk_ids:
                if cid in seen:
                    continue
                seen.add(cid)
                out.append(
                    chunk_evidence(
                        ScoredChunk(self.chunks[cid], ev.raw_score), "graph", len(out) + 1
                    )
                )
                out[-1].provenance["via_path"] = ev.evidence_id
        return out

    def retrieve(self, query: str, analysis: QueryAnalysis, k: int) -> list[Evidence]:
        return self.path_evidence(self.paths(analysis))[:k]


class GraphVectorRetriever:
    name = "graph_vector"

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        graph: GraphRetriever,
        gazetteer: Gazetteer | None,
        cfg: RetrievalConfig,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.graph = graph
        self.gazetteer = gazetteer
        self.cfg = cfg
        self.last_lists: dict[str, list[Evidence]] = {}

    def _vector_seeds(self, dense_hits: list[Evidence], top: int = 3) -> dict[str, float]:
        if self.gazetteer is None:
            return {}
        seeds: dict[str, float] = {}
        for ev in dense_hits[:top]:
            for m in self.gazetteer.find_mentions(ev.content):
                if self.graph.store.get_entity(m.entity_id) is not None:
                    seeds.setdefault(m.entity_id, 0.5)
        return dict(list(seeds.items())[:5])

    def retrieve(self, query: str, analysis: QueryAnalysis, k: int) -> list[Evidence]:
        pool = max(k, self.cfg.candidate_pool)
        dense = self.dense.retrieve(query, analysis, pool)
        sparse = self.sparse.retrieve(query, analysis, pool)
        extra = self._vector_seeds(dense) if not analysis.linked_entities else {}
        path_evs = self.graph.path_evidence(self.graph.paths(analysis, extra))
        graph_chunks = self.graph.grounded_chunks(path_evs)
        self.last_lists = {
            "dense": dense,
            "bm25": sparse,
            "graph": graph_chunks,
            "graph_paths": path_evs,
        }
        fused = fuse({"dense": dense, "bm25": sparse, "graph": graph_chunks}, self.cfg, pool)
        # Paths are kept as separate evidence (they carry relational facts) and are
        # interleaved after fusion: top paths go into the pool alongside chunks.
        return _interleave(fused, path_evs[: max(2, k // 3)], k)


class MultiHopRetriever:
    name = "multi_hop"

    def __init__(self, dense: DenseRetriever, graph: GraphRetriever, cfg: RetrievalConfig) -> None:
        self.dense = dense
        self.graph = graph
        self.cfg = cfg
        self.last_lists: dict[str, list[Evidence]] = {}

    def retrieve(self, query: str, analysis: QueryAnalysis, k: int) -> list[Evidence]:
        pool = max(k, self.cfg.candidate_pool)
        hops = max(2, min(self.cfg.graph.max_hops, analysis.hops_estimate or 2))
        path_evs = self.graph.path_evidence(self.graph.paths(analysis, max_hops=hops))
        graph_chunks = self.graph.grounded_chunks(path_evs)
        doc_ids = list(dict.fromkeys(d for ev in path_evs for d in ev.source_doc_ids))
        names: list[str] = []
        for ev in path_evs[:5]:
            for n in ev.provenance.get("nodes", []):
                ent = self.graph.store.get_entity(n)
                if ent and ent.type not in {"Paper", "Author"} and ent.name not in names:
                    names.append(ent.name)
        expanded = f"{query} {' '.join(names[:8])}".strip()
        restricted = self.dense.search(expanded, pool, doc_filter=doc_ids) if doc_ids else []
        open_dense = self.dense.search(query, pool)
        self.last_lists = {
            "graph": graph_chunks,
            "dense_restricted": restricted,
            "dense": open_dense,
            "graph_paths": path_evs,
        }
        fused = fuse(
            {"graph": graph_chunks, "dense_restricted": restricted, "dense": open_dense},
            self.cfg,
            pool,
        )
        return _interleave(fused, path_evs[: max(2, k // 3)], k)


def _interleave(chunks: list[Evidence], paths: list[Evidence], k: int) -> list[Evidence]:
    """Place one path after every two chunks until paths run out; re-number ranks.

    Chunk ranking (what retrieval metrics see) is preserved — only paths are inserted.
    """
    out: list[Evidence] = []
    pi = 0
    for i, ev in enumerate(chunks):
        out.append(ev)
        if (i + 1) % 2 == 0 and pi < len(paths):
            out.append(paths[pi])
            pi += 1
    out.extend(paths[pi:])
    budget_chunks = k
    trimmed: list[Evidence] = []
    n_chunks = 0
    for ev in out:
        if ev.type == EvidenceType.TEXT_CHUNK:
            if n_chunks >= budget_chunks:
                continue
            n_chunks += 1
        trimmed.append(ev)
    for i, ev in enumerate(trimmed, start=1):
        ev.rank = i
    return trimmed
