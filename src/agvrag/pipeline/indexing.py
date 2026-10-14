"""Offline build steps: parse -> chunk -> embed/index -> graph. All idempotent.

Artifacts per index (``data/index/<index_name>/``):
  chunks.jsonl, manifest.json, qdrant/ (local mode) or vectors.npy+ids.json (memory),
  graph.json, graph_report.json, graph_stats.json, edge_audit_sample.json
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from agvrag.chunking.chunkers import build_chunker, chunk_embed_text, chunk_stats
from agvrag.config import Settings, dump_settings
from agvrag.graph.builder import build_graph, graph_statistics, sample_edges_for_audit
from agvrag.graph.store import Neo4jGraphStore, NetworkXGraphStore
from agvrag.ingestion.pdf_parser import get_parser, parse_quality
from agvrag.pipeline.components import ComponentRegistry, index_dir, open_vector_store
from agvrag.schemas import Chunk, Document
from agvrag.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from agvrag.utils.logging import get_logger
from agvrag.vector_store.stores import InMemoryVectorStore

log = get_logger(__name__)


def parse_corpus(
    settings: Settings,
    parser_name: str | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Parse every PDF in ``raw_dir`` into ``documents_dir`` and write the quality report."""
    raw = settings.path("raw_dir")
    out_dir = ensure_dir(settings.path("documents_dir"))
    name = parser_name or settings.ingestion.parser
    parser = get_parser(
        name, settings.ingestion.min_section_chars, settings.ingestion.max_heading_words
    )
    pdfs = sorted(raw.glob("*.pdf"))[:limit] if limit else sorted(raw.glob("*.pdf"))
    quality: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for pdf in pdfs:
        target = out_dir / f"{pdf.stem}.json"
        meta_path = pdf.with_suffix(".json")
        meta = read_json(meta_path) if meta_path.exists() else {"doc_id": pdf.stem}
        if target.exists() and not force and parser_name is None:
            doc = Document.from_dict(read_json(target))
        else:
            try:
                doc = parser.parse(pdf, meta)
            except ValueError as exc:
                failures.append({"file": pdf.name, "error": str(exc)})
                log.error("parse failed", extra={"file": pdf.name, "error": str(exc)})
                continue
            if parser_name is None:
                write_json(target, doc.to_dict(), indent=None)
        quality.append(parse_quality(doc))
    ok = sum(1 for q in quality if q["sections_identified"])
    report = {
        "parser": name,
        "n_pdfs": len(pdfs),
        "n_parsed": len(quality),
        "failures": failures,
        "pct_sections_identified": round(100 * ok / max(1, len(pdfs)), 2),
        "mean_garbled_score": round(
            sum(q["garbled_score"] for q in quality) / max(1, len(quality)), 4
        ),
        "mean_references": round(sum(q["n_references"] for q in quality) / max(1, len(quality)), 2),
        "per_document": quality,
        "not_identified": [q["doc_id"] for q in quality if not q["sections_identified"]],
    }
    return report


def load_documents(settings: Settings) -> list[Document]:
    d = settings.path("documents_dir")
    docs = [Document.from_dict(read_json(p)) for p in sorted(d.glob("*.json"))]
    corpus_meta = {p.stem: read_json(p) for p in settings.path("raw_dir").glob("*.json")}
    for doc in docs:
        if "topic" not in doc.metadata and doc.doc_id in corpus_meta:
            doc.metadata["topic"] = corpus_meta[doc.doc_id].get("topic", "")
    return docs


def chunk_documents(
    settings: Settings, docs: list[Document], embedder_factory: Callable[[], Any] | None = None
) -> list[Chunk]:
    chunker = build_chunker(settings.chunking, embedder_factory)
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunker.chunk(d))
    return out


def build_index(
    settings: Settings,
    registry: ComponentRegistry | None = None,
    docs: list[Document] | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Chunk + embed + upsert. Re-running with the same config is a no-op upsert (same ids)."""
    registry = registry or ComponentRegistry()
    docs = docs if docs is not None else load_documents(settings)
    if not docs:
        raise ValueError("no parsed documents; run `agvrag parse` first")
    t0 = time.perf_counter()
    embedder = registry.embedder(settings)
    chunks = chunk_documents(settings, docs, lambda: embedder)
    idx = ensure_dir(index_dir(settings))
    texts = [chunk_embed_text(c) for c in chunks]
    vecs = embedder.embed_documents(texts)
    if rebuild and settings.vector_store.backend != "memory":
        store = open_vector_store(settings, embedder.dim)
        store.delete_collection()
        store.close()
    store = open_vector_store(settings, embedder.dim)
    store.upsert(chunks, np.asarray(vecs, dtype=np.float32))
    if isinstance(store, InMemoryVectorStore):
        store.save(idx)
    else:
        store.close()
    write_jsonl(idx / "chunks.jsonl", (c.to_dict() for c in chunks))
    manifest = {
        "index_name": settings.index_name,
        "built_at": datetime.now(UTC).isoformat(),
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "embedding_model": embedder.model_name,
        "dim": embedder.dim,
        "vector_backend": settings.vector_store.backend,
        "chunk_stats": chunk_stats(chunks),
        "seconds": round(time.perf_counter() - t0, 2),
        "embedding_cache_hit_rate": round(getattr(embedder, "hit_rate", 0.0), 4),
    }
    write_json(idx / "manifest.json", manifest)
    (idx / "config.yaml").write_text(dump_settings(settings), encoding="utf-8")
    registry.invalidate_index(settings)
    log.info("index built", extra={k: v for k, v in manifest.items() if k != "chunk_stats"})
    return manifest


def build_graph_index(
    settings: Settings, registry: ComponentRegistry | None = None, audit_sample: int = 50
) -> dict[str, Any]:
    registry = registry or ComponentRegistry()
    idx = index_dir(settings)
    chunk_path = idx / "chunks.jsonl"
    if not chunk_path.exists():
        raise FileNotFoundError("index not built; run `agvrag index` first")
    chunks = [Chunk.from_dict(d) for d in read_jsonl(chunk_path)]
    docs = load_documents(settings)
    llm = None
    if settings.graph.extractor in {"llm", "rule+llm"}:
        llm = registry.llm(settings, effort=settings.generation.extraction_effort)
    t0 = time.perf_counter()
    store, report = build_graph(
        docs,
        chunks,
        registry.gazetteer(settings),
        settings.graph.extractor,
        llm,
        settings.graph.min_edge_confidence,
        settings.graph.max_llm_extraction_chunks,
        settings.seed,
    )
    store.save(idx / "graph.json")
    stats = graph_statistics(store)
    stats["seconds"] = round(time.perf_counter() - t0, 2)
    stats["extractor"] = settings.graph.extractor
    write_json(idx / "graph_stats.json", stats)
    write_json(idx / "graph_report.json", report)
    audit_path = idx / "edge_audit_sample.json"
    if not audit_path.exists():  # never overwrite a (possibly hand-labelled) audit file
        write_json(audit_path, sample_edges_for_audit(store, audit_sample, settings.seed))
    if settings.graph.backend == "neo4j":
        sync_neo4j(settings, store, registry)
    registry.invalidate_index(settings)
    return stats


def sync_neo4j(settings: Settings, store: NetworkXGraphStore, registry: ComponentRegistry) -> None:
    if not registry.secrets.neo4j_password:
        raise ValueError("NEO4J_PASSWORD is required to sync the graph to Neo4j")
    neo = Neo4jGraphStore(
        settings.graph.neo4j_uri, settings.graph.neo4j_user, registry.secrets.neo4j_password
    )
    try:
        neo.clear()
        neo.add_entities(store.entities())
        neo.add_relations(store.relations())
        log.info("graph synced to neo4j", extra={"nodes": len(store.entities())})
    finally:
        neo.close()


def ingest_pdf(
    settings: Settings, pdf_path: Path, metadata: dict[str, Any], registry: ComponentRegistry
) -> dict[str, Any]:
    """Incrementally add one paper: parse, chunk, embed, upsert, extend the graph (rule-based)."""
    from agvrag.graph.builder import build_graph as _bg

    parser = get_parser(
        settings.ingestion.parser,
        settings.ingestion.min_section_chars,
        settings.ingestion.max_heading_words,
    )
    doc = parser.parse(pdf_path, metadata)
    write_json(settings.path("documents_dir") / f"{doc.doc_id}.json", doc.to_dict(), indent=None)
    embedder = registry.embedder(settings)
    new_chunks = chunk_documents(settings, [doc], lambda: embedder)
    idx = index_dir(settings)
    existing = (
        [Chunk.from_dict(d) for d in read_jsonl(idx / "chunks.jsonl")]
        if (idx / "chunks.jsonl").exists()
        else []
    )
    existing = [c for c in existing if c.doc_id != doc.doc_id]
    vecs = embedder.embed_documents([chunk_embed_text(c) for c in new_chunks])
    bundle_store = (
        registry.index(settings).vector_store
        if (idx / "chunks.jsonl").exists()
        else open_vector_store(settings, embedder.dim)
    )
    bundle_store.upsert(new_chunks, np.asarray(vecs, dtype=np.float32))
    if isinstance(bundle_store, InMemoryVectorStore):
        bundle_store.save(idx)
    all_chunks = existing + new_chunks
    write_jsonl(idx / "chunks.jsonl", (c.to_dict() for c in all_chunks))
    graph_path = idx / "graph.json"
    n_rel = 0
    if graph_path.exists():
        g = NetworkXGraphStore.load(graph_path)
        docs = load_documents(settings)
        sub, _ = _bg(
            [doc],
            new_chunks,
            registry.gazetteer(settings),
            "rule",
            None,
            settings.graph.min_edge_confidence,
        )
        # Citation resolution needs the full corpus title list: re-run for this doc.
        from agvrag.graph.extraction import RuleExtractor
        from agvrag.graph.resolution import EntityResolver

        rex = RuleExtractor(
            registry.gazetteer(settings),
            EntityResolver(registry.gazetteer(settings)),
            {d.doc_id: d.title for d in docs},
        )
        cites = [r for r in rex.extract(doc, new_chunks).relations if r.type == "CITES"]
        g.add_entities(sub.entities())
        g.add_relations(sub.relations())
        g.add_relations([r for r in cites if g.get_entity(r.target)])
        g.save(graph_path)
        n_rel = len(sub.relations()) + len(cites)
    registry.invalidate_index(settings)
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "sections": len(doc.sections),
        "chunks": len(new_chunks),
        "graph_relations_added": n_rel,
    }


def corpus_dir(settings: Settings) -> Path:
    return settings.path("raw_dir")
