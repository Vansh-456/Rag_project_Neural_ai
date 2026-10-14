"""
FastAPI application for Adaptive Graph + Vector RAG.
Exposes endpoints for querying, pipeline tracing, document ingestion,
graph visualization, and side-by-side pipeline comparisons.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.schemas import (
    CompareRequest,
    Document,
    EntityType,
    GraphEntity,
    GraphRelation,
    HealthResponse,
    PipelineTrace,
    QueryRequest,
    QueryResponse,
    RelationType,
    Section,
)
from src.config import load_config
from src.embeddings.embedder import get_embedder
from src.vector_store.store import get_vector_store
from src.retrieval.sparse import BM25Index
from src.graph.store import get_graph_store
from src.graph.extraction import EntityExtractor
from src.chunking.chunker import get_chunker
from src.generation.client import get_llm_client
from src.pipeline.rag_pipeline import UnifiedRAGPipeline
from src.evaluation.evaluator import BenchmarkEvaluator


# In-memory registry for traces
TRACE_STORE: dict[str, PipelineTrace] = {}

# Global pipeline instance
pipeline_instance: UnifiedRAGPipeline | None = None
benchmark_evaluator: BenchmarkEvaluator | None = None


def init_system() -> UnifiedRAGPipeline:
    """Initializes and seeds all pipeline stores and components."""
    settings = load_config()

    embedder = get_embedder(
        provider=settings.embeddings.provider,
        model_name=settings.embeddings.model_name,
        dimension=settings.embeddings.dimension,
        cache_dir=settings.embeddings.cache_dir
    )

    vector_store = get_vector_store(
        backend=settings.vector_store.backend,
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        collection_name=settings.vector_store.collection_name,
        dimension=settings.embeddings.dimension
    )

    sparse_index = BM25Index(
        k1=settings.sparse_retrieval.k1,
        b=settings.sparse_retrieval.b,
        lowercase=settings.sparse_retrieval.lowercase,
        remove_stopwords=settings.sparse_retrieval.remove_stopwords
    )

    graph_store = get_graph_store(
        backend=settings.graph.backend,
        neo4j_uri=settings.graph.neo4j_uri,
        neo4j_user=settings.graph.neo4j_user,
        neo4j_password=settings.graph.neo4j_password
    )

    llm_client = get_llm_client(
        provider=settings.generation.provider,
        openai_key=settings.openai_api_key,
        anthropic_key=settings.anthropic_api_key,
        ollama_url=settings.ollama_base_url,
        model_name=settings.generation.model_name
    )

    # Ingest existing seed papers if available
    corpus_file = Path(settings.corpus.documents_file)
    if corpus_file.exists():
        with open(corpus_file, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        chunker = get_chunker(
            strategy=settings.chunking.strategy,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap
        )
        extractor = EntityExtractor()

        all_chunks = []
        for d in raw_docs:
            doc = Document(
                doc_id=d["doc_id"],
                title=d["title"],
                authors=d.get("authors", []),
                year=d.get("year"),
                sections=[Section(heading=s["heading"], level=s["level"], text=s["text"]) for s in d.get("sections", [])],
                references=d.get("references", []),
                source_path=d.get("source_path", ""),
                abstract=d.get("abstract", "")
            )
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)

            # Add Paper node
            paper_entity = GraphEntity(
                entity_id=doc.doc_id,
                name=doc.title,
                entity_type=EntityType.PAPER,
                aliases=[doc.doc_id, doc.title.lower()],
                description=doc.abstract,
                source_doc_ids=[doc.doc_id]
            )
            graph_store.add_entity(paper_entity)

            # Add Author nodes and AUTHORED_BY relations
            for author in doc.authors:
                cleaned_author = author.lower().replace(".", "").replace(" ", "_").strip()
                author_id = f"ent_author_{cleaned_author}"
                author_entity = GraphEntity(
                    entity_id=author_id,
                    name=author,
                    entity_type=EntityType.AUTHOR,
                    aliases=[author.lower(), author.split()[-1].lower()],
                    description=f"Author of {doc.title}",
                    source_doc_ids=[doc.doc_id]
                )
                graph_store.add_entity(author_entity)
                graph_store.add_relation(GraphRelation(
                    source_id=author_id,
                    target_id=doc.doc_id,
                    relation_type=RelationType.AUTHORED_BY,
                    source_doc_ids=[doc.doc_id],
                    confidence=1.0
                ))

            # Add CITES relations
            for ref in doc.references:
                graph_store.add_relation(GraphRelation(
                    source_id=doc.doc_id,
                    target_id=ref,
                    relation_type=RelationType.CITES,
                    source_doc_ids=[doc.doc_id],
                    confidence=0.9
                ))

            # Extract graph entities and relations from chunks
            for c in chunks:
                ents, rels = extractor.extract_from_chunk(c)
                for ent in ents:
                    graph_store.add_entity(ent)
                    # Connect Paper to Method/Model it introduces
                    if "lora" in ent.entity_id and "lora" in doc.doc_id:
                        graph_store.add_relation(GraphRelation(
                            source_id=doc.doc_id,
                            target_id=ent.entity_id,
                            relation_type=RelationType.INTRODUCES,
                            source_doc_ids=[doc.doc_id],
                            confidence=1.0
                        ))
                    elif "bert" in ent.entity_id and "bert" in doc.doc_id:
                        graph_store.add_relation(GraphRelation(
                            source_id=doc.doc_id,
                            target_id=ent.entity_id,
                            relation_type=RelationType.INTRODUCES,
                            source_doc_ids=[doc.doc_id],
                            confidence=1.0
                        ))
                    elif "dpr" in ent.entity_id and "dpr" in doc.doc_id:
                        graph_store.add_relation(GraphRelation(
                            source_id=doc.doc_id,
                            target_id=ent.entity_id,
                            relation_type=RelationType.INTRODUCES,
                            source_doc_ids=[doc.doc_id],
                            confidence=1.0
                        ))
                    elif "fid" in ent.entity_id and "fid" in doc.doc_id:
                        graph_store.add_relation(GraphRelation(
                            source_id=doc.doc_id,
                            target_id=ent.entity_id,
                            relation_type=RelationType.INTRODUCES,
                            source_doc_ids=[doc.doc_id],
                            confidence=1.0
                        ))
                    elif "attention" in doc.doc_id and "transformer" in ent.entity_id:
                        graph_store.add_relation(GraphRelation(
                            source_id=doc.doc_id,
                            target_id=ent.entity_id,
                            relation_type=RelationType.INTRODUCES,
                            source_doc_ids=[doc.doc_id],
                            confidence=1.0
                        ))

                for rel in rels:
                    graph_store.add_relation(rel)

        if all_chunks:
            # Index into BM25
            sparse_index.index_chunks(all_chunks)
            # Embed and index into vector store
            texts_to_embed = [c.text for c in all_chunks]
            embeddings = embedder.embed_texts(texts_to_embed)
            vector_store.index_chunks(all_chunks, embeddings)

    pipeline = UnifiedRAGPipeline(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        sparse_index=sparse_index,
        graph_store=graph_store,
        llm_client=llm_client
    )
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, benchmark_evaluator
    pipeline_instance = init_system()
    benchmark_evaluator = BenchmarkEvaluator(pipeline_instance)
    yield


app = FastAPI(
    title="Adaptive Graph + Vector RAG API",
    description="Inspectable, portfolio-grade Adaptive Graph + Vector RAG research system.",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """Serves the interactive research dashboard."""
    index_file = Path(__file__).parent / "static" / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Adaptive Graph + Vector RAG API Online</h1>")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check and corpus statistics."""
    if not pipeline_instance:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialized")

    graph_stats = pipeline_instance.graph_store.get_graph_statistics()
    chunk_count = len(pipeline_instance.sparse_index.chunks)

    return HealthResponse(
        status="healthy",
        vector_store_status="ready",
        graph_store_status="ready",
        document_count=len(set(c.doc_id for c in pipeline_instance.sparse_index.chunks)),
        chunk_count=chunk_count,
        entity_count=graph_stats.get("node_count", 0),
        relation_count=graph_stats.get("edge_count", 0)
    )


@app.post("/query", response_model=QueryResponse)
def execute_query(req: QueryRequest):
    """Executes a query through the specified pipeline and records full trace."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    trace = pipeline_instance.run(
        query=req.query,
        pipeline_mode=req.pipeline,
        top_k=req.top_k
    )

    trace_id = str(uuid.uuid4())
    TRACE_STORE[trace_id] = trace

    evidence_formatted = [
        {
            "id": ev.evidence_id,
            "type": ev.type.value,
            "retriever": ev.retriever,
            "score": ev.raw_score,
            "rerank_score": ev.rerank_score,
            "content": ev.content,
            "doc_title": ev.provenance.get("doc_title", ""),
            "page": ev.provenance.get("page_start")
        }
        for ev in trace.reranked
    ]

    return QueryResponse(
        trace_id=trace_id,
        query=req.query,
        answer=trace.answer or "",
        pipeline=req.pipeline,
        strategy_chosen=trace.routing.strategy,
        citations=trace.citations,
        evidence_used=evidence_formatted,
        verification=trace.verification,
        metrics=trace.costs
    )


@app.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    """Retrieves full execution trace including timings, prompt context, and routing decisions."""
    if trace_id not in TRACE_STORE:
        raise HTTPException(status_code=404, detail="Trace ID not found")

    trace = TRACE_STORE[trace_id]
    return {
        "trace_id": trace_id,
        "query_analysis": {
            "query": trace.query_analysis.query,
            "intent": trace.query_analysis.intent,
            "entities": trace.query_analysis.entities,
            "complexity": trace.query_analysis.complexity,
            "hops_estimate": trace.query_analysis.hops_estimate,
            "rationale": trace.query_analysis.rationale
        },
        "routing": {
            "strategy": trace.routing.strategy,
            "params": trace.routing.params,
            "rationale": trace.routing.rationale
        },
        "retrieved_count": len(trace.retrieved),
        "reranked_count": len(trace.reranked),
        "retrieved_evidence": [
            {
                "id": e.evidence_id,
                "type": e.type.value,
                "retriever": e.retriever,
                "score": e.raw_score,
                "content": e.content,
                "provenance": e.provenance
            }
            for e in trace.retrieved
        ],
        "context_prompt": trace.context,
        "answer": trace.answer,
        "citations": trace.citations,
        "verification": trace.verification,
        "costs": trace.costs
    }


@app.post("/compare")
def compare_pipelines(req: CompareRequest):
    """Runs a query through two different pipelines for side-by-side inspection."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    trace_a = pipeline_instance.run(query=req.query, pipeline_mode=req.pipeline_a, top_k=req.top_k)
    trace_b = pipeline_instance.run(query=req.query, pipeline_mode=req.pipeline_b, top_k=req.top_k)

    return {
        "query": req.query,
        "pipeline_a": {
            "name": req.pipeline_a,
            "strategy": trace_a.routing.strategy,
            "answer": trace_a.answer,
            "citations": trace_a.citations,
            "latency_ms": trace_a.costs.get("total_latency_ms", 0.0),
            "evidence_count": len(trace_a.reranked),
            "verification": trace_a.verification
        },
        "pipeline_b": {
            "name": req.pipeline_b,
            "strategy": trace_b.routing.strategy,
            "answer": trace_b.answer,
            "citations": trace_b.citations,
            "latency_ms": trace_b.costs.get("total_latency_ms", 0.0),
            "evidence_count": len(trace_b.reranked),
            "verification": trace_b.verification
        }
    }


@app.get("/graph/visualization")
def get_graph_visualization():
    """Returns nodes and links for interactive graph visualization in the UI."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    return pipeline_instance.graph_store.export_for_visualization()


@app.post("/evaluate")
def run_evaluation(pipeline_mode: str = "adaptive", top_k: int = 5):
    """Executes benchmark evaluation across all 50 evaluation queries."""
    if not benchmark_evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not ready")

    results = benchmark_evaluator.evaluate_queries(pipeline_mode=pipeline_mode, top_k=top_k)
    return results


@app.get("/experiments")
def list_experiments():
    """Lists available experiment configurations."""
    exp_dir = Path("configs/experiments")
    if not exp_dir.exists():
        return []

    configs = []
    for p in exp_dir.glob("*.yaml"):
        configs.append({
            "name": p.stem,
            "path": str(p)
        })
    return configs
