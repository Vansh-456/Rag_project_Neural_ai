"""FastAPI service exposing the pipeline *and its internals*.

Endpoints (OpenAPI docs at ``/docs``):
  GET  /health                 liveness + index status (no auth)
  GET  /pipelines              available pipeline configs
  POST /query                  run a query -> full PipelineTrace (persisted)
  POST /query/stream           Server-Sent Events: analysis, routing, evidence, tokens, done
  POST /compare                same query through two pipelines
  GET  /trace/{id}, /traces    persisted traces
  POST /ingest                 upload a PDF (or give an arXiv id) and add it to the index
  GET  /experiments[/{id}]     results registry
  GET  /graph/stats, /graph/search, /graph/neighborhood/{entity_id}
  GET  /chunks/{chunk_id}      source passage (for clickable citations)
  GET  /metrics                request latency, cache hit rates

Pipelines are loaded lazily and share heavy components through ``ComponentRegistry``.
Pipeline execution is CPU-bound, so handlers run it in the threadpool.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from agvrag import __version__
from agvrag.api.support import LRUCache, RateLimiter, RequestMetrics, TraceStore
from agvrag.config import Secrets, Settings, apply_overlay, experiment_overlay, load_settings
from agvrag.pipeline.components import ComponentRegistry, IndexNotBuiltError, index_dir
from agvrag.pipeline.pipeline import RAGPipeline
from agvrag.utils.io import read_json
from agvrag.utils.logging import get_logger, setup_logging

log = get_logger("api")


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    pipeline: str | None = None
    generate: bool | None = None
    verify: bool | None = None


class CompareRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    pipelines: list[str] = Field(min_length=2, max_length=2)
    generate: bool | None = None


class IngestArxivRequest(BaseModel):
    arxiv_id: str = Field(pattern=r"^\d{4}\.\d{4,5}$")


class AppState:
    def __init__(self, settings: Settings, secrets_: Secrets) -> None:
        self.settings = settings
        self.secrets = secrets_
        self.registry = ComponentRegistry(secrets_)
        self.pipelines: dict[str, RAGPipeline] = {}
        self.lock = threading.Lock()
        self.traces = TraceStore(settings.path("traces_db"))
        self.cache = LRUCache(settings.api.retrieval_cache_size)
        self.limiter = RateLimiter(settings.api.rate_limit_per_minute)
        self.metrics = RequestMetrics()
        self.ingest_lock = threading.Lock()

    def pipeline_settings(self, name: str) -> Settings:
        if name not in self.settings.api.pipelines:
            raise HTTPException(404, f"unknown pipeline '{name}'")
        overlay = experiment_overlay(name)
        # Deployment-level choices always win over what an experiment file says about them.
        for key in ("paths", "vector_store", "graph", "logging", "api", "embedding", "chunking"):
            overlay.pop(key, None)
        return apply_overlay(self.settings, overlay)

    def pipeline(self, name: str) -> RAGPipeline:
        with self.lock:
            if name not in self.pipelines:
                try:
                    self.pipelines[name] = RAGPipeline(self.pipeline_settings(name), self.registry)
                except IndexNotBuiltError as exc:
                    raise HTTPException(503, str(exc)) from exc
            return self.pipelines[name]

    def reset(self) -> None:
        with self.lock:
            self.pipelines.clear()
        self.cache.clear()


def create_app(settings: Settings | None = None, secrets_: Secrets | None = None) -> FastAPI:
    settings = settings or load_settings()
    secrets_ = secrets_ or Secrets()
    setup_logging(settings.logging.level, settings.logging.json_logs)
    if settings.api.auth_enabled and not secrets_.api_password:
        raise RuntimeError("api.auth_enabled is true but API_PASSWORD is not set")
    state = AppState(settings, secrets_)
    app = FastAPI(
        title="Adaptive Graph + Vector RAG",
        version=__version__,
        description="Inspectable adaptive retrieval: router, evidence, graph paths, "
        "citations, verification and costs for every query.",
    )
    app.state.agv = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    basic = HTTPBasic(auto_error=False)

    def auth(creds: HTTPBasicCredentials | None = Depends(basic)) -> None:
        if not settings.api.auth_enabled:
            return
        ok = (
            creds is not None
            and secrets.compare_digest(creds.username, secrets_.api_username)
            and secrets.compare_digest(creds.password, secrets_.api_password or "")
        )
        if not ok:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )

    @app.middleware("http")
    async def limits_and_metrics(request: Request, call_next: Any) -> Any:
        t0 = time.perf_counter()
        route = request.url.path
        if route not in {"/health", "/docs", "/openapi.json"} and request.method != "OPTIONS":
            client = request.headers.get(
                "x-forwarded-for", request.client.host if request.client else "?"
            )
            ok, retry = state.limiter.allow(client.split(",")[0].strip())
            if not ok:
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(int(retry) + 1)},
                )
        response = await call_next(request)
        key = "/" + route.strip("/").split("/")[0] if route != "/" else "/"
        state.metrics.record(key, (time.perf_counter() - t0) * 1000, response.status_code)
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        idx = index_dir(settings)
        return {
            "status": "ok",
            "version": __version__,
            "index": settings.index_name,
            "index_built": (idx / "chunks.jsonl").exists(),
            "graph_built": (idx / "graph.json").exists() or settings.graph.backend == "neo4j",
            "pipelines_loaded": sorted(state.pipelines),
            "llm_provider": settings.generation.provider,
        }

    @app.get("/pipelines", dependencies=[Depends(auth)])
    def pipelines() -> list[dict[str, Any]]:
        out = []
        for name in settings.api.pipelines:
            s = state.pipeline_settings(name)
            out.append(
                {
                    "name": name,
                    "description": s.pipeline.description,
                    "router": s.pipeline.router,
                    "strategy": s.pipeline.strategy,
                    "rerank": s.pipeline.rerank,
                    "generate": s.pipeline.generate,
                    "verify": s.pipeline.verify,
                    "default": name == settings.api.default_pipeline,
                }
            )
        return out

    def _run(req: QueryRequest) -> dict[str, Any]:
        name = req.pipeline or settings.api.default_pipeline
        generate = True if req.generate is None else req.generate
        key = (name, req.query.strip(), generate, req.verify)
        cached = state.cache.get(key)
        if cached is not None:
            return {**cached, "cached": True}
        pipe = state.pipeline(name)
        trace = pipe.run(req.query.strip(), generate=generate, verify=req.verify).to_dict()
        state.traces.save(trace)
        state.cache.put(key, trace)
        return {**trace, "cached": False}

    @app.post("/query", dependencies=[Depends(auth)])
    async def query(req: QueryRequest) -> dict[str, Any]:
        return await run_in_threadpool(_run, req)

    @app.post("/query/stream", dependencies=[Depends(auth)])
    def query_stream(req: QueryRequest) -> StreamingResponse:
        pipe = state.pipeline(req.pipeline or settings.api.default_pipeline)

        def events() -> Iterator[str]:
            try:
                for ev in pipe.stream(req.query.strip()):
                    yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], default=str)}\n\n"
            except Exception as exc:  # surface errors to the client instead of a dropped stream
                log.exception("stream failed")
                yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/compare", dependencies=[Depends(auth)])
    async def compare(req: CompareRequest) -> dict[str, Any]:
        out = {}
        for name in req.pipelines:
            out[name] = await run_in_threadpool(
                _run, QueryRequest(query=req.query, pipeline=name, generate=req.generate)
            )
        return {"query": req.query, "results": out}

    @app.get("/trace/{trace_id}", dependencies=[Depends(auth)])
    def get_trace(trace_id: str) -> dict[str, Any]:
        t = state.traces.get(trace_id)
        if t is None:
            raise HTTPException(404, "trace not found")
        return t

    @app.get("/traces", dependencies=[Depends(auth)])
    def traces(limit: int = 50) -> list[dict[str, Any]]:
        return state.traces.recent(min(500, max(1, limit)))

    def _do_ingest(pdf: Path, meta: dict[str, Any]) -> dict[str, Any]:
        from agvrag.pipeline.indexing import ingest_pdf

        with state.ingest_lock:
            res = ingest_pdf(
                state.pipeline_settings(settings.api.default_pipeline), pdf, meta, state.registry
            )
            state.reset()
            return res

    @app.post("/ingest", dependencies=[Depends(auth)])
    async def ingest(
        file: UploadFile = File(...), title: str = Form(""), doc_id: str = Form("")
    ) -> dict[str, Any]:
        data = await file.read()
        if len(data) > settings.api.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, "file too large")
        if not data.startswith(b"%PDF"):
            raise HTTPException(400, "not a PDF")
        safe_id = "".join(
            ch
            for ch in (doc_id or Path(file.filename or "upload").stem)
            if ch.isalnum() or ch in "._-"
        )[:64]
        raw = settings.path("raw_dir")
        raw.mkdir(parents=True, exist_ok=True)
        pdf = raw / f"{safe_id}.pdf"
        pdf.write_bytes(data)
        meta: dict[str, Any] = {
            "doc_id": safe_id,
            "title": title or None,
            "authors": [],
            "source": "upload",
        }
        try:
            return await run_in_threadpool(_do_ingest, pdf, meta)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/ingest/arxiv", dependencies=[Depends(auth)])
    async def ingest_arxiv(req: IngestArxivRequest) -> dict[str, Any]:
        from agvrag.ingestion.arxiv import ArxivDownloader, PaperSpec

        def work() -> dict[str, Any]:
            dl = ArxivDownloader(settings.path("raw_dir"), settings.ingestion.arxiv_delay_seconds)
            meta = dl.fetch_metadata([req.arxiv_id]).get(req.arxiv_id)
            if meta is None:
                raise HTTPException(404, "arXiv id not found")
            rep = dl.download([PaperSpec(req.arxiv_id, meta["title"])])
            if rep["failed"]:
                raise HTTPException(502, f"download failed: {rep['failed']}")
            meta = read_json(settings.path("raw_dir") / f"{req.arxiv_id}.json")
            return _do_ingest(settings.path("raw_dir") / f"{req.arxiv_id}.pdf", meta)

        return await run_in_threadpool(work)

    @app.get("/experiments", dependencies=[Depends(auth)])
    def experiments() -> list[dict[str, Any]]:
        from agvrag.evaluation.runner import list_experiments

        return list_experiments(settings.path("results_dir"))

    @app.get("/experiments/{exp_id}", dependencies=[Depends(auth)])
    def experiment(exp_id: str) -> dict[str, Any]:
        from agvrag.evaluation.runner import read_per_query

        d = settings.path("results_dir") / exp_id
        if (
            not exp_id.replace("_", "").replace("-", "").replace(".", "").isalnum()
            or not (d / "metrics.json").exists()
        ):
            raise HTTPException(404, "experiment not found")
        failures = read_json(d / "failures.json") if (d / "failures.json").exists() else {}
        return {
            "metrics": read_json(d / "metrics.json"),
            "failures": failures,
            "per_query": read_per_query(settings.path("results_dir"), exp_id),
            "config": (d / "config.yaml").read_text(encoding="utf-8"),
        }

    def _graph() -> Any:
        try:
            return state.registry.graph(state.pipeline_settings("graph_only"))
        except IndexNotBuiltError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/graph/stats", dependencies=[Depends(auth)])
    def graph_stats() -> dict[str, Any]:
        p = index_dir(settings) / "graph_stats.json"
        if not p.exists():
            raise HTTPException(404, "graph statistics not found")
        return read_json(p)

    @app.get("/graph/search", dependencies=[Depends(auth)])
    def graph_search(q: str) -> list[dict[str, Any]]:
        try:
            return state.registry.linker(state.pipeline_settings("graph_only")).link(
                q, max_entities=10
            )
        except IndexNotBuiltError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/graph/neighborhood/{entity_id:path}", dependencies=[Depends(auth)])
    def neighborhood(entity_id: str, hops: int = 1, limit: int = 60) -> dict[str, Any]:
        g = _graph()
        if g.get_entity(entity_id) is None:
            raise HTTPException(404, "entity not found")
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        frontier = [entity_id]
        for _ in range(max(1, min(hops, 2))):
            nxt = []
            for nid in frontier:
                for rel, other, _direction in g.neighbors(nid)[:limit]:
                    edges[rel.key] = {
                        "source": rel.source,
                        "target": rel.target,
                        "type": rel.type,
                        "confidence": rel.confidence,
                        "evidence": rel.evidence_text[:300],
                        "chunks": rel.source_chunk_ids[:3],
                    }
                    if other not in nodes:
                        nxt.append(other)
                    nodes.setdefault(other, {})
                nodes.setdefault(nid, {})
            frontier = nxt[:limit]
        out_nodes = []
        for nid in nodes:
            e = g.get_entity(nid)
            if e:
                out_nodes.append({"id": nid, "name": e.name, "type": e.type})
        return {"center": entity_id, "nodes": out_nodes, "edges": list(edges.values())}

    @app.get("/chunks/{chunk_id:path}", dependencies=[Depends(auth)])
    def chunk(chunk_id: str) -> dict[str, Any]:
        try:
            chunks = state.registry.index(state.pipeline_settings("vector_only")).chunks
        except IndexNotBuiltError as exc:
            raise HTTPException(503, str(exc)) from exc
        c = chunks.get(chunk_id)
        if c is None:
            raise HTTPException(404, "chunk not found")
        return c.to_dict()

    @app.get("/metrics", dependencies=[Depends(auth)])
    def metrics() -> dict[str, Any]:
        emb = state.registry._cache.get(
            next((k for k in state.registry._cache if k[0] == "embedder"), ("",))
        )
        llm = state.registry._cache.get(
            next((k for k in state.registry._cache if k[0] == "llm"), ("",))
        )
        return {
            "requests": state.metrics.snapshot(),
            "cache": {
                "result_cache_hit_rate": round(state.cache.hit_rate, 4),
                "embedding_cache_hit_rate": round(getattr(emb, "hit_rate", 0.0), 4),
                "llm_cache_hit_rate": round(getattr(llm, "hit_rate", 0.0), 4),
            },
            "pipelines_loaded": sorted(state.pipelines),
        }

    return app
