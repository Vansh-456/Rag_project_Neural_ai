# Deployment

## Option A — Docker Compose (full stack: API + Qdrant + Neo4j + UI)

```bash
cp .env.example .env            # set NEO4J_PASSWORD and API_PASSWORD (required); LLM keys optional
docker compose up -d --build    # first start bootstraps corpus+index+graph inside the api container
docker compose logs -f api      # watch the bootstrap (~40-60 min on a 12-core CPU; one-time)
```

* UI: http://localhost:3000 · API docs: http://localhost:8000/docs · Neo4j browser: http://localhost:7474
* `AGVRAG_ENV=production` layers `configs/production.yaml`: Qdrant server, Neo4j graph backend,
  JSON logs, API basic auth on.
* The bootstrap downloads 79 arXiv PDFs (rate-limited to 1 request / 3 s), parses, chunks, embeds
  and builds the graph, then syncs it to Neo4j. Set `AUTO_BOOTSTRAP=false` and run
  `docker compose run --rm api agvrag bootstrap` if you prefer to do it explicitly.
* Data lives in named volumes (`agv_data`, `qdrant_data`, `neo4j_data`); rebuilding images does not
  re-embed.

### Using an LLM for answers

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
AGVRAG__GENERATION__PROVIDER=anthropic
AGVRAG__GENERATION__MODEL=claude-opus-5-5     # default when model is empty
AGVRAG__GENERATION__EFFORT=medium
```

Other providers: `openai` (needs `OPENAI_API_KEY`) or `ollama` (`AGVRAG__GENERATION__OLLAMA_URL`).
LLM responses are cached in `data/cache/llm_cache.sqlite`; hit rate is exposed at `/metrics`.
On Amazon Bedrock / Vertex / Foundry set `AGVRAG__GENERATION__ANTHROPIC_FALLBACKS=false` (server-side
refusal fallback is a Claude API feature).

### LLM-based graph extraction (optional, costs tokens)

```bash
docker compose run --rm api agvrag graph --extractor rule+llm
```
Bounded by `graph.max_llm_extraction_chunks` (default 400 chunks); token usage is written to
`graph_report.json`.

## Option B — Single process, no Docker

```bash
uv venv && uv pip install -e ".[ml,dev]"      # or: pip install -e ".[ml,dev]"
agvrag bootstrap                              # embedded Qdrant + NetworkX graph under data/
agvrag serve                                  # API on :8000
cd frontend && npm ci && API_URL=http://localhost:8000 npm run dev   # UI on :3000
```

## Hardening checklist

| Item | Where |
|---|---|
| API basic auth (constant-time compare); `/health` stays open for probes | `api.auth_enabled`, `API_USERNAME`/`API_PASSWORD` |
| Credentials never reach the browser — the Next.js server proxy adds them | `frontend/src/app/api/[...path]/route.ts` |
| Optional UI basic auth | `UI_BASIC_AUTH_USER` / `UI_BASIC_AUTH_PASSWORD` |
| Per-client sliding-window rate limit (429 + `Retry-After`) | `api.rate_limit_per_minute` |
| Upload limits: PDF magic bytes, size cap, sanitised file names | `/ingest`, `api.max_upload_mb` |
| Non-root containers, health checks, restart policy | `docker/*.Dockerfile`, `docker-compose.yml` |
| CORS allow-list | `api.cors_origins` |
| Secrets only via env / `.env` (gitignored) | `Secrets` in `config.py` |

Put a TLS-terminating reverse proxy (Caddy, nginx, a cloud load balancer) in front of port 3000
for anything internet-facing; the stack itself speaks plain HTTP.

## Operations

* `GET /metrics` — request counts, p50/p95 latency per route, result/embedding/LLM cache hit rates.
* `GET /traces` and `GET /trace/{id}` — every query is persisted (SQLite) with its full trace.
* `python scripts/load_test.py --url http://localhost:8000 --n 200 --concurrency 8` — small load test.
* Re-indexing with a new chunker/embedder creates a *new* index directory/collection (the name
  encodes the config), so the running index is never mutated in place.
