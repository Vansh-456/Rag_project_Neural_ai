# Adaptive Graph + Vector RAG

A **measured, inspectable** retrieval-augmented generation system that treats retrieval as a
decision problem: for each query a router chooses between dense, lexical, hybrid, knowledge-graph,
graph+vector and multi-hop (graph → vector) retrieval, and every stage — routing rationale, each
retriever's ranked list, graph paths, reranking, citations, claim verification, latency and cost —
is exposed in a trace, an API and a UI.

Built over a 79-paper RAG/LLM research corpus with a 71-query evaluation set covering
definition, relationship, comparison, multi-hop and experimental questions (+ unanswerable ones).

> **Headline (frozen test split):** fusing dense, BM25 and graph-grounded evidence reaches
> NDCG@5 **0.766** vs **0.621** for dense-only (p = 0.003); rule-based adaptive routing does **not**
> beat always-hybrid. Every number is reproducible with `scripts/reproduce.sh`.

## Results

Frozen **test** split, 33 answerable queries, NDCG@5 with 95% bootstrap CI; Δ = paired-bootstrap
difference vs dense-only. Full report: [`docs/ablation_report.md`](docs/ablation_report.md) ·
dev results: [`results/ablation_report_dev.md`](results/ablation_report_dev.md).

| System | NDCG@5 | Recall@5 | Hit@5 | Δ vs dense (p) | p50 latency (CPU) |
|---|---|---|---|---|---|
| dense only (bge-small) | 0.621 [0.49, 0.75] | 0.725 | 0.758 | – | 52 ms |
| BM25 only | 0.678 [0.55, 0.80] | 0.795 | 0.818 | +0.057 (0.35) | 11 ms |
| hybrid (RRF) | 0.683 [0.55, 0.82] | 0.773 | 0.788 | +0.062 (0.06) | 62 ms |
| hybrid + cross-encoder rerank | 0.661 [0.53, 0.78] | 0.785 | 0.849 | +0.040 (0.52) | 5.0 s |
| graph only | 0.522 [0.38, 0.67] | 0.621 | 0.667 | −0.099 (0.37) | 118 ms |
| **graph + vector (default)** | **0.766 [0.65, 0.88]** | **0.851** | **0.909** | **+0.144 (0.003)** | 186 ms |
| adaptive router (spec rules) | 0.684 [0.56, 0.81] | 0.780 | 0.818 | +0.062 (0.13) | 150 ms |
| adaptive router (dev-tuned) | 0.700 [0.58, 0.83] | 0.821 | 0.879 | +0.079 (0.10) | 175 ms |
| per-query oracle (ceiling) | 0.885 | – | – | – | – |

What the numbers say:

* **Graph evidence helps when fused with text retrieval** — graph+vector is the only system
  significantly better than dense-only overall; graph-only wins relationship questions (path entity
  recall 0.97) but fails on definitions and experimental results.
* **The adaptive router does not beat always-hybrid** (+0.017, p = 0.70) and costs more — the central
  hypothesis is not supported by rule-based routing, though the oracle shows real headroom (0.885).
* **Dev ≠ test at this size**: reranking was best on dev (0.584) and below hybrid on test; per-type
  differences without a significance star are noise.
* Graph audit precision 0.78 (50 random edges); parser identified sections in 79/79 papers.


## Why this exists

Most RAG demos pick one retrieval method and judge it by eyeballing answers. This project asks a
research question instead — *can a system choose and combine semantic and relational retrieval per
query to get better evidence at controlled cost?* — and answers it with controlled ablations,
per-query-type breakdowns, significance tests and honest negative results.

## Architecture

```
query ─► analyzer (intent, linked entities, relation cues, hops) ─► router ─► retriever
        vector | bm25 | hybrid (RRF) | graph (multi-hop paths) | graph+vector | multi-hop (graph→vector)
     ─► [cross-encoder rerank] ─► context builder (dedup, token budget, ordering)
     ─► [generator: cited answer or abstain] ─► [claim verifier ─► bounded re-retrieval]
every stage ─► PipelineTrace (lists, scores, paths, per-stage latency, tokens, cost, retries)
```

* **No retrieval frameworks.** BM25, RRF, graph traversal, routing, budgets and evaluation are
  implemented directly (`src/agvrag/`) on sentence-transformers, qdrant-client, networkx/neo4j.
* **Pipelines are configs.** `configs/experiments/*.yaml` — vector_only, bm25_only, hybrid,
  hybrid_rerank, graph_only, graph_vector, multi_hop, adaptive, adaptive_tuned, adaptive_budget,
  adaptive_verified — all over the same index, graph and evaluation code.
* **Graph with provenance.** 680 entities / 2.7k typed relations (INTRODUCES, AUTHORED_BY, CITES,
  USES, EVALUATES_ON, IMPROVES, …) extracted by rules + a seed alias table; every edge traces to a
  source chunk or reference string; manual 50-edge audit precision **0.78**. LLM extraction
  (JSON-schema validated) is a config switch.
* **Chunking-independent ground truth.** Labels are verbatim anchor phrases, so chunkers are
  compared fairly (`docs/labeling_guideline.md`).

Details: [`docs/architecture.md`](docs/architecture.md) · decisions: [`DECISIONS.md`](DECISIONS.md) ·
concepts/study notes: [`docs/concepts.md`](docs/concepts.md).

## Quickstart

**Docker (full stack: API + UI + Qdrant + Neo4j)**

```bash
cp .env.example .env    # set NEO4J_PASSWORD and API_PASSWORD
docker compose up -d --build
# first start downloads + indexes the corpus inside the api container (one-time, CPU: ~1 h)
open http://localhost:3000        # UI      ·  http://localhost:8000/docs  # API
```

**Local (no Docker)**

```bash
uv venv && uv pip install -e ".[ml,dev]"
agvrag bootstrap                         # download 79 papers, parse, index, build graph
agvrag -c adaptive_tuned query "Who introduced LoRA?" --answer
agvrag serve                             # API on :8000
cd frontend && npm ci && npm run dev     # UI on :3000
```

Answers default to a zero-cost **extractive** generator (quotes and cites evidence). For LLM
answers set `ANTHROPIC_API_KEY` and `AGVRAG__GENERATION__PROVIDER=anthropic` (default model
`claude-opus-5-5`), or use `openai` / `ollama`. See [`docs/deployment.md`](docs/deployment.md).

## Reproduce the results

```bash
scripts/reproduce.sh          # download → parse → index → graph → coverage → dev + test ablation → sweep
agvrag -c hybrid evaluate --split dev    # any single system
agvrag oracle --split test               # router vs per-query oracle
```

Each run writes `results/<exp_id>/{config.yaml, git_commit.txt, metrics.json, per_query.csv,
failures.json, traces.jsonl}`. The ablation writes [`docs/ablation_report.md`](docs/ablation_report.md)
and `docs/img/pareto.png`.

## The UI

* **Query** — pipeline selector, router decision (intent, linked entities, complexity, rationale),
  answer with clickable citations (opens the source passage), verification table, graph-path view
  (seeds outlined, top paths highlighted), per-retriever ranked lists with fusion contributions,
  rerank before/after, per-stage latency and cost; **compare mode** runs two pipelines side by side.
* **Experiments** — results registry; per-type metrics with CIs, per-query rows, resolved config.
* **Graph explorer** — entity search, 1–2-hop neighbourhood, every edge with its evidence sentence.

## Repository map

| Path | What |
|---|---|
| `src/agvrag/` | package: ingestion, chunking, embeddings, vector_store, graph, retrieval, routing, ranking, context, generation, verification, pipeline, evaluation, api |
| `configs/` | base / development / production + one YAML per experiment |
| `data/corpus/` | paper list (arXiv ids, title-verified) and gazetteer (seed alias table) |
| `data/evaluation/` | queries, label-based ground truth, expected graph paths (`scripts/build_eval_set.py` is the source) |
| `results/` | experiment registry, parse/graph audits |
| `frontend/` | Next.js + TypeScript UI |
| `docs/` | problem definition, architecture, concepts, experiments, ablation report, failure analysis, limitations, paper write-up, deployment |
| `tests/` | 120+ offline tests (fakes for embedder, LLM, stores) |

## Status and honesty notes

* Evaluation labels are **draft v1, single annotator** (the builder); anchors were written from the
  paper text before any retrieval run, but no second-annotator agreement exists yet. Absolute
  scores are a lower bound (strict passage anchors, shallow pooling); comparisons between systems
  use identical labels.
* All reported experiments ran on CPU with the embedded Qdrant + NetworkX backends; Docker uses
  Qdrant server + Neo4j through the same interfaces.
* No LLM was called for the reported numbers (no API budget at build time). LLM generation,
  extraction and verification are implemented and unit-tested with fakes.

See [`docs/limitations_and_future_work.md`](docs/limitations_and_future_work.md).

## License

MIT
