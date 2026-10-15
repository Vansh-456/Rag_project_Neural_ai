# PROJECT_STATE

## Current phase
Phases 0–18 built end to end (2026-10-14/15) — status: **v1 complete; evaluation labels are draft
(single annotator)**. Next work is evaluation hardening, not new components.

## Environment
Windows 11 · Python 3.11 · CPU only (12 threads, 16 GB) · Docker installed (daemon off during build;
experiments used embedded Qdrant + NetworkX) · no LLM API keys → extractive generator + rule
extractor · LLM budget assumed $0.

## Completed
- Phase 0: problem definition, 79-paper corpus (all arXiv titles verified), targets — `docs/problem_definition.md`
- Phase 1: repo, config system (YAML `extends` + env overrides), Protocols, logging, tests, CI, compose
- Phase 2: layout-aware PyMuPDF parser; 79/79 parsed, 100% with sections (was 86% before two parser fixes)
- Phase 3: four chunkers (fixed, recursive, semantic, structure-aware) + sweep command
- Phase 4–5: bge-small embedder + Qdrant; 71-query eval set with anchor labels; metrics + bootstrap CIs; registry
- Phase 6–7: scratch BM25, RRF + weighted fusion, cross-encoder rerank
- Phase 8–9: rule-based KG (680 nodes / 2.7k edges, 100% traceable, audit precision 0.78), entity linking, multi-hop paths
- Phase 10–11: graph+vector fusion, multi-hop graph→vector, rule router (spec table + dev-tuned table), oracle
- Phase 12–15: context builder, extractive + LLM generators with citation validation, verifier, bounded retries, budget
- Phase 16: dev + frozen-test ablation over 9 systems — `docs/ablation_report.md`
- Phase 17: FastAPI service, Next.js UI, Docker, auth, rate limiting, caching, traces, load test script
- Phase 18: docs + paper-style write-up

## Key decisions (see DECISIONS.md)
- Label-based (anchor) ground truth for chunker-independent evaluation — ADR-004
- Default embedder bge-small (CPU cost) — ADR-006
- Graph paths travel alongside chunks; graph evidence grounded to source chunks — ADR-010
- Router table re-fit on dev → `adaptive_tuned`; spec table kept as `adaptive` — ADR-015
- "we propose X" extraction pattern removed after 0/17 audit precision — ADR-016

## Current best configs
- chunking: structure_aware, 256 tokens, 32 overlap (sweep results in docs/experiments.md)
- embedder: BAAI/bge-small-en-v1.5 (query instruction prefix)
- retriever: hybrid (dense + scratch BM25, RRF k=60); best quality with cross-encoder rerank
- reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 over top-50
- router: `adaptive_tuned`

## Latest metrics
See `docs/ablation_report.md` (test) and `results/ablation_report_dev.md` (dev).

## Open problems / failure categories
- "right paper, wrong passage" dominates misses → partly unlabelled relevant passages (pooling needed)
- multi-hop chains lose to direct edges (hop decay)
- frequency-based USES edges ~55% precise; APPLIED_TO weak
- reranking is the quality win but costs ~5 s/query on CPU

## Next steps (max 5)
1. Second-annotator pass + pooled relabelling; report kappa
2. Run LLM generation + calibrated LLM judge (needs API budget)
3. LLM-assisted graph extraction on a budget; re-audit
4. Learned router on dev oracle labels
5. Uncertainty-gated reranking to cut latency

## Known issues / tech debt
- One test-split run executed by accident against the pre-fix graph (results archived unviewed in `results/superseded/`)
- mypy needs ~5 min (analyses torch); consider `follow_imports = skip` for heavy deps
