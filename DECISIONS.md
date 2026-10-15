# Architecture decision records (ADR-lite)

Each entry: context → decision → trade-off → what would make us revisit it. Measured outcomes are
linked to `results/` where they exist.

### ADR-001 — Build all phases in one pass instead of gated sessions (2026-10-14)
* **Context.** The builder prompt prescribes 18 gated phases; the owner asked for an end-to-end,
  deployable build in one go.
* **Decision.** Implement every phase, but keep each phase's *evidence* (parse report, chunk sweep,
  retrieval sweep, ablation, oracle) as separate, reproducible commands and result directories, so
  every gate can still be checked after the fact.
* **Revisit if.** A later phase's results invalidate an earlier default (e.g. the sweep picks a
  different chunker) — then rebuild the index with the winner and re-run the ablation.

### ADR-002 — `src/agvrag/` package layout (2026-10-14)
* The spec shows modules directly under `src/`. A named package (`agvrag`) avoids `import src.*`,
  installs cleanly (`pip install -e .`), and gives a console script (`agvrag`).

### ADR-003 — Deterministic heuristic token counter (2026-10-14)
* **Context.** Chunk sizes are token budgets; a downloaded tokenizer makes chunk ids differ between
  machines/offline containers.
* **Decision.** A rule-based counter calibrated against `cl100k_base` on this corpus (aggregate
  ratio 1.00, mean per-passage error 6.7%). The first version over-counted by ~40%; a unit test
  caught it and the calibration is recorded in `utils/text.py`.
* **Revisit if.** Exact token budgets for a specific LLM matter (then count with that LLM's
  tokenizer at context-building time only).

### ADR-004 — Label-based, chunking-independent ground truth (2026-10-14)
* **Decision.** Relevance = `(doc_id, verbatim anchor phrases, grade)`; a chunk is relevant if it
  is from `doc_id` and contains an anchor. Recall/NDCG count each label once.
* **Trade-off.** Generic anchors make queries easy; anchors split by chunk boundaries silently
  fail — `agvrag coverage` checks every label maps to ≥ 1 chunk for the active chunker.

### ADR-005 — Layout-aware PyMuPDF parser as the default (2026-10-14)
* **Measured.** On 10 papers: 19.9 sections/paper (layout) vs 11.2 (plain); both 100% "sections
  identified". Whole corpus: 79/79 parsed, 100% with sections, mean garbled score 0.014
  (`results/parse_report_pymupdf_layout.json`, `results/parse_comparison.json`).
* Two parser bugs were found on real PDFs and fixed with regression tests: (1) a centred abstract
  on two-column pages was treated as the right column (fixed: blocks straddling the midline are
  full-width); (2) section numbers laid out as a separate text run ("1" | "Introduction") were
  dropped as page numbers (fixed: page-number filtering only in page margins + merge of
  number-only lines with the following heading on the same baseline). Before the fixes 86% of
  papers had sections identified.
* GROBID was not evaluated (requires a Java service); listed as future work.

### ADR-006 — `bge-small-en-v1.5` as the default embedder (2026-10-14)
* **Context.** CPU only; embedding 6.4k chunks takes ~20-30 min on this machine.
* **Decision.** Small 384-d model with its query instruction prefix; larger models are config
  changes. The retrieval sweep compares against `all-MiniLM-L6-v2` (results in `docs/experiments.md`).

### ADR-007 — Embedded Qdrant locally, Qdrant server in Docker (2026-10-14)
* The same `QdrantVectorStore` code runs in-process (`path=`) for development/experiments and
  against the server (`url=`) in production. The Docker daemon was not running during the build, so
  all reported experiments used embedded Qdrant; results are identical by construction (exact
  search at this scale).

### ADR-008 — Rule-based graph extraction first, LLM extraction behind the same interface (2026-10-14)
* **Context.** No LLM API keys / budget at build time.
* **Decision.** Gazetteer (seed alias table) + cue patterns + frequency rules + reference-list
  citation resolution. `introduced_in` priors only produce an edge when grounded in that paper's
  own text. Every text-derived edge carries source chunk ids; metadata/reference edges carry an
  `origin`. `LLMExtractor` (JSON-schema validated, hallucinated-evidence guard) is implemented and
  tested with a fake LLM; enable with `agvrag graph --extractor rule+llm`.
* **Trade-off.** High precision on what the gazetteer covers, low recall outside it (new method
  names not in the gazetteer are invisible to the graph).

### ADR-009 — Entity resolution: conservative merges (2026-10-14)
* Over-merging creates false paths, which is worse than fragmentation. Automatic merges need
  string similarity ≥ 0.92; 0.82–0.92 goes to an optional LLM adjudicator; acronym matching requires
  a single 3–10 character token. A bug where multi-word phrases were treated as acronyms
  ("Natural Quest" → "Natural Questions") was caught by a test and fixed.

### ADR-010 — Graph paths travel alongside chunks (2026-10-14)
* Fused rankings are over chunks (so text and graph retrieval are comparable on chunk metrics);
  graph paths are interleaved as separate evidence and do not consume the chunk budget `k`. Graph
  path evidence is grounded: its `source_chunk_ids` are the chunks its edges were extracted from.

### ADR-011 — RRF as default fusion (2026-10-14)
* Scale-free; no score calibration between cosine and BM25. Weighted min-max fusion is available
  (`hybrid_weighted`) for comparison.

### ADR-012 — Extractive generator as the zero-cost default (2026-10-14)
* Quote-and-cite generator: faithful by construction, abstains when no sentence covers enough of
  the query. It is a floor for answer quality, not a substitute for an LLM. The grounded LLM
  prompt (`grounded_v1`) forces `[E#]`/`[G#]` citations and abstention; the default Claude model
  is `claude-opus-5-5` with effort `medium` and server-side refusal fallback enabled.

### ADR-013 — API applies experiment configs as overlays (2026-10-14)
* A bug surfaced in tests: rebuilding per-pipeline settings from YAML dropped deployment settings
  (embedder, chunking, backends). Pipelines are now `deployment settings + experiment overlay`,
  with deployment-owned sections excluded from the overlay.

### ADR-014 — Budgets raise before acting; retries are bounded and traced (2026-10-14)
* `BudgetTracker.charge` raises `BudgetExceeded` before the action would exceed a cap. The verify
  loop is bounded by `verification.max_iterations` *and* the budget; each retry is appended to
  `PipelineTrace.iterations`.

### ADR-015 — Re-fit the router's intent table on dev; keep the spec rules as a separate system (2026-10-15)
* **Measured (dev).** BM25 beats dense on this acronym-heavy corpus (NDCG@5 0.52 vs 0.36,
  p = 0.011); dense-only was the *worst* strategy for definitions, yet the spec rule routes
  definitions to `vector`. Experimental questions did best with hybrid + rerank.
* **Decision.** Make `router.intent_strategy` and `router.rerank_intents` YAML-configurable.
  `adaptive` keeps the spec's table; `adaptive_tuned` uses the dev-fit table
  (definition→hybrid, relationship→graph_vector, comparison→graph_vector, multi_hop→multi_hop,
  experimental→hybrid+rerank). Both are reported on the frozen test split.
* **Revisit if.** The test split disagrees with dev (it would mean the table over-fit 34 queries).

### ADR-016 — Remove the "we propose X" extraction pattern; nearest-mention window for cue relations (2026-10-15)
* **Measured.** v1 edge audit precision 0.74 (50 random edges). A full review of all 94 INTRODUCES
  edges: 77/77 gazetteer-grounded edges correct, 0/17 "propose-pattern" edges correct. IMPROVES 1/4
  in the sample — the cue rule linked every later mention in the sentence.
* **Decision.** Disable the propose pattern (kept behind `use_propose_pattern=False`); cue relations
  take only the nearest mention within 80 chars before / 60 chars after the cue, one object.
* **Result.** v2 audit precision 0.78 on a fresh 50-edge sample; IMPROVES 4/4; INTRODUCES 77/77.
  Remaining weak spots: frequency-based USES (0.55) and APPLIED_TO (1/3) — documented, not fixed.
* **Process note.** These fixes were driven by the graph audit, not by evaluation queries; the
  test split had not been run when they were made.
