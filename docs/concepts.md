# Concepts (study notes)

Each entry: **what** it is, **why** it exists, **what it costs**, **how it fails**, the **math**, one
**common mistake**, and a **check question**. Code pointers show where the idea lives in this repo.

---

## Chunking — `chunking/chunkers.py`

**What/why.** Retrieval returns units of text; chunking decides that unit. Too small loses context,
too large dilutes the embedding and wastes context-window tokens.
**Cost.** Re-chunking means re-embedding the whole corpus (~20-30 min per config on this CPU).
**Failure.** Fixed windows cut sentences and mix sections; semantic chunking can create tiny chunks
at topic shifts; structure-aware chunking depends on the parser finding headings.
**Math.** Fixed: windows `[i·(s−o), i·(s−o)+s)` over tokens, size `s`, overlap `o`.
**Common mistake.** Comparing chunkers with chunk-id ground truth — the ids change with the chunker.
This repo uses text anchors instead (`docs/labeling_guideline.md`).
**Check.** Why does structure-aware chunking prefix "title > section" to the embedded text?

## Dense retrieval & cosine similarity — `embeddings/`, `vector_store/`

**What/why.** Encode query and passages into vectors; nearest neighbours are semantically similar
even without word overlap ("adapter layers" ~ "bottleneck modules").
**Cost.** One encoder forward pass per passage (offline) and per query (online); ANN index memory.
**Failure.** Rare names, numbers, acronyms and out-of-domain jargon embed poorly.
**Math.** `cos(q, d) = q·d / (‖q‖‖d‖)`; with L2-normalised vectors this is the dot product.
**Common mistake.** Forgetting model-specific instruction prefixes (BGE query instruction, E5
`query:`/`passage:`) — a silent quality loss. `EmbeddingConfig.query_prefix` handles it.
**Check.** Why must the same normalisation be applied to stored and query vectors?

## BM25 — `retrieval/bm25.py`

**What/why.** Lexical scoring that rewards rare query terms, saturates repeated terms and normalises
by document length. Strong for names, acronyms, numbers.
**Cost.** Inverted index; query time proportional to postings of query terms.
**Failure.** Vocabulary mismatch ("fine-tuning" vs "adaptation"); no notion of meaning.
**Math.** `idf(t) = ln(1 + (N − n_t + 0.5)/(n_t + 0.5))`;
`score(q,d) = Σ_t idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))`.
**Common mistake.** Tokenising differently at index and query time (e.g. stemming only one side).
**Check.** What happens to BM25 scores as `b → 0`?

## Reciprocal Rank Fusion — `retrieval/fusion.py`

**What/why.** Combine ranked lists whose scores live on different scales (cosine vs BM25).
**Cost.** Negligible; but you must retrieve from every source.
**Failure.** Rewards agreement; a passage found only by one strong retriever can be out-ranked by
mediocre consensus.
**Math.** `RRF(d) = Σ_r w_r / (k + rank_r(d))`, typically `k = 60`.
**Common mistake.** Fusing raw scores without normalisation (weighted fusion needs min-max or
z-scores per list — implemented as `weighted_fuse`).
**Check.** Why does a larger `k` flatten the difference between rank 1 and rank 5?

## Cross-encoder reranking — `ranking/rerankers.py`

**What/why.** Score each (query, passage) pair jointly with full attention between them; much more
accurate than bi-encoder similarity.
**Cost.** One transformer pass *per candidate* at query time: latency grows linearly with pool `N`.
**Failure.** Trained on MS MARCO web queries; can prefer keyword-dense but unhelpful passages, and
truncates long passages.
**Common mistake.** Reranking a pool that never contained the relevant passage — reranking can only
reorder. Measure Recall@N of the candidate pool first.
**Check.** Why is the cross-encoder not used for first-stage retrieval?

## NDCG, MRR, Recall — `evaluation/metrics.py`

**Math.** `DCG@K = Σ_{i≤K} gain_i / log2(i+1)`, `NDCG = DCG/IDCG`; `MRR = 1/rank_first_relevant`;
`Recall@K = |labels satisfied in top K| / |labels|`.
**Common mistake.** Counting redundant chunks of the same passage as extra gain. Here each label
earns gain once.
**Check.** A system returns 5 chunks of the same relevant passage. What are its Precision@5 and
NDCG@5 under this repo's definitions?

## Entity resolution — `graph/resolution.py`

**What/why.** "LoRA", "Low-Rank Adaptation" and "low rank adaptation" must be one node, or graph
paths fragment.
**Cost.** Alias table maintenance; LLM adjudication calls for ambiguous pairs.
**Failure.** Over-merging (two different methods with similar names) is worse than under-merging:
it creates false paths.
**Common mistake.** Treating any multi-word phrase as an acronym of its capitals — this repo had that
bug ("Natural Quest" merged into "Natural Questions"); the fix and its test are in
`acronym_matches`.
**Check.** Why is the automatic merge threshold set higher than the "ask the LLM" threshold?

## Graph traversal & path scoring — `graph/traversal.py`

**Math.** `score(p) = link(seed) · Π conf(e_i) · decay^(h−1) · Π bonus(e_i) · connect(p)`.
**Why decay.** Each hop multiplies extraction error; long paths drift off-topic.
**Common mistake.** Unbounded BFS from hub nodes ("BERT", "Wikipedia"); bounded by
`max_expansions`, beam width and per-node neighbour caps.
**Check.** Why do paths that end at a *second* query entity get a bonus?

## Lost in the middle & context ordering — `context/builder.py`

LLMs use the beginning and end of long contexts better than the middle (Liu et al., 2023 — in the
corpus as 2307.03172). The `edges` ordering places the strongest evidence at both ends.

## Verification & sufficiency — `verification/verifier.py`, `pipeline/pipeline.py`

Claims are checked against the evidence they cite; if the supported fraction is below the
sufficiency threshold the pipeline reformulates the query and escalates the strategy, at most
`max_iterations` times and only within budget.
**Common mistake.** Letting the verifier and generator share a failure mode (same LLM, same prompt
style) — use a different method (NLI or lexical) or model for verification.

## Retrieval budget — `pipeline/budget.py`

Objective: `max_π E[Q(A,E) − λ·C(R)]`. Each action has a cost (`action_costs`); hard caps raise
`BudgetExceeded` before the action runs; the `adaptive_budget` pipeline starts with the cheapest
strategy and escalates only while retrieval confidence is below `min_confidence`.
