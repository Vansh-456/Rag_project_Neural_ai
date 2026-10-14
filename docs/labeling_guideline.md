# Labelling guideline (evaluation set v1)

## 1. Unit of relevance: labels, not chunks

A **label** is `(doc_id, anchors, grade)`:

* `doc_id` — the arXiv id of the paper containing the evidence.
* `anchors` — one or more **short verbatim phrases** (2-8 words) copied from the passage that
  answers the query. Matching is case- and whitespace-insensitive substring matching against the
  chunk text (including its "title > section" context prefix).
* `grade` — `2` = the passage directly answers the question; `1` = partially relevant /
  supporting context; `0` = not relevant (not stored).

A chunk is relevant to a label when it belongs to `doc_id` and contains any anchor. Because labels
are defined on text, **the same ground truth evaluates every chunking strategy fairly**.

## 2. Choosing anchors

* Prefer phrases that are **specific to the answering passage** (`"9%-19%"`,
  `"rank decomposition matrices"`) over generic ones (`"bm25"`). Generic anchors make many chunks
  relevant and inflate precision.
* Keep anchors short: a long anchor can be split across a chunk boundary and silently fail to
  match. `agvrag coverage` reports any label that matches zero chunks — fix those before trusting
  results.
* When several distinct passages answer the query (e.g. both papers in a comparison), create one
  label per passage.

## 3. Grading

| Grade | Use when |
|---|---|
| 2 | A reader could answer the question from this passage alone |
| 1 | The passage is about the right thing but needs other evidence (background, one side of a comparison) |

## 4. Query types

Assign exactly one of `definition`, `relationship`, `comparison`, `multi_hop`, `experimental`
(see `docs/problem_definition.md`). Unanswerable queries carry `answerable: false` and no labels;
they are scored only on abstention.

## 5. Graph expectations

For relationship and multi-hop queries, add `expected_paths.json` entries listing the canonical
entity ids (`method:lora`, `paper:2106.09685`, `author:kelvin guu`) and relation types a correct
path should contain. These are scored with path entity/relation recall and path hit.

## 6. Quality control

1. Run `agvrag coverage` after every edit (must report `n_unmapped: 0`).
2. Use `agvrag label <query_id>` to pool dense + BM25 candidates and add missed relevant passages;
   the tool refuses anchors that do not occur verbatim in the passage.
3. **Re-label a 20% sample after a gap of at least a week** and report agreement (Cohen's kappa on
   grades of pooled candidates). Until this is done the dataset is **draft v1** and every reported
   number is provisional.
4. Freeze the test split before the final ablation; tune only on dev.

## 7. Known limitations of v1

* Single annotator, who also built the system (anchors were picked from paper text before any
  retrieval run, but bias cannot be ruled out).
* Pooling depth is shallow: passages no system retrieved may be relevant but unlabelled, which
  under-estimates precision for systems that find them.
* Some anchors are generic (`"glue"`, `"bm25"`, `"mteb"`) — chosen where the question asks about any
  mention in the paper; they make those queries easier.
