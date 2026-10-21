"""``agvrag`` command-line interface. Every step of the research workflow is one command.

agvrag download                      fetch the arXiv corpus (rate-limited, title-verified)
agvrag parse [--compare]             PDFs -> Documents (+ parse-quality report, parser comparison)
agvrag chunk-sweep                   chunking statistics sweep (strategy x size x overlap)
agvrag index   [-c CONFIG]           chunk + embed + vector index
agvrag graph   [-c CONFIG]           knowledge graph (+ stats, audit sample)
agvrag bootstrap                     download + parse + index + graph
agvrag query "..." [-c CONFIG]       print router decision, ranked evidence, answer
agvrag coverage                      check every ground-truth label maps to >= 1 chunk
agvrag label QUERY_ID                interactive labelling helper
agvrag evaluate -c CONFIG [--split]  run one experiment -> results/<exp_id>/
agvrag sweep                         retrieval grid on dev (chunking x embedder x K)
agvrag oracle [--split]              router accuracy vs per-query oracle strategy
agvrag ablate [--split test]         final study -> docs/ablation_report.md
agvrag serve                         FastAPI service
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agvrag.config import Settings, load_settings
from agvrag.utils.logging import get_logger, setup_logging

log = get_logger("cli")


def _settings(args: argparse.Namespace) -> Settings:
    s = load_settings(args.config)
    setup_logging(s.logging.level, s.logging.json_logs)
    return s


def _out(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def cmd_download(args: argparse.Namespace) -> int:
    from agvrag.ingestion.arxiv import ArxivDownloader, load_corpus_list
    from agvrag.utils.io import write_json

    s = _settings(args)
    specs = load_corpus_list(s.path("corpus_file"))
    if args.limit:
        specs = specs[: args.limit]
    rep = ArxivDownloader(s.path("raw_dir"), s.ingestion.arxiv_delay_seconds).download(specs)
    write_json(s.path("raw_dir").parent / "download_report.json", rep)
    _out({k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()})
    return 0 if not rep["failed"] and not rep["title_mismatch"] else 1


def cmd_parse(args: argparse.Namespace) -> int:
    from agvrag.pipeline.indexing import parse_corpus
    from agvrag.utils.io import write_json

    s = _settings(args)
    results = s.path("results_dir")
    rep = parse_corpus(s, limit=args.limit, force=args.force)
    write_json(results / "parse_report_pymupdf_layout.json", rep)
    summary = {"layout": {k: v for k, v in rep.items() if k not in {"per_document"}}}
    if args.compare:
        plain = parse_corpus(s, parser_name="pymupdf_plain", limit=args.compare_n)
        write_json(results / "parse_report_pymupdf_plain.json", plain)
        layout_subset = [
            q
            for q in rep["per_document"]
            if q["doc_id"] in {p["doc_id"] for p in plain["per_document"]}
        ]
        summary["comparison_on_same_docs"] = {
            "n": len(layout_subset),
            "layout": _parse_summary(layout_subset),
            "plain": _parse_summary(plain["per_document"]),
        }
        write_json(results / "parse_comparison.json", summary["comparison_on_same_docs"])
    _out(summary)
    return 0


def _parse_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = max(1, len(rows))
    return {
        "pct_sections_identified": round(100 * sum(r["sections_identified"] for r in rows) / n, 1),
        "mean_sections": round(sum(r["n_sections"] for r in rows) / n, 1),
        "mean_empty_sections": round(sum(r["n_empty_sections"] for r in rows) / n, 2),
        "mean_references": round(sum(r["n_references"] for r in rows) / n, 1),
        "mean_garbled": round(sum(r["garbled_score"] for r in rows) / n, 4),
    }


def cmd_chunk_sweep(args: argparse.Namespace) -> int:
    from agvrag.chunking.chunkers import build_chunker, chunk_stats
    from agvrag.config import ChunkingConfig
    from agvrag.embeddings.embedders import HashingEmbedder
    from agvrag.pipeline.indexing import load_documents
    from agvrag.utils.io import write_json

    s = _settings(args)
    docs = load_documents(s)
    rows = []
    for strategy in args.strategies.split(","):
        for size in (128, 256, 512, 1024):
            for frac in (0.0, 0.1, 0.2):
                ov = int(size * frac)
                cfg = ChunkingConfig(strategy=strategy, chunk_size=size, overlap=ov)
                # Semantic chunking statistics use the cheap hashing embedder for the sweep.
                ch = build_chunker(cfg, lambda: HashingEmbedder(256))
                chunks = [c for d in docs for c in ch.chunk(d)]
                rows.append(
                    {"strategy": strategy, "size": size, "overlap": ov, **chunk_stats(chunks)}
                )
                log.info("sweep", extra=rows[-1])
    write_json(s.path("results_dir") / "chunking_sweep.json", rows)
    _out(rows)
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from agvrag.pipeline.indexing import build_index

    s = _settings(args)
    _out(build_index(s, rebuild=args.rebuild))
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    from agvrag.pipeline.indexing import build_graph_index

    s = _settings(args)
    if args.extractor:
        s = s.model_copy(update={"graph": s.graph.model_copy(update={"extractor": args.extractor})})
    _out(build_graph_index(s))
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    for fn, ns in (
        (cmd_download, args),
        (cmd_parse, argparse.Namespace(**vars(args), compare=False, force=False)),
        (cmd_index, argparse.Namespace(**vars(args), rebuild=False)),
        (cmd_graph, argparse.Namespace(**vars(args), extractor=None)),
    ):
        rc = fn(ns)
        if rc not in (0, 1):
            return rc
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from agvrag.pipeline.pipeline import RAGPipeline

    s = _settings(args)
    pipe = RAGPipeline(s)
    trace = pipe.run(args.text, generate=args.answer or None, verify=args.verify or None)
    w = sys.stdout.write
    a = trace.query_analysis
    w(
        f"\nQUERY     {args.text}\nINTENT    {a.intent}  complexity={a.complexity}  hops={a.hops_estimate}\n"
    )
    w(f"ENTITIES  {[(e['name'], e['type'], e['method']) for e in a.linked_entities]}\n")
    w(f"ROUTE     {trace.routing.strategy}  — {trace.routing.rationale}\n\n")
    final = trace.reranked or trace.evidence_pool
    for ev in final[: args.k]:
        p = ev.provenance
        score = (
            f"rerank={ev.rerank_score:.3f}"
            if ev.rerank_score is not None
            else f"score={ev.raw_score:.4f}"
        )
        w(f"#{ev.rank:<2} [{ev.type.value}] {score} via {ev.retriever}\n")
        if ev.type.value == "text_chunk":
            w(f"    {p.get('doc_title', '')[:70]} | {p.get('section')} | p.{p.get('page_start')}\n")
        w(f"    {ev.content[:300].replace(chr(10), ' ')}\n\n")
    if trace.answer is not None:
        w(
            f"ANSWER\n{trace.answer}\n\nCITATIONS {[(c['label'], c['doc_title']) for c in trace.citations]}\n"
        )
    if trace.verification and "support_rate" in trace.verification:
        w(
            f"VERIFY    support_rate={trace.verification['support_rate']} sufficient={trace.verification['sufficient']}\n"
        )
    w(f"COSTS     { {k: round(v, 1) for k, v in trace.costs.items()} }\n")
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    from agvrag.evaluation.dataset import label_coverage, load_eval_set
    from agvrag.pipeline.components import load_chunks

    s = _settings(args)
    rep = label_coverage(load_eval_set(s, "all"), load_chunks(s))
    _out(rep)
    return 0 if rep["n_unmapped"] == 0 else 1


def cmd_label(args: argparse.Namespace) -> int:
    from agvrag.evaluation.labeling import interactive_label

    s = _settings(args)
    return interactive_label(s, args.query_id, args.pool)


def cmd_evaluate(args: argparse.Namespace) -> int:
    from agvrag.evaluation.runner import run_experiment

    s = _settings(args)
    res = run_experiment(s, args.phase, args.split, save_traces=not args.no_traces)
    m = res["metrics"]
    o = m["overall"]
    _out(
        {
            "exp_id": res["exp_id"],
            "n": m["n_queries"],
            "overall": {k: o[k]["mean"] for k in ("ndcg@5", "recall@5", "mrr", "hit@5") if k in o},
            "per_type_ndcg@5": {
                t: (v.get("ndcg@5") or {}).get("mean") for t, v in m["per_type"].items()
            },
            "latency_ms_p50": o.get("latency_ms_p50"),
            "label_coverage": m["label_coverage"],
        }
    )
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Phase 5 grid: chunking strategy x size x embedder on dev, K in metrics."""
    from agvrag.evaluation.runner import run_experiment
    from agvrag.pipeline.components import ComponentRegistry
    from agvrag.pipeline.indexing import build_index
    from agvrag.utils.io import write_json

    reg = ComponentRegistry()
    rows = []
    for model in args.models.split(","):
        for strategy in args.strategies.split(","):
            for size in [int(x) for x in args.sizes.split(",")]:
                ov = int(size * 0.125)
                s = load_settings(
                    "vector_only",
                    overrides={
                        "chunking": {"strategy": strategy, "chunk_size": size, "overlap": ov},
                        "embedding": {"model": model, "query_prefix": _query_prefix(model)},
                    },
                )
                setup_logging(s.logging.level, s.logging.json_logs)
                build_index(s, reg)
                res = run_experiment(
                    s,
                    "sweep",
                    args.split,
                    reg,
                    save_traces=False,
                    tag=f"{strategy}{size}-{model.split('/')[-1]}",
                )
                o = res["metrics"]["overall"]
                rows.append(
                    {
                        "model": model,
                        "strategy": strategy,
                        "size": size,
                        "overlap": ov,
                        "exp_id": res["exp_id"],
                        **{
                            k: o[k]["mean"]
                            for k in ("ndcg@5", "recall@1", "recall@5", "recall@10", "mrr")
                            if k in o
                        },
                        "ci_ndcg@5": [o["ndcg@5"]["ci_low"], o["ndcg@5"]["ci_high"]]
                        if "ndcg@5" in o
                        else None,
                    }
                )
                write_json(Path(s.root) / "results" / "retrieval_sweep.json", rows)
    _out(rows)
    return 0


def _query_prefix(model: str) -> str:
    m = model.lower()
    if "bge" in m:
        return "Represent this sentence for searching relevant passages: "
    if "e5" in m:
        return "query: "
    return ""


def cmd_oracle(args: argparse.Namespace) -> int:
    from agvrag.evaluation.ablation import router_oracle
    from agvrag.pipeline.components import ComponentRegistry

    s = _settings(args)
    res = router_oracle(s, args.split, ComponentRegistry())
    _out({k: v for k, v in res.items() if k != "rows"})
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    from agvrag.evaluation.ablation import SYSTEMS, compare_systems, render_report, router_oracle
    from agvrag.pipeline.components import ComponentRegistry
    from agvrag.utils.io import write_json

    s = _settings(args)
    reg = ComponentRegistry()
    systems = args.systems.split(",") if args.systems else SYSTEMS
    cmp = compare_systems(systems, args.split, reg, baseline=args.baseline)
    oracle = (
        router_oracle(s, args.split, reg, router_config=args.oracle_config)
        if not args.no_oracle
        else None
    )
    docs = Path(s.root) / "docs"
    render_report(cmp, docs / "ablation_report.md", docs / "img" / "pareto.png", oracle)
    summary = {
        n: {"exp_id": r["exp_id"], "ndcg@5": r["metrics"]["overall"].get("ndcg@5")}
        for n, r in cmp["results"].items()
    }
    write_json(
        s.path("results_dir") / f"ablation_summary_{args.split}.json",
        {
            "summary": summary,
            "significance": cmp["significance"],
            "oracle": {k: v for k, v in (oracle or {}).items() if k != "rows"},
        },
    )
    _out(summary)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "agvrag.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agvrag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-c", "--config", default=None, help="config name or YAML path (default: base)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("download")
    sp.add_argument("--limit", type=int)
    sp.set_defaults(fn=cmd_download)
    sp = sub.add_parser("parse")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--compare", action="store_true", help="also run the plain parser")
    sp.add_argument("--compare-n", type=int, default=10)
    sp.set_defaults(fn=cmd_parse)
    sp = sub.add_parser("chunk-sweep")
    sp.add_argument("--strategies", default="fixed,recursive,semantic,structure_aware")
    sp.set_defaults(fn=cmd_chunk_sweep)
    sp = sub.add_parser("index")
    sp.add_argument("--rebuild", action="store_true")
    sp.set_defaults(fn=cmd_index)
    sp = sub.add_parser("graph")
    sp.add_argument("--extractor", choices=["rule", "llm", "rule+llm"])
    sp.set_defaults(fn=cmd_graph)
    sp = sub.add_parser("bootstrap")
    sp.add_argument("--limit", type=int)
    sp.set_defaults(fn=cmd_bootstrap)
    sp = sub.add_parser("query")
    sp.add_argument("text")
    sp.add_argument("-k", type=int, default=8)
    sp.add_argument("--answer", action="store_true", help="also generate an answer")
    sp.add_argument("--verify", action="store_true")
    sp.set_defaults(fn=cmd_query)
    sp = sub.add_parser("coverage")
    sp.set_defaults(fn=cmd_coverage)
    sp = sub.add_parser("label")
    sp.add_argument("query_id")
    sp.add_argument("--pool", type=int, default=15)
    sp.set_defaults(fn=cmd_label)
    sp = sub.add_parser("evaluate")
    sp.add_argument("--split", choices=["dev", "test", "all"])
    sp.add_argument("--phase", default="eval")
    sp.add_argument("--no-traces", action="store_true")
    sp.set_defaults(fn=cmd_evaluate)
    sp = sub.add_parser("sweep")
    sp.add_argument("--split", default="dev")
    sp.add_argument("--models", default="BAAI/bge-small-en-v1.5")
    sp.add_argument("--strategies", default="fixed,recursive,structure_aware")
    sp.add_argument("--sizes", default="128,256,512")
    sp.set_defaults(fn=cmd_sweep)
    sp = sub.add_parser("oracle")
    sp.add_argument("--split", default="dev")
    sp.set_defaults(fn=cmd_oracle)
    sp = sub.add_parser("ablate")
    sp.add_argument("--split", default="test")
    sp.add_argument("--systems")
    sp.add_argument("--baseline", default="vector_only")
    sp.add_argument("--no-oracle", action="store_true")
    sp.add_argument("--oracle-config", default="adaptive")
    sp.set_defaults(fn=cmd_ablate)
    sp = sub.add_parser("serve")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(fn=cmd_serve)
    return p


def _keep_awake() -> None:
    """On Windows, ask the OS not to sleep while a long command runs.

    perf_counter keeps counting through system sleep, so a suspended laptop would otherwise
    corrupt latency measurements (this happened once; see docs/experiments.md).
    """
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def _utf8_stdio() -> None:
    """Paper text is full of math symbols; a Windows console (cp1252) cannot encode them and
    ``print`` would crash mid-output. Force UTF-8 and replace anything still unencodable."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    args = build_parser().parse_args(argv)
    if args.cmd in {"index", "graph", "bootstrap", "evaluate", "sweep", "oracle", "ablate"}:
        _keep_awake()
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
