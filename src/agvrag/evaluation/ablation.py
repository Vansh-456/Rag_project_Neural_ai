"""Final study tooling: router oracle, multi-system comparison, significance, report.

* ``router_oracle`` runs every strategy on every query (retrieval only), picks the
  per-query best by utility U = NDCG@k - lambda * cost, and scores the router against it.
* ``compare_systems`` runs several configs on the same split and builds per-type tables,
  paired-bootstrap tests against a baseline, latency/cost tables and a Pareto plot.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agvrag.config import Settings, load_settings
from agvrag.evaluation.dataset import QUERY_TYPES, load_eval_set
from agvrag.evaluation.runner import run_experiment
from agvrag.evaluation.stats import paired_bootstrap
from agvrag.pipeline.components import ComponentRegistry
from agvrag.utils.io import ensure_dir, write_json
from agvrag.utils.logging import get_logger

log = get_logger(__name__)

STRATEGIES = ["vector", "bm25", "hybrid", "graph", "graph_vector", "multi_hop"]


def router_oracle(
    settings: Settings,
    split: str,
    registry: ComponentRegistry,
    metric: str = "ndcg@5",
    tolerance: float = 0.0,
    router_config: str = "adaptive",
) -> dict[str, Any]:
    """Per-query oracle over the pure strategies (no reranking) vs. ``router_config``'s choice."""
    queries = [q for q in load_eval_set(settings, split) if q.answerable and q.labels]
    base = load_settings(router_config, overrides={"evaluation": {"save_traces": False}})
    base = base.model_copy(update={"root": settings.root})
    per_strategy: dict[str, dict[str, dict[str, Any]]] = {}
    for strat in STRATEGIES:
        res = run_experiment(
            base,
            "oracle",
            split,
            registry,
            save_traces=False,
            strategy_override=strat,
            queries=queries,
        )
        per_strategy[strat] = {r["query_id"]: r for r in res["rows"]}
    adaptive = run_experiment(base, "oracle", split, registry, save_traces=False, queries=queries)
    routed = {r["query_id"]: r for r in adaptive["rows"]}
    lam = settings.budget.lambda_cost
    rows = []
    correct = 0
    for q in queries:
        utils = {
            s: per_strategy[s][q.id].get(metric, 0.0) - lam * per_strategy[s][q.id]["cost_units"]
            for s in STRATEGIES
        }
        best = max(utils.values())
        oracle = sorted(s for s, u in utils.items() if u >= best - 1e-9)
        chosen = routed[q.id]["strategy"]
        ok = utils.get(chosen, -1e9) >= best - tolerance
        correct += ok
        rows.append(
            {
                "query_id": q.id,
                "type": q.type,
                "router": chosen,
                "oracle": oracle,
                "router_utility": round(utils.get(chosen, 0.0), 4),
                "oracle_utility": round(best, 4),
                "correct": ok,
            }
        )
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["correct"])
    hyb = [per_strategy["hybrid"][q.id].get(metric, 0.0) for q in queries]
    ada = [routed[q.id].get(metric, 0.0) for q in queries]
    hyb_cost = sum(per_strategy["hybrid"][q.id]["cost_units"] for q in queries) / max(
        1, len(queries)
    )
    ada_cost = sum(routed[q.id]["cost_units"] for q in queries) / max(1, len(queries))
    oracle_quality = sum(
        max(per_strategy[s][q.id].get(metric, 0.0) for s in STRATEGIES) for q in queries
    ) / max(1, len(queries))
    result = {
        "router_config": router_config,
        "metric": metric,
        "lambda": lam,
        "split": split,
        "n": len(queries),
        "router_accuracy": round(correct / max(1, len(queries)), 4),
        "router_accuracy_per_type": {t: round(sum(v) / len(v), 4) for t, v in by_type.items()},
        "mean_quality": {
            s: round(
                sum(per_strategy[s][q.id].get(metric, 0.0) for q in queries) / max(1, len(queries)),
                4,
            )
            for s in STRATEGIES
        }
        | {
            router_config: round(sum(ada) / max(1, len(ada)), 4),
            "oracle": round(oracle_quality, 4),
        },
        "mean_cost_units": {
            s: round(
                sum(per_strategy[s][q.id]["cost_units"] for q in queries) / max(1, len(queries)), 4
            )
            for s in STRATEGIES
        }
        | {router_config: round(ada_cost, 4)},
        "adaptive_vs_always_hybrid": paired_bootstrap(hyb, ada)
        | {"cost_saved_units": round(hyb_cost - ada_cost, 4)},
        "rows": rows,
    }
    out = ensure_dir(
        settings.path("results_dir")
        / f"oracle_{router_config}_{split}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    )
    write_json(out / "oracle.json", result)
    result["dir"] = str(out)
    return result


SYSTEMS = [
    "bm25_only",
    "vector_only",
    "hybrid",
    "hybrid_rerank",
    "graph_only",
    "graph_vector",
    "adaptive",
    "adaptive_tuned",
    "adaptive_budget",
]


def compare_systems(
    systems: list[str],
    split: str,
    registry: ComponentRegistry,
    baseline: str = "vector_only",
    metric: str = "ndcg@5",
    root: str | None = None,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for name in systems:
        s = load_settings(name)
        if root:
            s = s.model_copy(update={"root": root})
        log.info("running system", extra={"system": name, "split": split})
        results[name] = run_experiment(s, "ablation", split, registry, save_traces=False)
    base_rows = {
        r["query_id"]: r for r in results[baseline]["rows"] if r["answerable"] and metric in r
    }
    sig: dict[str, Any] = {}
    for name, res in results.items():
        rows = {r["query_id"]: r for r in res["rows"] if r["answerable"] and metric in r}
        ids = sorted(set(rows) & set(base_rows))
        sig[name] = {
            "overall": paired_bootstrap(
                [base_rows[i][metric] for i in ids], [rows[i][metric] for i in ids]
            )
        }
        for t in QUERY_TYPES:
            tid = [i for i in ids if rows[i]["type"] == t]
            if tid:
                sig[name][t] = paired_bootstrap(
                    [base_rows[i][metric] for i in tid], [rows[i][metric] for i in tid]
                )
    return {
        "results": results,
        "significance": sig,
        "baseline": baseline,
        "metric": metric,
        "split": split,
    }


def _fmt(m: dict[str, Any] | None) -> str:
    if not m:
        return "–"
    return f"{m['mean']:.3f} [{m['ci_low']:.2f}, {m['ci_high']:.2f}]"


def render_report(
    cmp: dict[str, Any],
    out_md: Path,
    plot_path: Path | None = None,
    oracle: dict[str, Any] | None = None,
) -> str:
    res = cmp["results"]
    metric = cmp["metric"]
    lines = [
        "# Ablation report",
        "",
        f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by `agvrag ablate` — every number below "
        f"is computed from `results/<exp_id>/` (split: **{cmp['split']}**). Do not edit by hand._",
        "",
        f"Baseline for significance tests: **{cmp['baseline']}**; metric: **{metric}**; paired bootstrap, "
        "2000 resamples. Brackets are 95% bootstrap CIs of the mean.",
        "",
        "## Overall retrieval quality",
        "",
        "| System | n | NDCG@5 | Recall@5 | MRR | Hit@5 | Doc-recall@5 | Δ NDCG@5 vs baseline (p) | p50 ms | p95 ms | cost units |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res.items():
        o = r["metrics"]["overall"]
        s = cmp["significance"][name]["overall"]
        lines.append(
            f"| {name} | {o['n']} | {_fmt(o.get('ndcg@5'))} | {_fmt(o.get('recall@5'))} | {_fmt(o.get('mrr'))} | "
            f"{_fmt(o.get('hit@5'))} | {_fmt(o.get('doc_recall@5'))} | {s['diff']:+.3f} (p={s['p_value']:.3f}) | {o.get('latency_ms_p50', 0):.0f} | "
            f"{o.get('latency_ms_p95', 0):.0f} | {o.get('mean_cost_units', 0):.2f} |"
        )
    lines += [
        "",
        f"## {metric.upper()} per query type",
        "",
        "| System | " + " | ".join(QUERY_TYPES) + " |",
        "|---|" + "---|" * len(QUERY_TYPES),
    ]
    for name, r in res.items():
        pt = r["metrics"]["per_type"]
        cells = []
        for t in QUERY_TYPES:
            m = (pt.get(t) or {}).get(metric)
            sig = cmp["significance"][name].get(t)
            star = "*" if sig and sig["p_value"] < 0.05 and name != cmp["baseline"] else ""
            cells.append(f"{m['mean']:.3f}{star} (n={pt[t]['n']})" if m else "–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "`*` = paired-bootstrap p < 0.05 vs the baseline on that query type. Per-type n is small; "
        "treat unstarred differences as noise.",
        "",
    ]
    paths = [(n, r["metrics"]["overall"].get("path_entity_recall")) for n, r in res.items()]
    if any(p for _, p in paths):
        lines += [
            "## Graph path metrics (queries with expected paths)",
            "",
            "| System | path entity recall | path relation recall | path hit |",
            "|---|---|---|---|",
        ]
        for n, r in res.items():
            o = r["metrics"]["overall"]
            lines.append(
                f"| {n} | {_fmt(o.get('path_entity_recall'))} | {_fmt(o.get('path_relation_recall'))} | {_fmt(o.get('path_hit'))} |"
            )
        lines.append("")
    lines += ["## Strategy usage", "", "| System | strategies chosen |", "|---|---|"]
    for n, r in res.items():
        lines.append(f"| {n} | {r['metrics']['strategy_counts']} |")
    lines += ["", "## Failure categories (hit@5 = 0)", "", "| System | categories |", "|---|---|"]
    from agvrag.utils.io import read_json

    for n, r in res.items():
        f = read_json(Path(r["dir"]) / "failures.json")
        lines.append(f"| {n} | {f['category_counts']} |")
    if oracle:
        lines += [
            "",
            "## Router vs oracle",
            "",
            f"Router `{oracle.get('router_config', 'adaptive')}` accuracy (chosen strategy achieves the oracle's best utility): **{oracle['router_accuracy']:.2f}** "
            f"(n={oracle['n']}).",
            "",
            f"Per type: {oracle['router_accuracy_per_type']}",
            "",
            f"Mean {oracle['metric']} by strategy: {oracle['mean_quality']}",
            "",
            f"Mean cost units by strategy: {oracle['mean_cost_units']}",
            "",
            f"Adaptive vs always-hybrid: {oracle['adaptive_vs_always_hybrid']}",
            "",
        ]
    lines += ["", "## Experiment ids", ""] + [f"- {n}: `{r['exp_id']}`" for n, r in res.items()]
    if plot_path is not None and _pareto_plot(res, plot_path):
        lines += ["", f"![Quality vs cost]({plot_path.name})", ""]
    text = "\n".join(lines) + "\n"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(text, encoding="utf-8")
    return text


def _pareto_plot(res: dict[str, Any], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, xkey, xlabel in (
        (axes[0], "latency_ms_p50", "p50 latency (ms)"),
        (axes[1], "mean_cost_units", "mean retrieval cost (units)"),
    ):
        pts = []
        for n, r in res.items():
            o = r["metrics"]["overall"]
            if o.get("ndcg@5") and o.get(xkey) is not None:
                pts.append((o[xkey], o["ndcg@5"]["mean"], n))
        for x, y, n in pts:
            ax.scatter(x, y, s=40)
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        frontier = []
        best = -1.0
        for x, y, _name in sorted(pts):
            if y > best:
                frontier.append((x, y))
                best = y
        if frontier:
            ax.plot([p[0] for p in frontier], [p[1] for p in frontier], linestyle="--", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("NDCG@5")
        ax.grid(alpha=0.3)
    fig.suptitle("Quality vs cost (dashed: Pareto frontier)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
