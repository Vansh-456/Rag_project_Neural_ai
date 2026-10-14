"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ExperimentDetail, ExperimentSummary, MetricCI } from "@/lib/types";

const METRICS = ["ndcg@5", "recall@5", "mrr", "hit@5", "path_entity_recall"];

function ci(m: MetricCI | number | undefined) {
  if (m == null) return "–";
  if (typeof m === "number") return m.toFixed(2);
  return (
    <span title={`95% CI [${m.ci_low.toFixed(3)}, ${m.ci_high.toFixed(3)}]`}>
      {m.mean.toFixed(3)} <span className="muted">±{((m.ci_high - m.ci_low) / 2).toFixed(2)}</span>
    </span>
  );
}

export default function ExperimentsPage() {
  const [rows, setRows] = useState<ExperimentSummary[]>([]);
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [filter, setFilter] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.experiments().then(setRows).catch((e: Error) => setError(e.message));
  }, []);

  const shown = rows.filter((r) => !filter || r.exp_id.includes(filter));
  return (
    <div>
      <div className="panel">
        <h3>
          Results registry <span>{rows.length} runs — each row is results/&lt;exp_id&gt;/</span>
        </h3>
        <input
          type="text"
          placeholder="filter by id (e.g. ablation_, test, sweep_)"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{ padding: 6, marginBottom: 8, width: 320, border: "1px solid var(--border)",
                   borderRadius: 6, background: "var(--bg)", color: "var(--text)" }}
        />
        {error && <div className="error">{error}</div>}
        <table>
          <thead>
            <tr>
              <th>experiment</th>
              <th>pipeline</th>
              <th>split</th>
              <th className="num">n</th>
              <th className="num">NDCG@5</th>
              <th className="num">Recall@5</th>
              <th className="num">MRR</th>
              <th className="num">p50 ms</th>
            </tr>
          </thead>
          <tbody>
            {shown.slice(0, 200).map((r) => (
              <tr key={r.exp_id} className="clickable" onClick={() => api.experiment(r.exp_id).then(setDetail)}>
                <td className="mono">{r.exp_id}</td>
                <td>{r.pipeline}</td>
                <td>{r.split}</td>
                <td className="num">{r.n_queries}</td>
                <td className="num">{r["ndcg@5"]?.toFixed(3) ?? "–"}</td>
                <td className="num">{r["recall@5"]?.toFixed(3) ?? "–"}</td>
                <td className="num">{r.mrr?.toFixed(3) ?? "–"}</td>
                <td className="num">{r.latency_ms_p50?.toFixed(0) ?? "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {detail && (
        <div className="panel">
          <h3>
            {detail.metrics.exp_id} <button onClick={() => setDetail(null)}>close</button>
          </h3>
          <table>
            <thead>
              <tr>
                <th>query type</th>
                <th className="num">n</th>
                {METRICS.map((m) => (
                  <th key={m} className="num">{m}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries({ overall: detail.metrics.overall, ...detail.metrics.per_type }).map(([t, m]) => (
                <tr key={t}>
                  <td style={{ fontWeight: t === "overall" ? 700 : 400 }}>{t}</td>
                  <td className="num">{String(m.n ?? "")}</td>
                  {METRICS.map((k) => (
                    <td key={k} className="num">{ci(m[k] as MetricCI | undefined)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted">Strategies used: {JSON.stringify(detail.metrics.strategy_counts)}</p>
          <p className="muted">Failure categories: {JSON.stringify(detail.failures.category_counts ?? {})}</p>
          <h3 style={{ marginTop: 16 }}>Per query</h3>
          <div style={{ maxHeight: 360, overflow: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>id</th><th>type</th><th>strategy</th><th className="num">ndcg@5</th>
                  <th className="num">recall@5</th><th className="num">mrr</th><th className="num">ms</th>
                </tr>
              </thead>
              <tbody>
                {detail.per_query.map((q) => (
                  <tr key={q.query_id}>
                    <td className="mono">{q.query_id}</td><td>{q.type}</td><td>{q.strategy}</td>
                    <td className="num">{q["ndcg@5"] ? Number(q["ndcg@5"]).toFixed(3) : "–"}</td>
                    <td className="num">{q["recall@5"] ? Number(q["recall@5"]).toFixed(3) : "–"}</td>
                    <td className="num">{q.mrr ? Number(q.mrr).toFixed(3) : "–"}</td>
                    <td className="num">{Number(q.latency_ms).toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h3 style={{ marginTop: 16 }}>Resolved config</h3>
          <pre className="config">{detail.config}</pre>
        </div>
      )}
    </div>
  );
}
