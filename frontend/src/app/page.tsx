"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { PipelineInfo, PipelineTrace } from "@/lib/types";
import { TraceView } from "@/components/TraceView";

const EXAMPLES = [
  "What is LoRA?",
  "Who introduced ColBERT?",
  "How does LoRA differ from adapters?",
  "Which methods based on DPR were evaluated on Natural Questions?",
  "What evidence shows that HyDE improves zero-shot retrieval?",
  "How does RAG compare to REALM?",
];

export default function QueryPage() {
  const [pipelines, setPipelines] = useState<PipelineInfo[]>([]);
  const [query, setQuery] = useState("");
  const [pipeline, setPipeline] = useState("adaptive");
  const [compareWith, setCompareWith] = useState("");
  const [generate, setGenerate] = useState(true);
  const [verify, setVerify] = useState(false);
  const [traces, setTraces] = useState<PipelineTrace[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .pipelines()
      .then((p) => {
        setPipelines(p);
        const d = p.find((x) => x.default);
        if (d) setPipeline(d.name);
      })
      .catch((e: Error) => setError(`Cannot reach the API: ${e.message}`));
  }, []);

  async function run(q = query) {
    if (q.trim().length < 2) return;
    setQuery(q);
    setLoading(true);
    setError(null);
    try {
      if (compareWith && compareWith !== pipeline) {
        const res = await Promise.all([
          api.query(q, pipeline, generate, verify || undefined),
          api.query(q, compareWith, generate, verify || undefined),
        ]);
        setTraces(res);
      } else {
        setTraces([await api.query(q, pipeline, generate, verify || undefined)]);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const desc = pipelines.find((p) => p.name === pipeline)?.description;
  return (
    <div>
      <form
        className="querybar"
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about the RAG / LLM research corpus…"
          aria-label="Query"
          autoFocus
        />
        <select value={pipeline} onChange={(e) => setPipeline(e.target.value)} aria-label="Pipeline" title={desc}>
          {pipelines.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>
        <select value={compareWith} onChange={(e) => setCompareWith(e.target.value)} aria-label="Compare with">
          <option value="">no comparison</option>
          {pipelines
            .filter((p) => p.name !== pipeline)
            .map((p) => (
              <option key={p.name} value={p.name}>
                compare: {p.name}
              </option>
            ))}
        </select>
        <div className="toggles">
          <label>
            <input type="checkbox" checked={generate} onChange={(e) => setGenerate(e.target.checked)} /> answer
          </label>
          <label>
            <input type="checkbox" checked={verify} onChange={(e) => setVerify(e.target.checked)} /> verify
          </label>
        </div>
        <button className="primary" type="submit" disabled={loading}>
          {loading ? "Running…" : "Run"}
        </button>
      </form>
      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" onClick={() => run(ex)} disabled={loading}>
            {ex}
          </button>
        ))}
      </div>
      {error && <div className="error">{error}</div>}
      {!traces.length && !error && (
        <div className="empty">
          Every query shows the router&apos;s decision, each retriever&apos;s ranked list, graph paths, reranking,
          the cited answer, claim verification and per-stage cost. Pick a second pipeline to compare side by side.
        </div>
      )}
      <div className={`columns ${traces.length > 1 ? "two" : ""}`}>
        {traces.map((t) => (
          <TraceView key={t.trace_id} t={t} />
        ))}
      </div>
    </div>
  );
}
