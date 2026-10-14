"use client";

import { useMemo, useState } from "react";
import type { Evidence, PipelineTrace } from "@/lib/types";
import { GraphView, typeOf, type GLink, type GNode } from "./GraphView";
import { SourceDrawer } from "./SourceDrawer";

const fmt = (n: number | null | undefined, d = 3) => (n == null ? "–" : n.toFixed(d));

function retrieverClass(r: string) {
  if (r.includes("graph")) return "graph";
  if (r.includes("bm25")) return "bm25";
  return "dense";
}

function EvidenceItem({ ev, onOpen, showFusion }: { ev: Evidence; onOpen: (id: string) => void; showFusion?: boolean }) {
  const p = ev.provenance;
  const isPath = ev.type !== "text_chunk";
  const fusion = p.fusion?.sources;
  return (
    <li>
      <div className="evmeta">
        <span className="rank">#{ev.rank}</span>
        <span className={`chip ${retrieverClass(ev.retriever)}`}>{ev.retriever}</span>
        <span className="score">
          {ev.rerank_score != null ? `rerank ${fmt(ev.rerank_score)}` : `score ${fmt(ev.raw_score, 4)}`}
        </span>
        {p.pre_rerank_rank != null && (
          <span className={p.pre_rerank_rank > ev.rank ? "up" : p.pre_rerank_rank < ev.rank ? "down" : ""}>
            (was #{p.pre_rerank_rank})
          </span>
        )}
        {!isPath && (
          <span>
            {p.doc_title?.slice(0, 60)} · {p.section?.slice(0, 40)} · p.{p.page_start}
          </span>
        )}
        {isPath && <span>{p.hops}-hop path</span>}
        {ev.source_chunk_ids[0] && (
          <button className="link" onClick={() => onOpen(ev.source_chunk_ids[0])}>
            source
          </button>
        )}
      </div>
      {showFusion && fusion && (
        <div className="evmeta" style={{ marginTop: 2 }}>
          {Object.entries(fusion).map(([src, v]) => (
            <span key={src} className={`chip ${retrieverClass(src)}`}>
              {src}: #{v.rank}
              {v.rrf != null ? ` · rrf ${v.rrf.toFixed(4)}` : ""}
              {v.normalized != null ? ` · norm ${v.normalized.toFixed(2)}` : ""}
            </span>
          ))}
        </div>
      )}
      <div className={`evtext ${isPath ? "path" : "clamp"}`}>{ev.content}</div>
    </li>
  );
}

function RouterPanel({ t }: { t: PipelineTrace }) {
  const a = t.query_analysis;
  return (
    <div className="panel">
      <h3>
        Router decision <span className="mono">{t.pipeline}</span>
      </h3>
      <dl className="kv">
        <dt>Strategy</dt>
        <dd className="strategy">{t.routing.strategy}</dd>
        <dt>Intent</dt>
        <dd>{a.intent}</dd>
        <dt>Entities</dt>
        <dd style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {a.linked_entities.length === 0 && <span className="muted">none linked</span>}
          {a.linked_entities.map((e) => (
            <span key={e.entity_id} className="chip" title={`${e.entity_id} via ${e.method}`}>
              {e.name} <span className="muted">{e.type}</span>
            </span>
          ))}
        </dd>
        <dt>Relation cues</dt>
        <dd>{a.relation_cues.join(", ") || <span className="muted">–</span>}</dd>
        <dt>Complexity</dt>
        <dd>
          <div className="bar" style={{ width: 160, display: "inline-block", marginRight: 8 }}>
            <div style={{ width: `${a.complexity * 100}%` }} />
          </div>
          {a.complexity.toFixed(2)} · hops≈{a.hops_estimate}
        </dd>
      </dl>
      <div className="rationale">{t.routing.rationale}</div>
      <div className="rationale">Analyzer: {a.rationale}</div>
    </div>
  );
}

function AnswerPanel({ t, onOpen }: { t: PipelineTrace; onOpen: (id: string) => void }) {
  if (t.answer == null) return null;
  const labels = new Map(t.citations.map((c) => [c.label, c]));
  const invalid = new Set(t.verification?.citation_validation?.invalid_ids ?? []);
  const parts = t.answer.split(/(\[[EG]\d+(?:\s*[,;]\s*[EG]\d+)*\])/g);
  const abstained = t.answer.startsWith("INSUFFICIENT_EVIDENCE");
  return (
    <div className="panel">
      <h3>
        Answer
        {t.verification?.citation_validation && (
          <span className={t.verification.citation_validation.valid ? "up" : "down"}>
            citations {t.verification.citation_validation.valid ? "valid" : "invalid"} · coverage{" "}
            {(t.verification.citation_validation.citation_coverage * 100).toFixed(0)}%
          </span>
        )}
      </h3>
      <div className={`answer ${abstained ? "abstain" : ""}`}>
        {parts.map((part, i) => {
          const m = part.match(/^\[(.+)\]$/);
          if (!m) return <span key={i}>{part}</span>;
          return m[1].split(/\s*[,;]\s*/).map((lab) => {
            const c = labels.get(lab);
            const cls = `cite ${lab.startsWith("G") ? "g" : ""} ${invalid.has(lab) || !c ? "bad" : ""}`;
            return (
              <button
                key={`${i}-${lab}`}
                className={cls}
                title={c ? `${c.doc_title ?? c.doc_ids[0]} p.${c.page_start ?? "?"}` : "unknown evidence id"}
                onClick={() => c?.chunk_ids[0] && onOpen(c.chunk_ids[0])}
              >
                {lab}
              </button>
            );
          });
        })}
      </div>
    </div>
  );
}

function RetrievalPanel({ t, onOpen }: { t: PipelineTrace; onOpen: (id: string) => void }) {
  const lists = useMemo(() => {
    const l: Record<string, Evidence[]> = { "final pool": t.evidence_pool };
    for (const [k, v] of Object.entries(t.retrieval_lists)) l[k] = v;
    if (t.retrieved.length && !l[t.routing.strategy]) l["retrieved (pre-rerank)"] = t.retrieved;
    return l;
  }, [t]);
  const [tab, setTab] = useState("final pool");
  const items = lists[tab] ?? [];
  return (
    <div className="panel">
      <h3>
        Retrieval <span>{items.length} items</span>
      </h3>
      <div className="tabs">
        {Object.keys(lists).map((k) => (
          <button key={k} className={k === tab ? "on" : ""} onClick={() => setTab(k)}>
            {k} ({lists[k].length})
          </button>
        ))}
      </div>
      <ul className="evlist">
        {items.slice(0, 15).map((ev) => (
          <EvidenceItem key={`${tab}-${ev.evidence_id}-${ev.rank}`} ev={ev} onOpen={onOpen} showFusion={tab === "final pool"} />
        ))}
      </ul>
    </div>
  );
}

function RerankPanel({ t }: { t: PipelineTrace }) {
  if (!t.reranked.length) return null;
  return (
    <div className="panel">
      <h3>Reranking (before → after)</h3>
      <table>
        <thead>
          <tr>
            <th className="num">after</th>
            <th className="num">before</th>
            <th className="num">rerank score</th>
            <th>passage</th>
          </tr>
        </thead>
        <tbody>
          {t.reranked.slice(0, 10).map((e) => {
            const before = e.provenance.pre_rerank_rank ?? 0;
            return (
              <tr key={e.evidence_id}>
                <td className="num">{e.rank}</td>
                <td className={`num ${before > e.rank ? "up" : before < e.rank ? "down" : ""}`}>{before}</td>
                <td className="num score">{fmt(e.rerank_score)}</td>
                <td>
                  <div className="clamp">{e.content.slice(0, 220)}</div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function GraphPanel({ t }: { t: PipelineTrace }) {
  const { nodes, links } = useMemo(() => {
    const seeds = new Set(t.query_analysis.linked_entities.map((e) => e.entity_id));
    const nodes = new Map<string, GNode>();
    const links = new Map<string, GLink>();
    const all = [...t.evidence_pool, ...(t.retrieval_lists["graph_paths"] ?? [])];
    all.forEach((ev, idx) => {
      if (ev.type === "text_chunk") return;
      const names = ev.content.split(/\s+(?:<-[A-Z_]+-|-[A-Z_]+->)\s+/).map((s) => s.replace(/\s*\[.*\]$/, ""));
      (ev.provenance.nodes ?? []).forEach((id, i) => {
        if (!nodes.has(id)) nodes.set(id, { id, name: names[i] ?? id, type: typeOf(id), seed: seeds.has(id) });
      });
      (ev.provenance.edges ?? []).forEach((e) => {
        const key = `${e.source}|${e.type}|${e.target}`;
        const prev = links.get(key);
        links.set(key, { source: e.source, target: e.target, type: e.type, confidence: e.confidence,
          highlight: (prev?.highlight ?? false) || idx < 3 });
      });
    });
    return { nodes: [...nodes.values()], links: [...links.values()] };
  }, [t]);
  if (!nodes.length) return null;
  return (
    <div className="panel">
      <h3>
        Graph paths <span>{links.length} edges · seeds outlined · top paths highlighted</span>
      </h3>
      <GraphView nodes={nodes} links={links} />
    </div>
  );
}

function VerificationPanel({ t }: { t: PipelineTrace }) {
  const v = t.verification;
  if (!v?.claims) return null;
  return (
    <div className="panel">
      <h3>
        Verification ({v.method})
        <span className={v.sufficient ? "up" : "down"}>
          support {((v.support_rate ?? 0) * 100).toFixed(0)}% · {v.sufficient ? "sufficient" : "insufficient"} ·{" "}
          {v.iterations ?? 0} retries
        </span>
      </h3>
      <table>
        <tbody>
          {v.claims.map((c) => (
            <tr key={c.index}>
              <td style={{ width: 110 }}>
                <span className={`label ${c.label}`}>{c.label}</span>
              </td>
              <td>{c.claim}</td>
              <td className="mono muted">{c.cited.join(",") || "uncited"}</td>
              <td className="num score">{fmt(c.score, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {t.iterations.length > 1 && (
        <div style={{ marginTop: 8 }}>
          {t.iterations.map((it, i) => (
            <div key={i} className="mono muted">
              {JSON.stringify(it)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CostPanel({ t }: { t: PipelineTrace }) {
  const stages = Object.entries(t.costs)
    .filter(([k]) => k.startsWith("latency_ms.") && k !== "latency_ms.total")
    .map(([k, v]) => [k.replace("latency_ms.", ""), v] as const);
  const total = t.costs["latency_ms.total"] ?? 0;
  const max = Math.max(1, ...stages.map(([, v]) => v));
  return (
    <div className="panel">
      <h3>
        Cost & latency {t.cached && <span className="chip">cached</span>}
      </h3>
      <div style={{ marginBottom: 8 }}>
        <span className="stat"><b>{total.toFixed(0)}</b><span>ms total</span></span>
        <span className="stat"><b>{t.costs["budget.retrieval_calls"] ?? 0}</b><span>retrieval calls</span></span>
        <span className="stat"><b>{t.costs["budget.graph_hops"] ?? 0}</b><span>graph hops</span></span>
        <span className="stat"><b>{(t.costs["budget.cost_units"] ?? 0).toFixed(2)}</b><span>cost units</span></span>
        {t.costs["llm_input_tokens"] != null && (
          <span className="stat">
            <b>{(t.costs["llm_input_tokens"] ?? 0) + (t.costs["llm_output_tokens"] ?? 0)}</b>
            <span>LLM tokens · ${(t.costs["llm_cost_usd"] ?? 0).toFixed(4)}</span>
          </span>
        )}
      </div>
      {stages.map(([name, v]) => (
        <div key={name} className="costrow">
          <span>{name}</span>
          <div className="bar"><div style={{ width: `${(v / max) * 100}%` }} /></div>
          <span className="num mono">{v.toFixed(1)} ms</span>
        </div>
      ))}
    </div>
  );
}

export function TraceView({ t }: { t: PipelineTrace }) {
  const [chunkId, setChunkId] = useState<string | null>(null);
  return (
    <div>
      <RouterPanel t={t} />
      <AnswerPanel t={t} onOpen={setChunkId} />
      <VerificationPanel t={t} />
      <GraphPanel t={t} />
      <RetrievalPanel t={t} onOpen={setChunkId} />
      <RerankPanel t={t} />
      <CostPanel t={t} />
      <div className="muted mono" style={{ fontSize: 11 }}>
        trace {t.trace_id} · {t.created_at}
      </div>
      {chunkId && <SourceDrawer chunkId={chunkId} onClose={() => setChunkId(null)} />}
    </div>
  );
}
