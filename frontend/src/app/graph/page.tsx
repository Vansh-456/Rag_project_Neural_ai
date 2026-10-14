"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { GraphNeighborhood, LinkedEntity } from "@/lib/types";
import { GraphView, type GLink, type GNode } from "@/components/GraphView";
import { SourceDrawer } from "@/components/SourceDrawer";

export default function GraphPage() {
  const [q, setQ] = useState("LoRA");
  const [hits, setHits] = useState<LinkedEntity[]>([]);
  const [hood, setHood] = useState<GraphNeighborhood | null>(null);
  const [hops, setHops] = useState(1);
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [chunk, setChunk] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.graphStats().then(setStats).catch(() => setStats(null));
  }, []);

  async function search(e?: React.FormEvent) {
    e?.preventDefault();
    setError(null);
    try {
      const h = await api.graphSearch(q);
      setHits(h);
      if (h[0]) await open(h[0].entity_id);
      else setError("No entity linked for that text.");
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function open(id: string, h = hops) {
    try {
      setHood(await api.neighborhood(id, h));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const nodes: GNode[] = (hood?.nodes ?? []).map((n) => ({ ...n, seed: n.id === hood?.center }));
  const links: GLink[] = (hood?.edges ?? []).map((e) => ({
    source: e.source, target: e.target, type: e.type, confidence: e.confidence,
    highlight: e.source === hood?.center || e.target === hood?.center,
  }));

  return (
    <div>
      <form className="querybar" style={{ gridTemplateColumns: "1fr auto auto" }} onSubmit={search}>
        <input type="text" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Entity name (LoRA, DPR, GLUE…)" />
        <select value={hops} onChange={(e) => { const h = Number(e.target.value); setHops(h); if (hood) open(hood.center, h); }}>
          <option value={1}>1 hop</option>
          <option value={2}>2 hops</option>
        </select>
        <button className="primary" type="submit">Explore</button>
      </form>
      {hits.length > 1 && (
        <div className="examples">
          {hits.map((h) => (
            <button key={h.entity_id} className="chip" onClick={() => open(h.entity_id)}>
              {h.name} <span className="muted">{h.type}</span>
            </button>
          ))}
        </div>
      )}
      {error && <div className="error">{error}</div>}
      <div className="columns two">
        <div className="panel">
          <h3>Neighborhood <span>click a node to recenter</span></h3>
          <GraphView nodes={nodes} links={links} height={480} onNodeClick={(n) => open(n.id)} />
        </div>
        <div className="panel">
          <h3>Edges with provenance <span>{hood?.edges.length ?? 0}</span></h3>
          <div style={{ maxHeight: 480, overflow: "auto" }}>
            <table>
              <tbody>
                {(hood?.edges ?? []).map((e) => {
                  const name = (id: string) => hood?.nodes.find((n) => n.id === id)?.name ?? id;
                  return (
                    <tr key={`${e.source}${e.type}${e.target}`}>
                      <td>
                        <b>{name(e.source).slice(0, 40)}</b> <span className="chip graph">{e.type}</span>{" "}
                        <b>{name(e.target).slice(0, 40)}</b>
                        <div className="muted clamp" style={{ fontSize: 12 }}>{e.evidence}</div>
                      </td>
                      <td className="num score">{e.confidence.toFixed(2)}</td>
                      <td>{e.chunks[0] && <button className="link" onClick={() => setChunk(e.chunks[0])}>source</button>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      {stats && (
        <div className="panel">
          <h3>Graph statistics</h3>
          <pre className="config">{JSON.stringify(stats, null, 2)}</pre>
        </div>
      )}
      {chunk && <SourceDrawer chunkId={chunk} onClose={() => setChunk(null)} />}
    </div>
  );
}
