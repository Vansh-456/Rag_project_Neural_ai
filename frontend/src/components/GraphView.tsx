"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";

// react-force-graph uses canvas/window: client-only.
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

export interface GNode {
  id: string;
  name: string;
  type: string;
  seed?: boolean;
}
export interface GLink {
  source: string;
  target: string;
  type: string;
  confidence: number;
  highlight?: boolean;
}

export const TYPE_COLORS: Record<string, string> = {
  Paper: "#6b6860",
  Author: "#a07cc5",
  Method: "#3b5bdb",
  Model: "#0b7285",
  Dataset: "#b35c00",
  Metric: "#c2255c",
  Task: "#2b8a3e",
  Concept: "#868e96",
};

export function typeOf(id: string): string {
  const t = id.split(":")[0];
  return t.charAt(0).toUpperCase() + t.slice(1);
}

export function GraphView({
  nodes,
  links,
  onNodeClick,
  height = 340,
}: {
  nodes: GNode[];
  links: GLink[];
  onNodeClick?: (n: GNode) => void;
  height?: number;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  useEffect(() => {
    if (!box.current) return;
    const ro = new ResizeObserver((e) => setWidth(e[0].contentRect.width));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  // force-graph mutates the objects it is given: hand it copies.
  const data = useMemo(
    () => ({ nodes: nodes.map((n) => ({ ...n })), links: links.map((l) => ({ ...l })) }),
    [nodes, links],
  );
  const dark = typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches;
  if (!nodes.length) return <div className="muted">No graph evidence for this query.</div>;
  return (
    <div>
      <div className="graphbox" ref={box} style={{ height }}>
        <ForceGraph2D
          graphData={data}
          width={width}
          height={height}
          cooldownTicks={80}
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={1}
          linkColor={(l: object) => ((l as GLink).highlight ? (dark ? "#69c47c" : "#2b8a3e") : dark ? "#55534c" : "#cfccc4")}
          linkWidth={(l: object) => ((l as GLink).highlight ? 2.2 : 1)}
          linkLabel={(l: object) => `${(l as GLink).type} (${(l as GLink).confidence.toFixed(2)})`}
          nodeLabel={(n: object) => `${(n as GNode).name} [${(n as GNode).type}]`}
          onNodeClick={(n: object) => onNodeClick?.(n as GNode)}
          nodeCanvasObject={(n: object, ctx: CanvasRenderingContext2D, scale: number) => {
            const node = n as GNode & { x: number; y: number };
            const r = node.seed ? 6 : 4.5;
            ctx.beginPath();
            ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
            ctx.fillStyle = TYPE_COLORS[node.type] ?? "#888";
            ctx.fill();
            if (node.seed) {
              ctx.lineWidth = 2 / scale;
              ctx.strokeStyle = dark ? "#fff" : "#000";
              ctx.stroke();
            }
            const label = node.name.length > 28 ? node.name.slice(0, 26) + "…" : node.name;
            ctx.font = `${11 / scale}px system-ui`;
            ctx.fillStyle = dark ? "#ecebe6" : "#1d1c1a";
            ctx.fillText(label, node.x + r + 2, node.y + 3 / scale);
          }}
        />
      </div>
      <div className="legend">
        {Object.entries(TYPE_COLORS).map(([t, c]) => (
          <span key={t}>
            <i style={{ background: c }} />
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}
