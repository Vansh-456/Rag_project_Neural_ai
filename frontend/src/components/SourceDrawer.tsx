"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Chunk } from "@/lib/types";

export function SourceDrawer({ chunkId, onClose }: { chunkId: string; onClose: () => void }) {
  const [chunk, setChunk] = useState<Chunk | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setChunk(null);
    setError(null);
    api.chunk(chunkId).then(setChunk).catch((e: Error) => setError(e.message));
  }, [chunkId]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const arxiv = chunk && /^\d{4}\.\d{4,5}$/.test(chunk.doc_id) ? `https://arxiv.org/abs/${chunk.doc_id}` : null;
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-label="Source passage">
        <button onClick={onClose} style={{ float: "right" }}>
          Close
        </button>
        <h3 className="muted" style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.06em" }}>
          Source passage
        </h3>
        {error && <div className="error">{error}</div>}
        {!chunk && !error && <div className="muted">Loading…</div>}
        {chunk && (
          <>
            <h2 style={{ fontSize: 16 }}>{chunk.metadata.doc_title ?? chunk.doc_id}</h2>
            <div className="muted" style={{ marginBottom: 12 }}>
              {chunk.section_heading} · pages {chunk.page_start}–{chunk.page_end} · {chunk.token_count} tokens ·{" "}
              {arxiv ? (
                <a href={arxiv} target="_blank" rel="noreferrer">
                  arXiv:{chunk.doc_id}
                </a>
              ) : (
                chunk.doc_id
              )}
            </div>
            <p style={{ whiteSpace: "pre-wrap", lineHeight: 1.65 }}>{chunk.text}</p>
            <div className="mono muted" style={{ fontSize: 11 }}>
              {chunk.chunk_id}
            </div>
          </>
        )}
      </aside>
    </>
  );
}
