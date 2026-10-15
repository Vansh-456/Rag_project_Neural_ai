// Typed client for the agvrag API. All calls go through the Next.js proxy at /api/*,
// which injects server-side credentials, so no secret ever reaches the browser.
import type {
  Chunk,
  ExperimentDetail,
  ExperimentSummary,
  GraphNeighborhood,
  LinkedEntity,
  PipelineInfo,
  PipelineTrace,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => call<Record<string, unknown>>("/health"),
  pipelines: () => call<PipelineInfo[]>("/pipelines"),
  query: (query: string, pipeline: string, generate = true, verify?: boolean) =>
    call<PipelineTrace>("/query", {
      method: "POST",
      body: JSON.stringify({ query, pipeline, generate, verify }),
    }),
  compare: (query: string, pipelines: [string, string]) =>
    call<{ query: string; results: Record<string, PipelineTrace> }>("/compare", {
      method: "POST",
      body: JSON.stringify({ query, pipelines, generate: true }),
    }),
  trace: (id: string) => call<PipelineTrace>(`/trace/${encodeURIComponent(id)}`),
  chunk: (id: string) => call<Chunk>(`/chunks/${encodeURIComponent(id)}`),
  experiments: () => call<ExperimentSummary[]>("/experiments"),
  experiment: (id: string) => call<ExperimentDetail>(`/experiments/${encodeURIComponent(id)}`),
  graphSearch: (q: string) => call<LinkedEntity[]>(`/graph/search?q=${encodeURIComponent(q)}`),
  neighborhood: (id: string, hops = 1) =>
    call<GraphNeighborhood>(`/graph/neighborhood/${encodeURIComponent(id)}?hops=${hops}`),
  graphStats: () => call<Record<string, unknown>>("/graph/stats"),
};
