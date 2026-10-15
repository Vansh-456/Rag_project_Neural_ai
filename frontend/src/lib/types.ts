// Mirrors the dataclasses in src/agvrag/schemas.py (PipelineTrace.to_dict()).
// Request/endpoint types can be regenerated from the live OpenAPI spec with `npm run gen:api`.

export type EvidenceType = "text_chunk" | "graph_path" | "graph_fact";

export interface FusionSource {
  rank: number;
  raw_score: number;
  rrf?: number;
  normalized?: number;
  weight?: number;
}

export interface PathEdge {
  source: string;
  type: string;
  target: string;
  confidence: number;
  evidence: string;
  extractor: string;
  chunks: string[];
}

export interface Evidence {
  evidence_id: string;
  type: EvidenceType;
  content: string;
  source_doc_ids: string[];
  source_chunk_ids: string[];
  retriever: string;
  raw_score: number;
  rank: number;
  rerank_score: number | null;
  provenance: {
    doc_title?: string;
    section?: string;
    page_start?: number;
    page_end?: number;
    pre_rerank_rank?: number;
    via_path?: string;
    nodes?: string[];
    directions?: string[];
    hops?: number;
    edges?: PathEdge[];
    fusion?: { method: string; k?: number; sources: Record<string, FusionSource> };
  };
}

export interface LinkedEntity {
  entity_id: string;
  name: string;
  type: string;
  surface: string;
  method: string;
  score: number;
}

export interface QueryAnalysis {
  query: string;
  intent: string;
  entities: string[];
  complexity: number;
  hops_estimate: number;
  rationale: string;
  linked_entities: LinkedEntity[];
  relation_cues: string[];
}

export interface RoutingDecision {
  strategy: string;
  params: Record<string, unknown>;
  rationale: string;
}

export interface Citation {
  label: string;
  sentence_index: number;
  sentence: string;
  evidence_id: string;
  type: EvidenceType;
  chunk_ids: string[];
  doc_ids: string[];
  page_start?: number | null;
  doc_title?: string | null;
}

export interface ClaimResult {
  index: number;
  claim: string;
  cited: string[];
  label: "supported" | "contradicted" | "unsupported";
  score: number;
  reason?: string;
}

export interface Verification {
  method?: string;
  claims?: ClaimResult[];
  n_claims?: number;
  support_rate?: number;
  sufficient?: boolean;
  contradictions?: { a: string; b: string; score: number }[];
  iterations?: number;
  citation_validation?: {
    valid: boolean;
    invalid_ids: string[];
    uncited_claims: string[];
    citation_coverage: number;
    n_claim_sentences: number;
  };
}

export interface PipelineTrace {
  trace_id: string;
  pipeline: string;
  query_analysis: QueryAnalysis;
  routing: RoutingDecision;
  retrieved: Evidence[];
  reranked: Evidence[];
  context: string;
  answer: string | null;
  citations: Citation[];
  verification: Verification | null;
  costs: Record<string, number>;
  retrieval_lists: Record<string, Evidence[]>;
  evidence_pool: Evidence[];
  iterations: Record<string, unknown>[];
  created_at: string;
  cached?: boolean;
}

export interface PipelineInfo {
  name: string;
  description: string;
  router: string;
  strategy: string;
  rerank: boolean;
  generate: boolean;
  verify: boolean;
  default: boolean;
}

export interface Chunk {
  chunk_id: string;
  doc_id: string;
  text: string;
  section_heading: string | null;
  page_start: number | null;
  page_end: number | null;
  strategy: string;
  token_count: number;
  metadata: { doc_title?: string; context?: string; year?: number };
}

export interface MetricCI {
  mean: number;
  ci_low: number;
  ci_high: number;
}

export interface ExperimentSummary {
  exp_id: string;
  pipeline: string;
  split: string;
  n_queries: number;
  "ndcg@5": number | null;
  "recall@5": number | null;
  mrr: number | null;
  latency_ms_p50: number | null;
  created_at: string;
}

export interface ExperimentDetail {
  metrics: {
    overall: Record<string, MetricCI | number>;
    per_type: Record<string, Record<string, MetricCI | number>>;
    strategy_counts: Record<string, number>;
    exp_id: string;
    pipeline: string;
    split: string;
    n_queries: number;
  };
  failures: { failures: Record<string, unknown>[]; category_counts: Record<string, number> };
  per_query: Record<string, string>[];
  config: string;
}

export interface GraphNeighborhood {
  center: string;
  nodes: { id: string; name: string; type: string }[];
  edges: { source: string; target: string; type: string; confidence: number; evidence: string; chunks: string[] }[];
}
