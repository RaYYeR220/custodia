/**
 * Client for the Custodia API.
 *
 * Every call goes through the Next route handlers under /api so the browser
 * never needs to know where the Python service lives, which is what lets the
 * same build run from `npm run dev` and from compose without configuration.
 */

export type Tier = "owner" | "assistant" | "tool" | "external";
export type FactStatus = "active" | "superseded" | "retracted" | "quarantined";

export interface EvidenceItem {
  fact_id: number;
  text: string;
  tier: Tier;
  status: FactStatus;
  valid_from: number;
  valid_to: number;
  session: string;
  session_index: number;
  turn_index: number;
  turn_text: string;
  turn_ts: number;
  score: number;
  hops: number;
  path: string[];
  superseded_by: number | null;
}

export interface Warrant {
  question: string;
  asked_at: number;
  as_of: number | null;
  evidence: EvidenceItem[];
  seeds: { entities: string[]; terms: string[] };
  paths_examined: number;
  facts_considered: number;
  quarantined_seen: number;
  elapsed_ms: number;
}

export interface Verdict {
  answered: boolean;
  answer: string;
  citations: number[];
  abstained_because: string;
  warrant: Warrant;
  latency_ms: number;
  model: string;
  verified: number;
  checks: string[];
  corpus?: string;
  elapsed_ms?: number;
}

export interface Rejection {
  rule: string;
  reason: string;
  text: string;
  tier: Tier;
  ts: number;
  session?: string;
  fact_id?: number | null;
}

export interface Stats {
  corpus: string;
  sessions: number;
  turns: number;
  facts: number;
  entities: number;
  rejections: number;
  quarantined: number;
  superseded: number;
  answers: number;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  graph: { uri: string; reachable: boolean };
  model: { configured: boolean; answer: string | null; mode: "live" | "cache-only" };
}

export interface Integrity {
  ok: boolean;
  orphan_facts: number;
  dangling_supersedes: number;
  quarantined_warrantable: number;
  corpus?: string;
  [key: string]: unknown;
}

export interface TimelineSession {
  sid: string;
  ts: number;
  title: string;
  facts: number;
}

export interface GraphNode {
  id: string;
  label: string;
  props: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  type: string;
  source: string;
  target: string;
}

export interface AttackResult {
  corpus: string;
  injected: { text: string; tier: Tier; origin: string; session: string };
  ingest: Record<string, number | string>;
  before: Verdict;
  after: Verdict;
  answer_changed: boolean;
  quarantined: number;
  rejections: number;
}

export interface WalkthroughStep {
  id: string;
  question: string;
  as_of?: string;
  why: string;
  expect: Record<string, unknown>;
}

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? body?.error ?? detail;
    } catch {
      /* the body was not JSON; the status text will do */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => call<Health>("/health"),
  stats: (corpus?: string) => call<Stats>(`/stats${corpus ? `?corpus=${encodeURIComponent(corpus)}` : ""}`),
  integrity: (corpus?: string) =>
    call<Integrity>(`/integrity${corpus ? `?corpus=${encodeURIComponent(corpus)}` : ""}`),
  timeline: (corpus?: string) =>
    call<{ corpus: string; sessions: TimelineSession[] }>(
      `/graph/timeline${corpus ? `?corpus=${encodeURIComponent(corpus)}` : ""}`,
    ),
  rejections: (corpus?: string) =>
    call<{ corpus: string; rejections: Rejection[] }>(
      `/rejections${corpus ? `?corpus=${encodeURIComponent(corpus)}` : ""}`,
    ),
  policy: () => call<{ rules: { rule: string; description: string }[] }>("/policy"),
  walkthrough: () => call<{ steps: WalkthroughStep[]; description: string }>("/demo/walkthrough"),
  seed: (force = false) => call<Record<string, unknown>>(`/demo/seed?force=${force}`, { method: "POST" }),

  ask: (question: string, opts: { corpus?: string; asOf?: number | null; record?: boolean } = {}) =>
    call<Verdict>("/ask", {
      method: "POST",
      body: JSON.stringify({
        question,
        corpus: opts.corpus,
        as_of: opts.asOf ?? null,
        record: opts.record ?? true,
      }),
    }),

  attack: (body: { text: string; question: string; tier: Tier; corpus?: string; origin?: string }) =>
    call<AttackResult>("/attack", { method: "POST", body: JSON.stringify(body) }),

  neighbourhood: (params: { corpus?: string; entity?: string; factId?: number; maxLen?: number }) => {
    const q = new URLSearchParams();
    if (params.corpus) q.set("corpus", params.corpus);
    if (params.entity) q.set("entity", params.entity);
    if (params.factId !== undefined) q.set("fact_id", String(params.factId));
    if (params.maxLen) q.set("max_len", String(params.maxLen));
    return call<{ corpus: string; nodes: GraphNode[]; edges: GraphEdge[] }>(`/graph/neighbourhood?${q}`);
  },

  fact: (factId: number, corpus?: string) =>
    call<Record<string, unknown>>(
      `/fact/${factId}${corpus ? `?corpus=${encodeURIComponent(corpus)}` : ""}`,
    ),
};

export { ApiError };
