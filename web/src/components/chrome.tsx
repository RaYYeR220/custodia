"use client";

/** Shared furniture: the masthead, status pills, section rules and small marks. */

import { ReactNode } from "react";
import type { Health, Integrity, Stats, Tier } from "@/lib/api";

export const TIER_COLOR: Record<Tier, string> = {
  owner: "text-brass",
  assistant: "text-bone-dim",
  tool: "text-vermilion",
  external: "text-vermilion",
};

export const TIER_NOTE: Record<Tier, string> = {
  owner: "said by the principal",
  assistant: "stated by the assistant",
  tool: "arrived from a tool result",
  external: "arrived from outside the conversation",
};

export function stamp(ts: number): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toISOString().slice(0, 10);
}

export function Rule({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 my-6">
      <div className="h-px flex-1 bg-rule-soft" />
      {label && <span className="label">{label}</span>}
      <div className="h-px flex-1 bg-rule-soft" />
    </div>
  );
}

export function Pill({
  tone = "neutral",
  children,
  title,
}: {
  tone?: "neutral" | "good" | "bad" | "brass";
  children: ReactNode;
  title?: string;
}) {
  const tones = {
    neutral: "text-bone-faint border-rule",
    good: "text-verdigris border-verdigris/40",
    bad: "text-vermilion border-vermilion/50",
    brass: "text-brass border-brass/40",
  } as const;
  return (
    <span title={title} className={`seal ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function Masthead({
  health,
  stats,
  integrity,
  view,
  onView,
}: {
  health: Health | null;
  stats: Stats | null;
  integrity: Integrity | null;
  view: string;
  onView: (v: string) => void;
}) {
  const views = [
    ["ask", "Enquiry"],
    ["attack", "Attack console"],
    ["ledger", "Refusal ledger"],
  ];
  return (
    <header className="border-b border-rule">
      <div className="max-w-[1400px] mx-auto px-6 md:px-10 pt-7 pb-4">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div>
            <h1 className="display text-[2.6rem] leading-none tracking-tight text-bone">
              Custodia
            </h1>
            <p className="text-bone-faint mt-2 text-[12.5px] max-w-[54ch]">
              Agent memory with a chain of custody. Nothing is remembered without its
              source, nothing is answered without a warrant.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Pill tone={health?.graph.reachable ? "good" : "bad"} title={health?.graph.uri}>
              hydradb {health?.graph.reachable ? "live" : "down"}
            </Pill>
            <Pill tone={health?.model.mode === "live" ? "brass" : "neutral"} title={health?.model.answer ?? ""}>
              {health?.model.mode === "live" ? "model live" : "cache only"}
            </Pill>
            {integrity && (
              <Pill tone={integrity.ok ? "good" : "bad"} title="every fact reachable from its source turn">
                provenance {integrity.ok ? "intact" : "broken"}
              </Pill>
            )}
            {stats && (
              <Pill title="what memory currently holds">
                {stats.facts} facts · {stats.sessions} sessions
              </Pill>
            )}
          </div>
        </div>

        <nav className="flex gap-7 mt-7 -mb-px">
          {views.map(([key, title]) => (
            <button
              key={key}
              onClick={() => onView(key)}
              className={`label pb-3 border-b-2 transition-colors ${
                view === key
                  ? "border-brass text-bone"
                  : "border-transparent hover:text-bone-dim"
              }`}
            >
              {title}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}
