"use client";

import { useEffect, useState } from "react";
import { api, type Integrity, type Rejection } from "@/lib/api";
import { Pill, stamp } from "./chrome";

export function Ledger({ corpus }: { corpus?: string }) {
  const [rejections, setRejections] = useState<Rejection[]>([]);
  const [rules, setRules] = useState<{ rule: string; description: string }[]>([]);
  const [integrity, setIntegrity] = useState<Integrity | null>(null);

  useEffect(() => {
    api.rejections(corpus).then((r) => setRejections(r.rejections)).catch(() => {});
    api.policy().then((r) => setRules(r.rules)).catch(() => {});
    api.integrity(corpus).then(setIntegrity).catch(() => {});
  }, [corpus]);

  return (
    <div className="max-w-[1400px] mx-auto px-6 md:px-10 py-8 grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px] gap-10">
      <section className="min-w-0">
        <div className="flex items-baseline justify-between">
          <span className="label">Refused writes</span>
          <span className="label">{rejections.length} on record</span>
        </div>
        <p className="mt-2 text-[12.5px] text-bone-faint max-w-[68ch]">
          A refused write is not discarded. It is stored with the rule that caught it and the
          content that tripped it, because an attack you deleted is an attack you cannot show
          anyone. None of these can enter a warrant.
        </p>

        <div className="mt-5 space-y-3">
          {rejections.length === 0 && (
            <div className="card p-4 text-bone-faint text-[12.5px]">
              Nothing refused in this corpus yet. Fire something from the attack console.
            </div>
          )}
          {rejections.map((r, i) => (
            <article key={i} className="card card--filed p-4 border-vermilion/30">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Pill tone="bad">{r.rule}</Pill>
                  <span className="label">arrived as {r.tier}</span>
                </div>
                <span className="label">{stamp(r.ts)}</span>
              </div>
              <p className="mt-2 text-[13px] text-bone-dim">{r.reason}</p>
              <blockquote className="mt-3 pl-3 border-l-2 border-vermilion/40 text-[12px] text-bone-faint italic whitespace-pre-wrap">
                {r.text.length > 400 ? r.text.slice(0, 400) + "…" : r.text}
              </blockquote>
            </article>
          ))}
        </div>
      </section>

      <aside>
        <div className="sticky top-6 space-y-6">
          <div>
            <span className="label">Provenance</span>
            <div className={`card p-4 mt-3 ${integrity?.ok ? "border-verdigris/40" : "border-vermilion/50"}`}>
              <div className="display text-[1.3rem] text-bone">
                {integrity ? (integrity.ok ? "Intact" : "Broken") : "…"}
              </div>
              <p className="text-[12px] text-bone-faint mt-1">
                Checked live against the graph, not asserted.
              </p>
              {integrity && (
                <dl className="mt-3 space-y-1 text-[12px]">
                  {Object.entries(integrity)
                    .filter(([k]) => !["ok", "corpus"].includes(k))
                    .map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-3">
                        <dt className="text-bone-faint">{k.replace(/_/g, " ")}</dt>
                        <dd className={Number(v) > 0 ? "text-vermilion" : "text-bone-dim"}>{String(v)}</dd>
                      </div>
                    ))}
                </dl>
              )}
            </div>
          </div>

          <div>
            <span className="label">Rules that run on every write</span>
            <div className="mt-3 space-y-2">
              {rules.map((r) => (
                <div key={r.rule} className="border-l-2 border-rule pl-3 py-1">
                  <div className="text-[12.5px] text-brass">{r.rule}</div>
                  <div className="text-[11.5px] text-bone-faint leading-snug">{r.description}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}
