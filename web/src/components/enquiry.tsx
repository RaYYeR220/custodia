"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { api, type EvidenceItem, type TimelineSession, type Verdict, type WalkthroughStep } from "@/lib/api";
import { Pill, TIER_COLOR, TIER_NOTE, stamp } from "./chrome";

const ABSTENTION_TITLE: Record<string, string> = {
  "no-evidence": "No warrant",
  "insufficient-evidence": "No warrant",
  "no-citations": "No warrant",
  "invented-citation": "Citation refused",
  "unverified-citation": "Citation refused",
  "model-unavailable": "Held",
  "malformed-response": "Held",
};

export function Enquiry({ corpus }: { corpus?: string }) {
  const [question, setQuestion] = useState("");
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [asOf, setAsOf] = useState<number | null>(null);
  const [sessions, setSessions] = useState<TimelineSession[]>([]);
  const [steps, setSteps] = useState<WalkthroughStep[]>([]);
  const [focus, setFocus] = useState<number | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    api.timeline(corpus).then((r) => setSessions(r.sessions)).catch(() => {});
    api.walkthrough().then((r) => setSteps(r.steps)).catch(() => {});
  }, [corpus]);

  const bounds = useMemo(() => {
    if (!sessions.length) return null;
    const first = sessions[0].ts;
    const last = sessions[sessions.length - 1].ts;
    return { first, last: last + 86400 };
  }, [sessions]);

  async function run(q: string, moment: number | null) {
    if (!q.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setVerdict(await api.ask(q, { corpus, asOf: moment }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setVerdict(null);
    } finally {
      setBusy(false);
    }
  }

  const cited = new Set(verdict?.citations ?? []);
  const citedEvidence = (verdict?.warrant.evidence ?? []).filter((e) => cited.has(e.fact_id));
  const restEvidence = (verdict?.warrant.evidence ?? []).filter((e) => !cited.has(e.fact_id));

  return (
    <div className="max-w-[1400px] mx-auto px-6 md:px-10 py-8 grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_400px] gap-10">
      {/* ------------------------------------------------------------ docket */}
      <section className="min-w-0">
        <div className="card ruled p-5">
          <div className="flex items-baseline justify-between mb-3">
            <span className="label">Enquiry</span>
            {asOf && bounds && (
              <span className="label text-brass">as memory stood {stamp(asOf)}</span>
            )}
          </div>
          <textarea
            ref={inputRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                run(question, asOf);
              }
            }}
            rows={2}
            placeholder="Ask memory something…"
            className="w-full bg-transparent outline-none resize-none statement text-bone placeholder:text-bone-faint/50"
          />
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-rule-soft">
            <span className="label">↵ to ask · ⇧↵ for a new line</span>
            <button
              onClick={() => run(question, asOf)}
              disabled={busy || !question.trim()}
              className="seal border-brass text-brass hover:bg-brass hover:text-ground transition-colors disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-brass"
            >
              {busy ? "searching" : "ask"}
            </button>
          </div>
        </div>

        {/* time scrubber */}
        {bounds && (
          <div className="mt-4 card p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="label">Ask as memory stood</span>
              <button
                onClick={() => {
                  setAsOf(null);
                  if (verdict) run(verdict.warrant.question, null);
                }}
                className={`label ${asOf ? "text-brass hover:underline" : "opacity-30"}`}
              >
                now
              </button>
            </div>
            <input
              type="range"
              min={bounds.first}
              max={bounds.last}
              step={3600}
              value={asOf ?? bounds.last}
              onChange={(e) => setAsOf(Number(e.target.value))}
              onMouseUp={() => verdict && run(verdict.warrant.question, asOf)}
              onTouchEnd={() => verdict && run(verdict.warrant.question, asOf)}
              className="w-full accent-[var(--color-brass)]"
            />
            <div className="relative h-6 mt-1">
              {sessions.map((s) => {
                const pct = ((s.ts - bounds.first) / (bounds.last - bounds.first)) * 100;
                return (
                  <span
                    key={s.sid}
                    title={`${s.sid} · ${s.facts} facts`}
                    style={{ left: `${pct}%` }}
                    className="absolute top-0 -translate-x-1/2 w-px h-2 bg-rule"
                  />
                );
              })}
              <span className="absolute left-0 top-2 label">{stamp(bounds.first)}</span>
              <span className="absolute right-0 top-2 label">{stamp(bounds.last)}</span>
            </div>
          </div>
        )}

        {/* suggestions */}
        {!verdict && steps.length > 0 && (
          <div className="mt-6">
            <span className="label">Worth asking</span>
            <div className="mt-3 space-y-2">
              {steps.slice(0, 6).map((s) => (
                <button
                  key={s.id}
                  onClick={() => {
                    setQuestion(s.question);
                    const moment = s.as_of ? Math.floor(Date.parse(s.as_of) / 1000) : null;
                    setAsOf(moment);
                    run(s.question, moment);
                  }}
                  className="block w-full text-left px-3 py-2 border border-rule-soft hover:border-brass/50 hover:bg-ground-2 transition-colors group"
                >
                  <span className="text-bone-dim group-hover:text-bone">{s.question}</span>
                  {s.as_of && <span className="label ml-2 text-brass">as of {s.as_of.slice(0, 10)}</span>}
                  <span className="block note mt-1.5">{s.why}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="mt-6 card p-4 border-vermilion/50">
            <span className="label text-vermilion">service</span>
            <p className="mt-1 text-bone-dim">{error}</p>
          </div>
        )}

        {/* ---------------------------------------------------------- answer */}
        {verdict && (
          <div className="mt-8" key={`${verdict.warrant.question}-${verdict.warrant.as_of}`}>
            {verdict.answered ? (
              <div className="file-in">
                <span className="label">On the record</span>
                <p className="statement mt-3 text-bone">{verdict.answer}</p>
                <div className="flex flex-wrap items-center gap-2 mt-5">
                  <span className="label">cited</span>
                  {citedEvidence.map((e, i) => (
                    <button
                      key={e.fact_id}
                      onMouseEnter={() => setFocus(e.fact_id)}
                      onMouseLeave={() => setFocus(null)}
                      className={`seal transition-colors ${
                        focus === e.fact_id
                          ? "border-brass text-ground bg-brass"
                          : "border-brass/40 text-brass"
                      }`}
                    >
                      {i + 1} · {e.session}
                    </button>
                  ))}
                  {verdict.verified > 0 && (
                    <Pill tone="good" title="citations that survived the second-pass entailment check">
                      {verdict.verified} verified
                    </Pill>
                  )}
                </div>
              </div>
            ) : (
              <div className="mt-4">
                <div className="stamp text-[2.1rem] md:text-[2.6rem]">
                  {ABSTENTION_TITLE[verdict.abstained_because] ?? "No warrant"}
                </div>
                <p className="mt-6 text-bone-dim max-w-[60ch]">{verdict.answer}</p>
                <div className="mt-5 border-l-2 border-vermilion/40 pl-4 space-y-1">
                  <div className="label">Search record</div>
                  <div className="text-[12.5px] text-bone-faint">
                    entities: {verdict.warrant.seeds.entities.join(", ") || "none matched"}
                  </div>
                  <div className="text-[12.5px] text-bone-faint">
                    terms: {verdict.warrant.seeds.terms.slice(0, 12).join(", ") || "none"}
                  </div>
                  <div className="text-[12.5px] text-bone-faint">
                    {verdict.warrant.paths_examined} paths examined ·{" "}
                    {verdict.warrant.facts_considered} facts considered ·{" "}
                    {verdict.warrant.evidence.length} met the evidence floor
                  </div>
                  <div className="text-[12.5px] text-bone-faint">
                    refused at: <span className="text-vermilion">{verdict.abstained_because}</span>
                  </div>
                </div>
              </div>
            )}

            <div className="flex flex-wrap gap-3 mt-6 text-[11px] text-bone-faint">
              <span>{verdict.warrant.elapsed_ms.toFixed(0)} ms retrieval</span>
              <span>·</span>
              <span>{verdict.latency_ms.toFixed(0)} ms total</span>
              <span>·</span>
              <span>{verdict.model || "no model"}</span>
              {verdict.warrant.quarantined_seen > 0 && (
                <>
                  <span>·</span>
                  <span className="text-vermilion">
                    {verdict.warrant.quarantined_seen} quarantined fact
                    {verdict.warrant.quarantined_seen === 1 ? "" : "s"} retrieved and refused
                  </span>
                </>
              )}
            </div>

            {verdict.checks.length > 0 && (
              <details className="mt-4">
                <summary className="label cursor-pointer hover:text-bone-dim">
                  gate checks ({verdict.checks.length})
                </summary>
                <ul className="mt-2 space-y-1 text-[12px] text-bone-faint">
                  {verdict.checks.map((c) => (
                    <li key={c}>· {c}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}
      </section>

      {/* ------------------------------------------------------ evidence rail */}
      <aside className="min-w-0">
        <div className="sticky top-6">
          <div className="flex items-baseline justify-between">
            <span className="label">Evidence</span>
            {verdict && (
              <span className="label">
                {citedEvidence.length} cited of {verdict.warrant.evidence.length}
              </span>
            )}
          </div>

          {!verdict && (
            <>
              <p className="mt-4 note">
                Every answer is assembled from facts, and every fact is filed against the
                turn that produced it. Ask something and the cards that were used appear
                here, with the session, the date and the channel each one arrived on.
              </p>
              {sessions.length > 0 && (
                <div className="mt-6">
                  <div className="flex items-baseline justify-between">
                    <span className="label">What memory holds</span>
                    <span className="label">{sessions.length} sessions</span>
                  </div>
                  <ol className="mt-3 border-l border-rule-soft">
                    {sessions.map((s) => (
                      <li key={s.sid} className="relative pl-4 py-1.5">
                        <span className="absolute left-0 top-3.5 -translate-x-1/2 w-1.5 h-1.5 bg-brass-dim rounded-full" />
                        <div className="flex items-baseline justify-between gap-3">
                          <span className="text-[12.5px] text-bone-dim truncate">{s.title || s.sid}</span>
                          <span className="label shrink-0">{stamp(s.ts)}</span>
                        </div>
                        <span className="note">
                          {s.facts} fact{s.facts === 1 ? "" : "s"}
                        </span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </>
          )}

          <div className="mt-4 space-y-3 max-h-[calc(100vh-10rem)] overflow-y-auto pr-1">
            {citedEvidence.map((e, i) => (
              <EvidenceCard
                key={e.fact_id}
                item={e}
                index={i + 1}
                cited
                focused={focus === e.fact_id}
                onFocus={setFocus}
                corpus={corpus}
                delay={i * 60}
              />
            ))}
            {restEvidence.length > 0 && (
              <details className="pt-2">
                <summary className="label cursor-pointer hover:text-bone-dim">
                  {restEvidence.length} retrieved but not cited
                </summary>
                <div className="mt-3 space-y-3 opacity-60">
                  {restEvidence.map((e) => (
                    <EvidenceCard
                      key={e.fact_id}
                      item={e}
                      cited={false}
                      focused={focus === e.fact_id}
                      onFocus={setFocus}
                      corpus={corpus}
                    />
                  ))}
                </div>
              </details>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
}

function EvidenceCard({
  item,
  index,
  cited,
  focused,
  onFocus,
  corpus,
  delay = 0,
}: {
  item: EvidenceItem;
  index?: number;
  cited: boolean;
  focused: boolean;
  onFocus: (id: number | null) => void;
  corpus?: string;
  delay?: number;
}) {
  const [chain, setChain] = useState<Record<string, unknown> | null>(null);
  const [open, setOpen] = useState(false);

  const quarantined = item.status === "quarantined";
  const superseded = item.status === "superseded";

  return (
    <article
      onMouseEnter={() => onFocus(item.fact_id)}
      onMouseLeave={() => onFocus(null)}
      style={{ animationDelay: `${delay}ms` }}
      className={`card card--filed p-3.5 file-in transition-colors ${
        focused ? "border-brass" : ""
      } ${quarantined ? "border-vermilion/50" : ""}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          {index !== undefined && (
            <span className="display text-brass text-[15px] leading-none">{index}</span>
          )}
          <span className={`label ${TIER_COLOR[item.tier]}`} title={TIER_NOTE[item.tier]}>
            {item.tier}
          </span>
        </div>
        <span className="label">{stamp(item.turn_ts || item.valid_from)}</span>
      </div>

      <p
        className={`mt-2 text-[13.5px] leading-relaxed ${
          quarantined ? "line-through text-bone-faint" : "text-bone"
        }`}
      >
        {item.text}
      </p>

      <div className="flex flex-wrap items-center gap-2 mt-3">
        {quarantined && <Pill tone="bad">refused</Pill>}
        {superseded && <Pill tone="neutral">superseded</Pill>}
        {item.superseded_by && <Pill tone="neutral">replaced</Pill>}
        {!cited && !quarantined && <Pill tone="neutral">not cited</Pill>}
        <span className="label">{item.session}</span>
        {item.hops > 0 && <span className="label">{item.hops} hops</span>}
      </div>

      {item.turn_text && (
        <blockquote className="mt-3 pl-3 border-l border-rule text-[12px] text-bone-faint italic">
          {item.turn_text.length > 220 ? item.turn_text.slice(0, 220) + "…" : item.turn_text}
        </blockquote>
      )}

      <button
        onClick={async () => {
          setOpen((v) => !v);
          if (!chain) {
            try {
              setChain(await api.fact(item.fact_id, corpus));
            } catch {
              /* the chain view is supplementary; a failure should not break the card */
            }
          }
        }}
        className="label mt-3 hover:text-brass transition-colors"
      >
        {open ? "hide chain" : "chain of custody"} · fact {item.fact_id}
      </button>

      {open && (
        <div className="mt-2">
          {item.path.length > 0 && (
            <div className="text-[11.5px] text-brass-dim font-mono">{item.path.join("  →  ")}</div>
          )}
          {chain && (
            <pre className="mt-2 text-[10.5px] text-bone-faint overflow-x-auto whitespace-pre-wrap">
              {JSON.stringify(chain, null, 1)}
            </pre>
          )}
        </div>
      )}
    </article>
  );
}
