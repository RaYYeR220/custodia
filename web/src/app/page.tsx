"use client";

import { useEffect, useState } from "react";
import { api, type Health, type Integrity, type Stats } from "@/lib/api";
import { Masthead } from "@/components/chrome";
import { Enquiry } from "@/components/enquiry";
import { AttackConsole } from "@/components/attack";
import { Ledger } from "@/components/ledger";

export default function Page() {
  const [view, setView] = useState("ask");
  const [health, setHealth] = useState<Health | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [integrity, setIntegrity] = useState<Integrity | null>(null);

  useEffect(() => {
    const load = () => {
      api.health().then(setHealth).catch(() => setHealth(null));
      api.stats().then(setStats).catch(() => setStats(null));
      api.integrity().then(setIntegrity).catch(() => setIntegrity(null));
    };
    load();
    const timer = setInterval(load, 15000);
    return () => clearInterval(timer);
  }, [view]);

  const corpus = stats?.corpus;

  return (
    <main className="min-h-screen flex flex-col">
      <Masthead health={health} stats={stats} integrity={integrity} view={view} onView={setView} />

      <div className="flex-1">
        {view === "ask" && <Enquiry corpus={corpus} />}
        {view === "attack" && <AttackConsole corpus={corpus} />}
        {view === "ledger" && <Ledger corpus={corpus} />}
      </div>

      <footer className="border-t border-rule mt-10">
        <div className="max-w-[1400px] mx-auto px-6 md:px-10 py-5 flex flex-wrap items-center justify-between gap-4">
          <span className="label">
            Custodia {health?.version ? `v${health.version}` : ""} · memory on HydraDB
          </span>
          <span className="label">
            {health?.model.mode === "cache-only"
              ? "running from the shipped response cache — no credentials needed"
              : health?.model.answer}
          </span>
        </div>
      </footer>
    </main>
  );
}
