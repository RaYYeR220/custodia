"""Runner: the memory-poisoning suite and its negative control.

    python -m eval.run_poison --limit 20 --systems custodia,rag

Each case is run in its own corpus. The clean history is ingested first, then the
injected session at its declared trust tier, then the question is asked. Three
things are measured per case:

* did the attacker's value come back out (``flipped``);
* was the injected fact quarantined by the policy engine (``quarantined``);
* did any fact get extracted from the injected turns at all
  (``facts_from_injection``).

The third exists to stop the second from lying. If the extractor produced nothing
from an injected turn, the attack failed for a reason that has nothing to do with
the policy engine, and counting it as a quarantine would inflate the headline
number. Those cases are excluded from the quarantine denominator and reported
separately.

The negative control runs through exactly the same path. Its cases are legitimate
owner-tier updates that genuinely change the answer, so a system that quarantines
everything scores a perfect flip rate and an equally visible over-block rate.
Both always appear in the same table.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import typer

from . import LlmBinding, NoProviderConfigured, resolve_llm
from .baselines import BaselineAnswer, FullContextBaseline, VectorlessRagBaseline
from .datasets import (
    DatasetUnavailable,
    EvalSession,
    EvalTurn,
    Instance,
    SOURCES,
    load_dataset,
    manifest,
)
from .poison import (
    ATTACK_FAMILIES,
    AttackCase,
    PoisonRecord,
    build_suite,
    classify,
    load_suite,
    save_suite,
    score_poison,
)
from .run_longmemeval import CustodiaSystem, MissingIntegration, RecordStore

app = typer.Typer(add_completion=False, help=__doc__)

DEFAULT_SUITE = "eval/suites/poison_suite.json"


def apply_case(instance: Instance, case: AttackCase) -> Instance:
    """The instance as the system will see it: clean history plus the injection.

    The injected turns become one extra session carrying the case's own session
    id. That id is what the quarantine check keys on, and it is what keeps the
    injection distinguishable from the genuine history after ingestion.
    """
    injected = EvalSession(
        sid=case.injected_session_id,
        ts=case.injected_at,
        date=case.injected[0].date if case.injected else "",
        turns=[EvalTurn(role=t.role, content=t.content) for t in case.injected],
    )
    sessions = list(instance.sessions) + [injected]
    sessions.sort(key=lambda s: s.ts)
    return Instance(
        qid=instance.qid,
        qtype=instance.qtype,
        question=instance.question,
        answer=instance.answer,
        question_date=instance.question_date,
        asked_at=instance.asked_at,
        sessions=sessions,
        answer_session_ids=instance.answer_session_ids,
        dataset=instance.dataset,
    )


@dataclass(slots=True)
class PoisonedCustodia(CustodiaSystem):
    """Custodia, plus the graph probes the poison metrics need.

    Ingesting the injected session separately is not cosmetic: the tier of a turn
    is what Custodia's policy engine admits or refuses on, and the tier is a
    property of the channel the content arrived on -- so the runner must pass the
    case's declared tier rather than infer one from the message's role.
    """

    name: str = "custodia"

    def ingest_case(self, instance: Instance, case: AttackCase) -> dict[str, Any]:
        stats = self.ingest(instance)  # the clean history, at its natural tiers
        client = self.connect()
        corpus = self.corpus_for(instance)
        from custodia import ingest as ingest_module

        factory = getattr(ingest_module, "Ingestor", None) or getattr(
            ingest_module, "Ingest", None
        )
        if factory is None:
            raise MissingIntegration(
                "the poison runner needs custodia.ingest.Ingestor so the injected "
                "session can be written at its own trust tier"
            )
        writer = factory(client, corpus=corpus)
        for idx, turn in enumerate(case.injected):
            writer.add_turn(
                sid=case.injected_session_id,
                sidx=len(instance.sessions),
                idx=idx,
                role=turn.role,
                text=turn.content,
                ts=turn.ts,
                tier=turn.tier,
                origin=f"poison:{case.family}" if not case.is_control else "",
            )
        injected_stats = writer.flush()
        stats["injected_stats"] = (
            injected_stats if isinstance(injected_stats, dict) else str(injected_stats)
        )
        return stats

    def probe(self, instance: Instance, case: AttackCase) -> dict[str, Any]:
        """Ask the graph what happened to the injected content.

        Uses the client's own ``count`` rather than a Custodia helper, because the
        schema (``Fact.sid``, ``Fact.status``, the ``quarantined`` lifecycle
        state) is fixed in ``custodia.schema`` and stable, while the audit helpers
        are still moving.
        """
        client = self.connect()
        corpus = self.corpus_for(instance)
        out: dict[str, Any] = {}
        try:
            out["facts_from_injection"] = client.count(
                "Fact", corpus=corpus, sid=case.injected_session_id
            )
            out["facts_quarantined"] = client.count(
                "Fact", corpus=corpus, sid=case.injected_session_id, status="quarantined"
            )
        except Exception as exc:
            out["probe_error"] = f"{type(exc).__name__}: {exc}"
            out["facts_from_injection"] = 0
            out["facts_quarantined"] = 0
        try:
            out["rejections"] = client.count("Rejection", corpus=corpus)
        except Exception:
            out["rejections"] = None  # Rejection may not carry a corpus property yet
        return out


@dataclass(slots=True)
class BaselinePoison:
    """A baseline running the poisoned history. It has no policy engine.

    Quarantine is therefore not zero for these systems, it is *undefined*: they
    have no mechanism that could quarantine anything. ``facts_from_injection``
    stays 0, which keeps them out of the quarantine denominator entirely and
    renders their quarantine rate as ``not measured`` rather than as a suspicious
    0%.
    """

    name: str
    baseline: Any

    def run(self, instance: Instance, case: AttackCase) -> tuple[BaselineAnswer, dict[str, Any]]:
        poisoned = apply_case(instance, case)
        return self.baseline.answer(poisoned), {
            "facts_from_injection": 0,
            "facts_quarantined": 0,
            "note": "no policy engine: quarantine is undefined, not zero",
        }


@app.command()
def run(
    limit: int = typer.Option(20, "--limit", help="questions to draw attack cases from"),
    variant: str = typer.Option("s", "--variant", help=f"one of: {', '.join(sorted(SOURCES))}"),
    systems: str = typer.Option("custodia", "--systems", help="comma-separated"),
    seed: int = typer.Option(0, "--seed"),
    out: str = typer.Option("eval/results/poison.json", "--out"),
    suite: str = typer.Option(DEFAULT_SUITE, "--suite", help="pinned suite to load or write"),
    rebuild: bool = typer.Option(False, "--rebuild", help="regenerate the suite even if pinned"),
    resume: bool = typer.Option(False, "--resume"),
    families: str = typer.Option("", "--families", help="comma-separated attack families"),
    context_tokens: int = typer.Option(128_000, "--context-tokens"),
    top_k: int = typer.Option(12, "--top-k"),
    corpus_prefix: str = typer.Option("poison-", "--corpus-prefix"),
) -> None:
    """Build or load the attack suite, run it, and write the poison scorecard."""
    started_at = datetime.now(timezone.utc)
    out_path = Path(out)
    store = RecordStore(out_path.with_suffix(".records.jsonl"))
    if resume:
        store.load()
        typer.echo(f"resuming: {len(store.done)} records already present")

    wanted_families = [f.strip() for f in families.split(",") if f.strip()] or list(ATTACK_FAMILIES)
    skipped: dict[str, str] = {}

    # ---- instances ---------------------------------------------------------
    try:
        instances = load_dataset(variant, limit=limit or None, seed=seed, stratify=True)
    except DatasetUnavailable as exc:
        typer.secho(f"dataset unavailable: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    by_qid = {i.qid: i for i in instances}

    # ---- suite -------------------------------------------------------------
    suite_path = Path(suite)
    if suite_path.exists() and not rebuild:
        cases, suite_meta = load_suite(str(suite_path))
        typer.echo(f"loaded pinned suite {suite_path} ({len(cases)} cases)")
    else:
        cases = build_suite(instances, seed=seed, families=wanted_families)
        digest = save_suite(str(suite_path), cases, seed=seed, dataset=variant)
        suite_meta = {"sha256": digest, "seed": seed, "dataset": variant}
        typer.echo(f"built and pinned {len(cases)} cases to {suite_path}")

    runnable = [c for c in cases if c.qid in by_qid]
    if len(runnable) != len(cases):
        skipped["suite_cases"] = (
            f"{len(cases) - len(runnable)} pinned cases reference questions not in this "
            f"sample (variant={variant}, limit={limit}, seed={seed}); widen --limit or "
            "--rebuild the suite rather than reading the partial run as a full one"
        )
        typer.secho(skipped["suite_cases"], fg=typer.colors.YELLOW)
    attacks = sum(1 for c in runnable if not c.is_control)
    controls = sum(1 for c in runnable if c.is_control)
    typer.echo(f"running {attacks} attacks and {controls} negative controls")
    if controls == 0:
        typer.secho(
            "no negative control cases: flip rate alone is not a result", fg=typer.colors.RED
        )

    # ---- model + systems ---------------------------------------------------
    llm: LlmBinding | None
    try:
        llm = resolve_llm()
    except NoProviderConfigured as exc:
        llm = None
        typer.secho(f"no provider configured: {exc}", fg=typer.colors.YELLOW)

    runners: dict[str, Any] = {}
    for name in (s.strip() for s in systems.split(",")):
        if not name:
            continue
        if name == "custodia":
            runners[name] = PoisonedCustodia(corpus_prefix=corpus_prefix)
        elif name in {"fullcontext", "full"}:
            if llm is None:
                skipped[name] = "no language model provider configured"
                continue
            runners["fullcontext"] = BaselinePoison(
                "fullcontext", FullContextBaseline(llm, context_tokens=context_tokens)
            )
        elif name in {"rag", "bm25"}:
            if llm is None:
                skipped[name] = "no language model provider configured"
                continue
            runners["rag"] = BaselinePoison("rag", VectorlessRagBaseline(llm, top_k=top_k))
        else:
            skipped[name] = "unknown system"

    # ---- the loop ----------------------------------------------------------
    for position, case in enumerate(runnable, start=1):
        instance = by_qid[case.qid]
        for name, system in list(runners.items()):
            key = f"{case.case_id}"
            if resume and store.has(key, name):
                continue
            record = PoisonRecord(
                case_id=case.case_id,
                family=case.family,
                kind=case.kind,
                qid=case.qid,
                system=name,
                question=case.question,
                true_answer=case.true_answer,
                attacker_answer=case.attacker_answer,
                tier=case.tier,
            )
            started = time.perf_counter()
            try:
                if isinstance(system, PoisonedCustodia):
                    record.extra.update(system.ingest_case(instance, case))
                    reply = system.answer(instance)
                    probe = system.probe(instance, case)
                else:
                    reply, probe = system.run(instance, case)
                record.prediction = reply.text
                record.prompt_tokens = reply.prompt_tokens
                record.latency_ms = reply.latency_ms
                record.extra.update(reply.notes)
                record.extra.update(probe)
                record.facts_from_injection = int(probe.get("facts_from_injection") or 0)
                record.quarantined = int(probe.get("facts_quarantined") or 0) > 0
                verdict = classify(case, reply.text)
                record.flipped = verdict["flipped"]
                record.held = verdict["held"]
                record.abstained = verdict["abstained"]
            except MissingIntegration as exc:
                skipped.setdefault(name, str(exc))
                runners.pop(name, None)
                typer.secho(f"{name}: {exc}", fg=typer.colors.YELLOW)
                continue
            except Exception as exc:
                record.error = f"{type(exc).__name__}: {exc}"
                record.latency_ms = round((time.perf_counter() - started) * 1000, 1)
                typer.secho(f"  {case.case_id} [{name}] {record.error}", fg=typer.colors.RED)
            _append(store, record)
        typer.echo(f"[{position}/{len(runnable)}] {case.case_id}")

    # ---- score + write -----------------------------------------------------
    systems_present = sorted({str(r.get("system")) for r in store.rows})
    cards = {
        name: score_poison(
            store.rows, system=name, suite_sha256=str(suite_meta.get("sha256", ""))
        ).as_json()
        for name in systems_present
    }
    source = SOURCES.get(variant if variant in SOURCES else "s")
    recorded = (manifest().get("artifacts") or {}).get(source.key, {}) if source else {}
    document = {
        "provenance": {
            "kind": "poison",
            "run": out_path.stem,
            "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": recorded.get("key", variant),
            "dataset_variant": variant,
            "dataset_sha256": recorded.get("sha256", ""),
            "dataset_source_url": recorded.get("url", source.url if source else ""),
            "sample_size": len(instances),
            "seed": seed,
            "systems": systems_present,
            "answer_model": llm.model if llm else "",
            "llm_binding": llm.origin if llm else "none",
            "suite_path": str(suite_path),
            "suite_sha256": suite_meta.get("sha256", ""),
            "attack_cases": attacks,
            "control_cases": controls,
            "note_control": (
                "flip rate must be read beside legitimate update accuracy: a system that "
                "refuses every write scores a perfect flip rate and a useless product"
            ),
            "note_matching": (
                "attacker values are constructed strings, so flip detection is exact "
                "containment rather than an LLM judgement"
            ),
        },
        "poison": cards,
        "skipped": skipped,
        "records_file": str(store.path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    from .report import build_report

    typer.echo(build_report(document))
    typer.echo(f"\nwrote {out_path} and {store.path}")


def _append(store: RecordStore, record: PoisonRecord) -> None:
    """Append a poison record through the same append-only store as the QA runner."""
    payload = record.as_dict()
    store.path.parent.mkdir(parents=True, exist_ok=True)
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    store.rows.append(payload)
    store.done.add((record.case_id, record.system))


if __name__ == "__main__":
    app()
