"""Runner: LongMemEval (and siblings) across Custodia and the two baselines.

    python -m eval.run_longmemeval --limit 50 --systems custodia,fullcontext,rag

A full run is long -- LongMemEval-S is ~50 sessions and ~122k estimated tokens per
question -- so the runner is built to be interrupted. Every ``(question, system)``
result is appended to a JSONL sidecar the moment it is produced; ``--resume`` reads
that file and skips what is already there. The scorecard is always recomputed from
the sidecar, never accumulated in memory, so a resumed run and an uninterrupted
one produce the same numbers.

Custodia is imported inside the functions that use it. The harness therefore still
runs the baselines when the memory layer is unavailable, and says so in the
result file's ``skipped`` block rather than quietly reporting a two-system
comparison as a three-system one.

The contract this runner expects from ``src/custodia`` is documented on
:class:`CustodiaSystem`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import typer

from . import LlmBinding, NoProviderConfigured, estimate_tokens, resolve_llm
from .baselines import BaselineAnswer, FullContextBaseline, VectorlessRagBaseline
from .datasets import (
    DatasetUnavailable,
    Instance,
    SOURCES,
    dataset_stats,
    load_dataset,
    manifest,
)
from .scorers import (
    RunRecord,
    fallback_judge,
    judge,
    looks_like_abstention,
    score_systems,
)

app = typer.Typer(add_completion=False, help=__doc__)

DEFAULT_SYSTEMS = "custodia,fullcontext,rag"



def _custodia_extractor() -> Any:
    """Extraction bound to a live model, with an eval-sized window.

    The default six-turn window is right for a chat session being ingested as it
    happens; over a fifty-session benchmark haystack it costs three times the
    calls for no extra recall, and the provider's request ceiling is the binding
    constraint on a full run. LongMemEval is written in the first person, so the
    principal is simply "user".
    """
    from custodia.extract import extract_session
    from custodia.llm import LLM

    llm = LLM()
    return lambda turns: extract_session(
        turns, llm=llm, principal="user", window=16, overlap=3
    )

class MissingIntegration(RuntimeError):
    """A Custodia module or entry point the runner needs is not present yet."""


def _first(module: Any, names: Sequence[str]) -> Any | None:
    for name in names:
        attribute = getattr(module, name, None)
        if attribute is not None:
            return attribute
    return None


# --------------------------------------------------------------------------- #
# the system under test
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CustodiaSystem:
    """Adapter onto Custodia: ingest one instance, then ask its question.

    It binds to the interfaces ``src/custodia`` actually publishes:

    ingest
        ``custodia.ingest.ingest_sessions(client, corpus, sessions)`` where a
        session is a mapping with ``sid`` / ``ts`` / ``turns``, and a turn is a
        mapping with ``role`` / ``text`` / ``ts`` / ``tier`` / ``origin``.
        Sessions are handed over as plain mappings rather than :class:`EvalTurn`
        objects on purpose: the eval record calls its field ``content`` and has
        no tier, and letting the ingestor infer either would put the harness's
        assumptions inside the system under test.

    retrieve + answer
        ``custodia.retrieve.Retriever(client, corpus, index=...)`` feeding
        ``custodia.gate.Gate(retriever).ask(question, as_of=...)``, which returns
        a ``Verdict`` carrying ``answered`` / ``answer`` / ``citations`` /
        ``abstained_because`` / ``warrant``.

    The lexical index is rebuilt from the graph after each ingest, because the
    retriever seeds on it and a retriever handed ``index=None`` silently loses
    every fact no entity extractor caught -- which would understate Custodia and
    still look like a result.

    Anything missing raises :class:`MissingIntegration` naming what was looked
    for. The runner records that as a skipped system; it never invents an answer.
    """

    name: str = "custodia"
    corpus_prefix: str = "eval-"
    client: Any = None
    llm: Any = None

    def connect(self) -> Any:
        if self.client is None:
            from custodia.config import settings
            from custodia.hydra.client import HydraClient

            config = settings()
            self.client = HydraClient(config.hydra_uri, config.hydra_token)
            if not self.client.ping(retries=5, delay=1.0):
                raise MissingIntegration(
                    f"HydraDB is not answering at {config.hydra_uri}; "
                    "start it before running the custodia system"
                )
        return self.client

    def corpus_for(self, instance: Instance) -> str:
        """One corpus per question, so haystacks cannot leak into each other."""
        return f"{self.corpus_prefix}{instance.qid}"

    @staticmethod
    def as_payload(sessions: Sequence[Any]) -> list[dict[str, Any]]:
        """Eval sessions as the mappings ``ingest_sessions`` accepts.

        Turn timestamps are the session time plus the turn's position, so order
        within a session survives into the graph. Without that every turn in a
        session shares one timestamp and any within-session ordering question
        becomes unanswerable for reasons that are the harness's fault.
        """
        payload: list[dict[str, Any]] = []
        for session in sessions:
            payload.append(
                {
                    "sid": session.sid,
                    "ts": session.ts,
                    "turns": [
                        {
                            "role": turn.role,
                            "text": turn.content,
                            "ts": session.ts + idx,
                            "idx": idx,
                            "tier": _tier_for(turn.role),
                            "origin": "",
                        }
                        for idx, turn in enumerate(session.turns)
                    ],
                }
            )
        return payload

    def ingest(self, instance: Instance) -> dict[str, Any]:
        client = self.connect()
        corpus = self.corpus_for(instance)
        try:
            from custodia import ingest as ingest_module
        except ImportError as exc:
            raise MissingIntegration(f"custodia.ingest is not importable: {exc}") from exc

        fn = _first(ingest_module, ("ingest_sessions", "ingest_instance"))
        if fn is None:
            raise MissingIntegration(
                "custodia.ingest exposes neither ingest_sessions nor ingest_instance"
            )
        started = time.perf_counter()
        report = fn(
            client,
            corpus,
            self.as_payload(instance.sessions),
            extract=_custodia_extractor(),
        )
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "corpus": corpus,
            "ingest_ms": round(elapsed, 1),
            "ingest_report": _report_dict(report),
        }

    def build_index(self, corpus: str) -> Any:
        """Rebuild the BM25 index the retriever seeds on. ``None`` if unavailable."""
        try:
            from custodia.lexical import LexicalIndex
        except ImportError:
            return None
        try:
            return LexicalIndex.build(self.connect(), corpus)
        except Exception:
            return None

    def answer(self, instance: Instance) -> BaselineAnswer:
        client = self.connect()
        corpus = self.corpus_for(instance)
        try:
            from custodia.gate import Gate
            from custodia.retrieve import Retriever
        except ImportError as exc:
            raise MissingIntegration(f"custodia.gate/retrieve is not importable: {exc}") from exc

        from custodia.llm import LLM

        llm = LLM()
        index = self.build_index(corpus)
        retriever = Retriever(client, corpus, index=index, llm=llm)
        gate = Gate(retriever, llm=llm)
        started = time.perf_counter()
        verdict = gate.ask(instance.question, as_of=None)
        elapsed = (time.perf_counter() - started) * 1000

        text = str(getattr(verdict, "answer", "") or "")
        answered = getattr(verdict, "answered", None)
        citations = getattr(verdict, "citations", None) or []
        warrant = getattr(verdict, "warrant", None)
        prompt_tokens = estimate_tokens(_warrant_text(warrant)) if warrant is not None else 0

        return BaselineAnswer(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            latency_ms=round(getattr(verdict, "latency_ms", 0) or elapsed, 1),
            truncated=False,
            notes={
                # the gate's own structured flag, not a phrase heuristic
                "abstained_flag": (not bool(answered)) if answered is not None else None,
                "abstained_because": str(getattr(verdict, "abstained_because", "") or ""),
                "citations": len(citations),
                "verified": getattr(verdict, "verified", None),
                "warrant_facts": len(getattr(warrant, "evidence", []) or []),
                "warrant_quarantined_seen": getattr(warrant, "quarantined_seen", None),
                "index": "custodia.lexical" if index is not None else "none (lexical seeding off)",
                "corpus": corpus,
            },
        )

    def provenance(self) -> dict[str, Any]:
        return {"system": self.name, "corpus_prefix": self.corpus_prefix}


def _report_dict(report: Any) -> Any:
    if isinstance(report, dict):
        return report
    fields = getattr(type(report), "__dataclass_fields__", None)
    if fields:
        return {name: getattr(report, name, None) for name in fields}
    return str(report)


def _warrant_text(warrant: Any) -> str:
    """The evidence text a warrant would put in front of the model."""
    evidence = getattr(warrant, "evidence", None)
    if evidence is None:
        return str(warrant)
    return "\n".join(str(getattr(e, "text", e)) for e in evidence)


def _tier_for(role: str) -> str:
    """Map a benchmark role to a Custodia trust tier by name.

    LongMemEval histories are entirely principal/assistant, so everything lands
    at ``owner`` or ``assistant``; the poison suite is where the lower tiers get
    exercised.
    """
    lowered = (role or "").strip().lower()
    if lowered in {"user", "human", "owner"}:
        return "owner"
    if lowered in {"assistant", "ai", "agent", "bot"}:
        return "assistant"
    if lowered in {"tool", "function", "observation"}:
        return "tool"
    return "external"


# --------------------------------------------------------------------------- #
# record store (resume)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RecordStore:
    """Append-only JSONL of per-question records; the unit of resumability."""

    path: Path
    done: set[tuple[str, str]] = field(default_factory=set)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a hard kill; drop it and move on
            self.rows.append(row)
            # QA records key on the question id, poison records on the case id;
            # one store serves both so resume behaves identically in each runner
            key = row.get("case_id") or row.get("qid")
            self.done.add((str(key), str(row.get("system"))))

    def append(self, record: RunRecord) -> None:
        payload = record.as_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.rows.append(payload)
        self.done.add((record.qid, record.system))

    def has(self, qid: str, system: str) -> bool:
        return (qid, system) in self.done


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #


def _grade(
    instance: Instance,
    prediction: str,
    *,
    judge_llm: LlmBinding | None,
) -> tuple[bool, str, str]:
    qtype = "abstention" if instance.is_abstention else instance.qtype
    if judge_llm is not None:
        verdict = judge(instance.question, instance.answer, prediction, qtype, llm=judge_llm)
        if verdict.method != "llm-judge-failed":
            return verdict.correct, verdict.method, verdict.reason
        # a failed judge call is recorded as such, not retried into the fallback
        return verdict.correct, verdict.method, verdict.reason
    verdict = fallback_judge(instance.question, instance.answer, prediction, qtype)
    return verdict.correct, verdict.method, verdict.reason


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@app.command()
def run(
    limit: int = typer.Option(0, "--limit", help="questions to sample; 0 = the whole dataset"),
    variant: str = typer.Option("s", "--variant", help=f"one of: {', '.join(sorted(SOURCES))}"),
    systems: str = typer.Option(DEFAULT_SYSTEMS, "--systems", help="comma-separated"),
    seed: int = typer.Option(0, "--seed", help="sampling seed; the sample is deterministic"),
    out: str = typer.Option("eval/results/longmemeval.json", "--out"),
    resume: bool = typer.Option(False, "--resume", help="skip questions already in the sidecar"),
    types: str = typer.Option("", "--types", help="comma-separated question types to keep"),
    judge_mode: str = typer.Option(
        "auto", "--judge", help="llm | fallback | auto (llm when a provider is configured)"
    ),
    context_tokens: int = typer.Option(
        128_000, "--context-tokens", help="input budget assumed for the full-context baseline"
    ),
    top_k: int = typer.Option(12, "--top-k", help="chunks the RAG baseline retrieves"),
    corpus_prefix: str = typer.Option("eval-", "--corpus-prefix"),
    allow_large: bool = typer.Option(False, "--allow-large", help="permit >1 GB downloads"),
) -> None:
    """Ingest, ask, score and write a resumable result file."""
    started_at = datetime.now(timezone.utc)
    out_path = Path(out)
    store = RecordStore(out_path.with_suffix(".records.jsonl"))
    if resume:
        store.load()
        typer.echo(f"resuming: {len(store.done)} records already present")

    wanted = [s.strip() for s in systems.split(",") if s.strip()]
    type_filter = [t.strip() for t in types.split(",") if t.strip()] or None
    skipped: dict[str, str] = {}

    # ---- dataset -----------------------------------------------------------
    try:
        instances = load_dataset(
            variant,
            limit=limit or None,
            seed=seed,
            stratify=True,
            types=type_filter,
            allow_large=allow_large,
        )
    except DatasetUnavailable as exc:
        typer.secho(f"dataset unavailable: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    stats = dataset_stats(instances)
    typer.echo(
        f"loaded {stats['instances']} instances "
        f"({stats['abstention']['items']} abstention), "
        f"~{stats['tokens_estimated']['per_instance_mean']:,.0f} estimated tokens each"
    )

    # ---- model -------------------------------------------------------------
    llm: LlmBinding | None
    try:
        llm = resolve_llm()
    except NoProviderConfigured as exc:
        llm = None
        typer.secho(f"no provider configured: {exc}", fg=typer.colors.YELLOW)

    use_llm_judge = judge_mode == "llm" or (judge_mode == "auto" and llm is not None)
    if judge_mode == "llm" and llm is None:
        typer.secho("--judge llm requested but no provider is configured", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    judge_llm = llm if use_llm_judge else None
    if judge_llm is None:
        skipped["judge"] = (
            "graded by the non-LLM lexical fallback, which is materially weaker than "
            "the rubric judge and under-reports paraphrased answers"
        )

    # ---- systems -----------------------------------------------------------
    runners: dict[str, Any] = {}
    provenance_systems: dict[str, Any] = {}
    for name in wanted:
        if name == "custodia":
            runners[name] = CustodiaSystem(corpus_prefix=corpus_prefix)
            provenance_systems[name] = runners[name].provenance()
        elif name in {"fullcontext", "full", "longcontext"}:
            if llm is None:
                skipped[name] = "no language model provider configured"
                continue
            baseline = FullContextBaseline(llm, context_tokens=context_tokens)
            runners["fullcontext"] = baseline
            provenance_systems["fullcontext"] = baseline.provenance()
        elif name in {"rag", "bm25"}:
            if llm is None:
                skipped[name] = "no language model provider configured"
                continue
            baseline = VectorlessRagBaseline(llm, top_k=top_k)
            runners["rag"] = baseline
            provenance_systems["rag"] = baseline.provenance()
        else:
            skipped[name] = "unknown system"
    if not runners:
        typer.secho("no runnable systems; nothing to measure", fg=typer.colors.RED)

    # ---- the loop ----------------------------------------------------------
    for position, instance in enumerate(instances, start=1):
        for name, system in list(runners.items()):
            if resume and store.has(instance.qid, name):
                continue
            record = RunRecord(
                qid=instance.qid,
                system=name,
                qtype=instance.qtype,
                is_abstention=instance.is_abstention,
                question=instance.question,
                gold=instance.answer,
            )
            try:
                if isinstance(system, CustodiaSystem):
                    record.extra.update(system.ingest(instance))
                reply: BaselineAnswer = system.answer(instance)
                record.prediction = reply.text
                record.prompt_tokens = reply.prompt_tokens
                record.latency_ms = reply.latency_ms
                record.truncated = reply.truncated
                record.extra.update(reply.notes)
                flag = reply.notes.get("abstained_flag")
                record.abstained = (
                    bool(flag) if flag is not None else looks_like_abstention(reply.text)
                )
                correct, method, reason = _grade(instance, reply.text, judge_llm=judge_llm)
                record.correct = correct
                record.judge_method = method
                record.judge_reason = reason
            except MissingIntegration as exc:
                skipped.setdefault(name, str(exc))
                runners.pop(name, None)
                typer.secho(f"{name}: {exc}", fg=typer.colors.YELLOW)
                continue
            except Exception as exc:  # measured as a failure, never as a score
                record.error = f"{type(exc).__name__}: {exc}"
                record.judge_method = "not-graded"
                typer.secho(f"  {instance.qid} [{name}] {record.error}", fg=typer.colors.RED)
            store.append(record)
        typer.echo(f"[{position}/{len(instances)}] {instance.qid}")

    # ---- score + write -----------------------------------------------------
    cards = score_systems(store.rows)
    source = SOURCES.get(variant if variant in SOURCES else "s")
    recorded = (manifest().get("artifacts") or {}).get(source.key, {}) if source else {}
    document = {
        "provenance": {
            "kind": "longmemeval",
            "run": out_path.stem,
            "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": recorded.get("key", variant),
            "dataset_variant": variant,
            "dataset_sha256": recorded.get("sha256", ""),
            "dataset_source_url": recorded.get("url", source.url if source else ""),
            "dataset_items_available": recorded.get("items", ""),
            "sample_size": len(instances),
            "seed": seed,
            "stratified": True,
            "types": type_filter or "all",
            "systems": sorted(runners),
            "answer_model": llm.model if llm else "",
            "judge_model": judge_llm.model if judge_llm else "",
            "llm_binding": llm.origin if llm else "none",
            "judge_mode": "llm-judge" if judge_llm else "lexical-fallback (weaker)",
            "system_settings": provenance_systems,
        },
        "dataset_stats": stats,
        "scorecards": {name: card.as_json() for name, card in cards.items()},
        "skipped": skipped,
        "records_file": str(store.path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    from .report import build_report

    typer.echo(build_report(document))
    typer.echo(f"\nwrote {out_path} and {store.path}")


if __name__ == "__main__":
    app()
