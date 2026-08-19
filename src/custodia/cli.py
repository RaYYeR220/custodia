"""Command line surface.

``custodia verify`` is the one that matters to a reviewer: it seeds the demo
memory, runs every scripted question, checks the graph's own integrity, and
prints a pass/fail table. Nothing in it is pre-recorded -- each line is the
result of a live query against the running graph.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from custodia import __version__, schema
from custodia.config import settings
from custodia.hydra import HydraClient, HydraError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Agent memory with a chain of custody, on HydraDB.",
)
console = Console()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _client() -> HydraClient:
    cfg = settings()
    client = HydraClient(cfg.hydra_uri, cfg.hydra_token)
    if not client.ping(retries=3, delay=1.0):
        console.print(
            Panel(
                f"Cannot reach HydraDB at [bold]{cfg.hydra_uri}[/].\n\n"
                "Start it with [bold]docker compose up -d hydradb[/] and try again.",
                title="graph unreachable",
                border_style="red",
            )
        )
        raise typer.Exit(code=2)
    return client


def _gate(client: HydraClient, corpus: str) -> Any:
    from custodia.audit import Auditor
    from custodia.gate import Gate
    from custodia.lexical import LexicalIndex
    from custodia.llm import LLM
    from custodia.retrieve import Retriever

    llm = LLM()
    retriever = Retriever(client, corpus, index=LexicalIndex.build(client, corpus), llm=llm)
    return Gate(retriever, llm=llm, auditor=Auditor(client, corpus))


def _corpus(value: str | None) -> str:
    return value or settings().corpus


def _moment(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        console.print(f"[red]could not read a date from {value!r}[/]")
        raise typer.Exit(code=2)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _stamp(ts: int) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _print_verdict(verdict: Any) -> None:
    if verdict.answered:
        console.print(Panel(verdict.answer, title="answer", border_style="green"))
        table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
        table.add_column("fact", style="cyan", no_wrap=True)
        table.add_column("claim")
        table.add_column("tier", style="dim")
        table.add_column("recorded", style="dim")
        table.add_column("session", style="dim")
        cited = set(verdict.citations)
        for ev in verdict.warrant.evidence:
            if ev.fid in cited:
                table.add_row(str(ev.fid), ev.text, ev.tier, _stamp(ev.turn_ts or ev.valid_from), ev.sid)
        console.print(table)
    else:
        console.print(
            Panel(
                verdict.answer or "Memory has nothing that supports this.",
                title=f"declined ({verdict.abstained_because})",
                border_style="yellow",
            )
        )
    extra = []
    if verdict.warrant.quarantined_seen:
        extra.append(f"{verdict.warrant.quarantined_seen} quarantined fact(s) retrieved and refused")
    extra.append(f"{len(verdict.warrant.evidence)} fact(s) in the warrant")
    extra.append(f"{verdict.warrant.paths_examined} path(s) examined")
    extra.append(f"{verdict.latency_ms:.0f} ms")
    console.print("[dim]" + " · ".join(extra) + "[/]")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


@app.command()
def version() -> None:
    """Print the version and the configured endpoints."""
    cfg = settings()
    console.print(f"custodia {__version__}")
    console.print(f"[dim]graph[/] {cfg.hydra_uri}")
    console.print(f"[dim]model[/] {cfg.answer_model if cfg.has_llm else 'not configured (cache-only)'}")


@app.command()
def seed(
    corpus: str = typer.Option(None, help="corpus to write to"),
    force: bool = typer.Option(False, "--force", help="re-seed even if facts already exist"),
) -> None:
    """Ingest the shipped demo corpus."""
    from custodia.demo import seed_demo

    client = _client()
    started = time.perf_counter()
    report = seed_demo(client, force=force, corpus=corpus)
    if not report.get("seeded"):
        console.print(f"[yellow]already seeded[/] ({report['facts']} facts). Use --force to rebuild.")
        return
    console.print(
        f"[green]seeded[/] {report['corpus']}: "
        f"{report['sessions']} sessions · {report['turns']} turns · {report['facts']} facts · "
        f"{report['entities']} entities · {report['edges']} edges "
        f"({report.get('quarantined', 0)} quarantined) in {(time.perf_counter()-started):.1f}s"
    )


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, readable=True, help="corpus JSON file"),
    corpus: str = typer.Option(None, help="corpus name; defaults to the one in the file"),
) -> None:
    """Ingest a corpus file in the same shape as demo/corpus.json."""
    from custodia.demo import to_turns, _iso
    from custodia.ingest import Ingestor
    from custodia.policy import Policy

    data = json.loads(path.read_text(encoding="utf-8"))
    name = corpus or data.get("corpus") or path.stem
    client = _client()
    ingestor = Ingestor(client, name, policy=Policy())
    for sidx, session in enumerate(data["sessions"]):
        ingestor.stage_session(
            session["sid"],
            ts=_iso(session["date"]),
            idx=sidx,
            turns=to_turns(session, name, sidx),
        )
    report = ingestor.flush()
    console.print_json(data=report.as_dict())


@app.command()
def ask(
    question: str = typer.Argument(..., help="what to ask memory"),
    corpus: str = typer.Option(None, help="which memory to read"),
    as_of: str = typer.Option(None, "--as-of", help="ISO date; answer as memory stood then"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    """Ask a question. Memory answers with citations, or declines."""
    client = _client()
    gate = _gate(client, _corpus(corpus))
    verdict = gate.ask(question, as_of=_moment(as_of))
    if json_out:
        console.print_json(data=verdict.as_dict())
    else:
        _print_verdict(verdict)
    raise typer.Exit(code=0 if verdict.answered else 1)


@app.command()
def evidence(
    question: str = typer.Argument(..., help="what to look for"),
    corpus: str = typer.Option(None),
    as_of: str = typer.Option(None, "--as-of", help="ISO date; retrieve as memory stood then"),
    paths: bool = typer.Option(False, "--paths", help="show the chain the engine walked to each fact"),
    limit: int = typer.Option(10, help="how many facts to show"),
) -> None:
    """Show the supporting facts for a question without answering it.

    The retrieval half on its own: what memory found, how strongly, and by which
    route. Useful when you want to reason over the evidence yourself, and when
    you want to see why an answer was or was not possible.
    """
    from custodia.lexical import LexicalIndex
    from custodia.llm import LLM
    from custodia.retrieve import Retriever

    name = _corpus(corpus)
    client = _client()
    retriever = Retriever(client, name, index=LexicalIndex.build(client, name), llm=LLM())
    warrant = retriever.warrant(question, as_of=_moment(as_of), k=limit)

    console.print(f"[dim]seeds[/] entities: {', '.join(warrant.seeds['entities']) or 'none'}")
    console.print(f"[dim]seeds[/] terms: {', '.join(warrant.seeds['terms'][:10]) or 'none'}")
    console.print(
        f"[dim]{warrant.paths_examined} paths examined · {warrant.facts_considered} facts considered "
        f"· {len(warrant.evidence)} met the evidence floor · {warrant.elapsed_ms:.0f} ms[/]\n"
    )
    for ev in warrant.evidence[:limit]:
        console.print(f"[cyan]{ev.score:.2f}[/]  {ev.text}")
        console.print(f"      [dim]{ev.tier} · {ev.sid} · {_stamp(ev.turn_ts or ev.valid_from)}[/]")
        if paths and ev.path:
            console.print(f"      [yellow]{'  ->  '.join(ev.path)}[/]")
    if warrant.quarantined_seen:
        console.print(
            f"\n[red]{warrant.quarantined_seen} quarantined fact(s) retrieved and refused[/]"
        )


@app.command()
def why(
    fact_id: int = typer.Argument(..., help="fact id from a citation"),
    corpus: str = typer.Option(None),
) -> None:
    """Show where a fact came from and what it replaced."""
    from custodia.audit import Auditor

    chain = Auditor(_client(), _corpus(corpus)).explain(fact_id)
    if not chain.get("found"):
        console.print("[red]no such fact in this corpus[/]")
        raise typer.Exit(code=1)
    console.print_json(data=chain)


@app.command()
def stats(corpus: str = typer.Option(None)) -> None:
    """Counts for a corpus."""
    name = _corpus(corpus)
    client = _client()
    table = Table(box=None, show_header=False)
    for label in (schema.SESSION, schema.TURN, schema.FACT, schema.ENTITY, schema.REJECTION, schema.ANSWER):
        table.add_row(label.lower(), str(client.count(label, corpus=name)))
    for status in (schema.QUARANTINED, schema.SUPERSEDED):
        rows = client.run(
            "MATCH (f:Fact) WHERE f.corpus = $c AND f.status = $s RETURN count(*) AS n",
            c=name,
            s=status,
        )
        table.add_row(f"facts ({status})", str(rows[0]["n"] if rows else 0))
    console.print(Panel(table, title=name))


@app.command()
def audit(
    corpus: str = typer.Option(None),
    limit: int = typer.Option(20, help="how many refused writes to show"),
) -> None:
    """Refused writes and the integrity of the provenance chain."""
    from custodia.audit import Auditor

    name = _corpus(corpus)
    auditor = Auditor(_client(), name)

    refused = auditor.rejections(limit)
    table = Table(show_header=True, header_style="dim", box=None)
    table.add_column("rule", style="red")
    table.add_column("reason")
    table.add_column("content", overflow="fold")
    for row in refused:
        table.add_row(row.get("rule", ""), row.get("reason", ""), (row.get("text") or "")[:140])
    console.print(Panel(table if refused else "[dim]nothing refused[/]", title=f"refused writes ({len(refused)})"))

    report = auditor.integrity()
    ok = report.get("ok")
    console.print(
        Panel(
            "\n".join(f"{k}: {v}" for k, v in report.items() if k != "ok"),
            title="integrity " + ("[green]ok[/]" if ok else "[red]failed[/]"),
            border_style="green" if ok else "red",
        )
    )
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def policy() -> None:
    """Print the trust rules that run on every write."""
    from custodia.policy import Policy

    table = Table(show_header=True, header_style="dim", box=None)
    table.add_column("rule", style="cyan")
    table.add_column("what it refuses", overflow="fold")
    for rule in Policy().describe():
        table.add_row(rule["rule"], rule["description"])
    console.print(table)


@app.command()
def attack(
    text: str = typer.Option(..., "--text", help="content an untrusted source is trying to store"),
    question: str = typer.Option(..., "--question", help="the question the attacker wants to move"),
    tier: str = typer.Option("tool", help="channel the content arrives on: external, tool, assistant, owner"),
    corpus: str = typer.Option(None),
) -> None:
    """Try to poison memory, live, and show what happened to the answer."""
    from custodia.ingest import Ingestor
    from custodia.policy import Policy
    from custodia.schema import Tier, Turn

    name = _corpus(corpus)
    client = _client()
    gate = _gate(client, name)

    before = gate.ask(question, record=False)
    console.print(Panel(before.answer or "(declined)", title="before", border_style="dim"))

    parsed = Tier.parse(tier)
    now = int(time.time())
    sid = f"attack-{now}"
    turn = Turn(
        corpus=name,
        sid=sid,
        idx=0,
        sidx=9_999,
        role="tool" if parsed <= Tier.TOOL else "user",
        text=text,
        ts=now,
        tier=parsed,
        # only content from outside the conversation carries an origin; setting
        # one on an owner write would clamp it down to the external tier
        origin="cli-attack" if parsed <= Tier.TOOL else "",
    )
    from custodia.demo import extractor, load_corpus

    principal = load_corpus().get("principal", "user") if name == "demo" else "user"
    ingestor = Ingestor(client, name, policy=Policy(), extract=extractor(principal))
    ingestor.stage_session(sid, ts=now, idx=9_999, turns=[turn])
    report = ingestor.flush()
    console.print(
        f"[dim]injected at tier [bold]{parsed.label}[/bold]: "
        f"{report.facts} fact(s) written, {report.quarantined} quarantined, {report.rejections} refused[/]"
    )

    after = _gate(client, name).ask(question, record=False)
    _print_verdict(after)

    # The citation set is the signal, not the wording: a model that rewrites
    # "Yes, X" as "X" changed the string and nothing else, while one that starts
    # citing the injected fact changed the answer however similar it reads.
    changed = set(before.citations) != set(after.citations)
    reworded = (before.answer or "").strip() != (after.answer or "").strip()
    if reworded and not changed:
        console.print("[dim]wording differs, citations identical - same answer, same evidence[/]")

    # What counts as success depends on the channel. A tool or external write
    # moving the answer is a breach. The principal's own write moving it is the
    # negative control passing - a guard that refuses the owner too is not
    # discriminating, it is just broken.
    control = parsed >= Tier.ASSISTANT
    good = changed if control else not changed
    if control:
        verdict_text = (
            "[green]accepted - the principal changed their own record[/]"
            if changed
            else "[red]wrongly refused - the guard is over-blocking, not discriminating[/]"
        )
    else:
        verdict_text = (
            "[red]the answer moved - the injection got through[/]"
            if changed
            else "[green]held - the answer did not move[/]"
        )
    console.print(Panel(verdict_text, border_style="green" if good else "red"))
    raise typer.Exit(code=0 if good else 1)


@app.command()
def demo(
    check: bool = typer.Option(False, "--check", help="assert each step's stated expectation"),
    corpus: str = typer.Option(None),
) -> None:
    """Run the scripted walkthrough over the demo corpus."""
    from custodia.demo import check_walkthrough

    client = _client()
    name = _corpus(corpus)
    gate = _gate(client, name)
    results = check_walkthrough(gate)

    passed = sum(1 for r in results if r["passed"])
    for r in results:
        mark = "[green]pass[/]" if r["passed"] else "[red]fail[/]"
        console.print(f"\n{mark} [bold]{r['id']}[/]  {r['question']}")
        if r["as_of"]:
            console.print(f"      [dim]as of {r['as_of']}[/]")
        console.print(f"      {r['answer'] or '(declined: ' + r['abstained_because'] + ')'}")
        console.print(f"      [dim]{r['why']}[/]")
        for f in r["failures"]:
            console.print(f"      [red]- {f}[/]")

    console.print(f"\n[bold]{passed}/{len(results)} steps passed[/]")
    if check and passed != len(results):
        raise typer.Exit(code=1)


@app.command()
def verify(corpus: str = typer.Option(None), seed_first: bool = typer.Option(True, "--seed/--no-seed")) -> None:
    """End-to-end check a reviewer can run: seed, answer, refuse, audit.

    Every line is produced by a live query. Exits non-zero if anything fails.
    """
    from custodia.audit import Auditor
    from custodia.demo import check_walkthrough, seed_demo

    cfg = settings()
    client = _client()
    name = _corpus(corpus)
    console.print(Panel(f"graph [bold]{cfg.hydra_uri}[/]\ncorpus [bold]{name}[/]", title="custodia verify"))

    failures: list[str] = []

    if seed_first:
        report = seed_demo(client, corpus=name)
        console.print(
            f"[dim]memory:[/] {client.count(schema.SESSION, corpus=name)} sessions · "
            f"{client.count(schema.TURN, corpus=name)} turns · "
            f"{client.count(schema.FACT, corpus=name)} facts · "
            f"{client.count(schema.ENTITY, corpus=name)} entities"
            + ("" if report.get("seeded") else "  [dim](already present)[/]")
        )

    results = check_walkthrough(_gate(client, name))
    table = Table(show_header=True, header_style="dim", box=None)
    table.add_column("", width=4)
    table.add_column("step", style="cyan")
    table.add_column("outcome", overflow="fold")
    for r in results:
        table.add_row(
            "[green]ok[/]" if r["passed"] else "[red]fail[/]",
            r["id"],
            (r["answer"] or f"declined: {r['abstained_because']}")[:110],
        )
        if not r["passed"]:
            failures.extend(f"{r['id']}: {f}" for f in r["failures"])
    console.print(table)

    integrity = Auditor(client, name).integrity()
    console.print(
        Panel(
            "\n".join(f"{k}: {v}" for k, v in integrity.items()),
            title="graph integrity",
            border_style="green" if integrity.get("ok") else "red",
        )
    )
    if not integrity.get("ok"):
        failures.append("integrity check failed")

    if failures:
        console.print(f"\n[red]{len(failures)} failure(s)[/]")
        for f in failures:
            console.print(f"  [red]- {f}[/]")
        raise typer.Exit(code=1)
    console.print(f"\n[green]all {len(results)} steps passed, provenance intact[/]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8080),
) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("custodia.api:app", host=host, port=port, reload=False)


@app.command()
def mcp() -> None:
    """Run the MCP server on stdio, so any agent can use this memory."""
    from custodia.mcp_server import mcp as server

    server.run()


def main() -> None:
    try:
        app()
    except HydraError as exc:
        console.print(f"[red]graph refused a query:[/] {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
