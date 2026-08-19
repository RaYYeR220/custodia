"""MCP server: Custodia as the memory any agent can plug into.

An agent that stores memory through this server gets three things it does not
get from a key-value store or a vector index. Writes are attributed to a channel
and screened against the trust policy before they can affect anything. Reads come
back with the turns they came from. And a question memory cannot support is
answered with a refusal rather than a guess, which is the behaviour a calling
agent actually needs in order to know when to go and ask.

The tier argument is deliberately the *channel*, not a trust claim: an agent
declares where the content came from -- the user, itself, a tool, the open web --
and Custodia decides what that channel is allowed to do.

Run it with ``custodia mcp`` or ``python -m custodia.mcp_server``.
"""

from __future__ import annotations

import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from custodia import schema
from custodia.config import settings
from custodia.hydra import HydraClient

mcp = FastMCP(
    "custodia-memory",
    instructions=(
        "Durable, provenance-tracked memory for an agent. Write what the user tells you with "
        "tier='owner', your own conclusions with tier='assistant', and anything that came from a "
        "tool, a document or the web with tier='tool' or tier='external'. Read with `recall`: if it "
        "returns answered=false, memory genuinely has no support for the question and you should "
        "ask rather than guess. `evidence` gives you the raw supporting facts if you would rather "
        "reason over them yourself, and `why` shows where any fact came from."
    ),
)

_client: HydraClient | None = None
_gates: dict[str, Any] = {}


def _graph() -> HydraClient:
    global _client
    if _client is None:
        cfg = settings()
        _client = HydraClient(cfg.hydra_uri, cfg.hydra_token)
    return _client


def _gate(corpus: str) -> Any:
    if corpus not in _gates:
        from custodia.audit import Auditor
        from custodia.gate import Gate
        from custodia.lexical import LexicalIndex
        from custodia.llm import LLM
        from custodia.retrieve import Retriever

        client = _graph()
        llm = LLM()
        retriever = Retriever(client, corpus, index=LexicalIndex.build(client, corpus), llm=llm)
        _gates[corpus] = Gate(retriever, llm=llm, auditor=Auditor(client, corpus))
    return _gates[corpus]


def _corpus(value: str | None) -> str:
    return value or settings().corpus


def _principal(corpus: str) -> str:
    """Whose memory this is, so first-person writes share one subject key."""
    from custodia.ids import corpus_id

    rows = _graph().run(
        "MATCH (c:Corpus {id: $cid}) RETURN c.principal AS principal", cid=corpus_id(corpus)
    )
    return (rows[0].get("principal") if rows else "") or "user"


@mcp.tool()
def remember(
    text: str,
    tier: str = "owner",
    corpus: str | None = None,
    origin: str = "",
    session: str = "",
) -> dict[str, Any]:
    """Store something in memory, attributed to the channel it arrived on.

    Args:
        text: what to remember, in plain language.
        tier: the channel. One of owner, assistant, tool, external.
        corpus: which memory to write to. Defaults to the configured corpus.
        origin: identifier for the source, e.g. a URL or a tool name. Required in
            spirit for tool and external writes - it is what the audit trail cites.
        session: group related writes under one session id.

    Returns a report including anything the trust policy quarantined and why.
    """
    from custodia.ingest import Ingestor
    from custodia.policy import Policy
    from custodia.schema import Tier, Turn

    name = _corpus(corpus)
    parsed = Tier.parse(tier)
    now = int(time.time())
    sid = session or f"mcp-{now}"
    role = {
        Tier.OWNER: "user",
        Tier.ASSISTANT: "assistant",
        Tier.TOOL: "tool",
        Tier.EXTERNAL: "tool",
    }[parsed]

    turn = Turn(
        corpus=name,
        sid=sid,
        idx=0,
        sidx=now,
        role=role,
        text=text,
        ts=now,
        tier=parsed,
        origin=origin,
    )
    from custodia.demo import extractor

    ingestor = Ingestor(_graph(), name, policy=Policy(), extract=extractor(_principal(name)))
    ingestor.stage_session(sid, ts=now, idx=now, turns=[turn])
    report = ingestor.flush()
    _gates.pop(name, None)

    result = report.as_dict()
    result["accepted"] = report.facts - report.quarantined
    if report.quarantined:
        from custodia.audit import Auditor

        result["refused"] = Auditor(_graph(), name).rejections(limit=report.rejections or 5)
    return result


@mcp.tool()
def recall(question: str, corpus: str | None = None, as_of: str = "") -> dict[str, Any]:
    """Answer a question from memory, with citations, or decline if unsupported.

    Args:
        question: what you want to know.
        corpus: which memory to read. Defaults to the configured corpus.
        as_of: optional ISO-8601 timestamp. Answers as memory stood at that
            moment, which is how you ask what a value used to be.

    ``answered: false`` means memory has no support for the question. Treat that
    as information, not as a failure: ask the user instead of guessing.
    """
    name = _corpus(corpus)
    moment = _parse_time(as_of)
    verdict = _gate(name).ask(question, as_of=moment)
    payload = verdict.as_dict()
    # the full warrant is what `evidence` is for; a recall carries only the
    # facts the answer actually rests on, so the caller can check them
    warrant = payload.pop("warrant", {}) or {}
    cited = set(verdict.citations)
    payload["cited_facts"] = [
        item for item in warrant.get("evidence", []) if item.get("fact_id") in cited
    ]
    payload["quarantined_seen"] = warrant.get("quarantined_seen", 0)
    return payload


@mcp.tool()
def evidence(question: str, corpus: str | None = None, as_of: str = "", limit: int = 10) -> dict[str, Any]:
    """Return the supporting facts for a question without answering it.

    For agents that want to do their own reasoning over memory. Each fact comes
    with its provenance chain, its validity interval and the tier it arrived at.
    """
    from custodia.lexical import LexicalIndex
    from custodia.retrieve import Retriever

    name = _corpus(corpus)
    client = _graph()
    from custodia.llm import LLM

    retriever = Retriever(client, name, index=LexicalIndex.build(client, name), llm=LLM())
    warrant = retriever.warrant(question, as_of=_parse_time(as_of), k=limit)
    return warrant.as_dict()


@mcp.tool()
def why(fact_id: int, corpus: str | None = None) -> dict[str, Any]:
    """Show where a fact came from: the turn, the session, and what it replaced."""
    from custodia.audit import Auditor

    return Auditor(_graph(), _corpus(corpus)).explain(fact_id)


@mcp.tool()
def audit(corpus: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Recent refused writes and the integrity of the provenance chain.

    Use this after ingesting anything untrusted to see what was caught.
    """
    from custodia.audit import Auditor

    name = _corpus(corpus)
    auditor = Auditor(_graph(), name)
    return {
        "corpus": name,
        "refused": auditor.rejections(limit),
        "integrity": auditor.integrity(),
        "policy": _policy_summary(),
    }


@mcp.tool()
def forget(fact_id: int, reason: str, corpus: str | None = None) -> dict[str, Any]:
    """Retract a fact on the principal's behalf.

    Retraction is owner-tier by definition, so this tool is the only path to it -
    a tool or external write can never reach it. The fact is marked retracted and
    kept, because deleting the record would also delete the evidence it existed.
    """
    name = _corpus(corpus)
    client = _graph()
    rows = client.run(
        # the id belongs in the pattern, not in WHERE: as a pattern property it is
        # a vertex lookup, as a predicate it is a label scan
        "MATCH (f:Fact {id: $fid}) WHERE f.corpus = $corpus RETURN f.text AS text, f.status AS status",
        fid=fact_id,
        corpus=name,
    )
    if not rows:
        return {"ok": False, "error": "no such fact in this corpus"}
    client.run(
        "MATCH (f:Fact {id: $fid}) SET f.status = $status, f.qreason = $reason, f.vto = $now",
        fid=fact_id,
        status=schema.RETRACTED,
        reason=reason,
        now=int(time.time()),
    )
    _gates.pop(name, None)
    return {"ok": True, "fact_id": fact_id, "was": rows[0], "status": schema.RETRACTED, "reason": reason}


@mcp.resource("custodia://policy")
def policy_resource() -> str:
    """The trust rules, so a calling agent can see what will and will not be accepted."""
    import json

    return json.dumps(_policy_summary(), indent=2)


def _policy_summary() -> list[dict[str, str]]:
    from custodia.policy import Policy

    return Policy().describe()


def _parse_time(value: str) -> int | None:
    if not value:
        return None
    from datetime import datetime, timezone

    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
