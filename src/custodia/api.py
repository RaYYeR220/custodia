"""HTTP surface.

The API is the seam every other surface goes through: the web client, the demo
walkthrough, and anything a third party wants to wire in. It deliberately
exposes the parts of the system that are usually hidden -- the warrant behind an
answer, the writes that were refused, the integrity of the provenance chain --
because being able to inspect those is the product.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from custodia import __version__, schema
from custodia.config import settings
from custodia.hydra import HydraClient, HydraError
from custodia.ids import corpus_id

app = FastAPI(
    title="Custodia",
    version=__version__,
    summary="Agent memory with a chain of custody",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# shared handles
# --------------------------------------------------------------------------- #

_client: HydraClient | None = None
_gates: dict[str, Any] = {}


def graph() -> HydraClient:
    global _client
    if _client is None:
        cfg = settings()
        _client = HydraClient(cfg.hydra_uri, cfg.hydra_token)
    return _client


def gate_for(corpus: str) -> Any:
    """One Gate per corpus, kept warm so the lexical index is not rebuilt per call."""
    if corpus not in _gates:
        from custodia.audit import Auditor
        from custodia.gate import Gate
        from custodia.lexical import LexicalIndex
        from custodia.llm import LLM
        from custodia.retrieve import Retriever

        client = graph()
        llm = LLM()
        index = LexicalIndex.build(client, corpus)
        retriever = Retriever(client, corpus, index=index, llm=llm)
        _gates[corpus] = Gate(retriever, llm=llm, auditor=Auditor(client, corpus))
    return _gates[corpus]


def invalidate(corpus: str) -> None:
    _gates.pop(corpus, None)


def _corpus(value: str | None) -> str:
    return value or settings().corpus


def _principal(corpus: str) -> str:
    """Whose memory this is, so extracted first-person claims share one subject."""
    rows = graph().run("MATCH (c:Corpus {id: $cid}) RETURN c.principal AS principal", cid=corpus_id(corpus))
    return (rows[0].get("principal") if rows else "") or "user"


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    corpus: str | None = None
    as_of: int | None = Field(default=None, description="unix seconds; answer as memory stood then")
    record: bool = True


class IngestTurn(BaseModel):
    role: str
    text: str
    origin: str = ""


class IngestSession(BaseModel):
    sid: str
    ts: int
    title: str = ""
    turns: list[IngestTurn]


class IngestRequest(BaseModel):
    corpus: str | None = None
    sessions: list[IngestSession]


class AttackRequest(BaseModel):
    corpus: str | None = None
    text: str = Field(min_length=1, description="the content an untrusted source is trying to store")
    tier: str = Field(default="external", description="external | tool | assistant | owner")
    question: str = Field(min_length=1, description="the question the attacker wants to move")
    origin: str = "attack-console"


# --------------------------------------------------------------------------- #
# health and status
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict[str, Any]:
    cfg = settings()
    try:
        reachable = bool(graph().run("MATCH (c:Corpus) RETURN count(*) AS n"))
    except (HydraError, OSError):
        reachable = False
    return {
        "status": "ok" if reachable else "degraded",
        "version": __version__,
        "graph": {"uri": cfg.hydra_uri, "reachable": reachable},
        "model": {
            "configured": cfg.has_llm,
            "answer": cfg.answer_model if cfg.has_llm else None,
            "mode": "live" if cfg.has_llm else "cache-only",
        },
    }


@app.get("/corpora")
def corpora() -> dict[str, Any]:
    rows = graph().run("MATCH (c:Corpus) RETURN c.name AS name, c.principal AS principal")
    out = []
    for row in rows:
        name = row.get("name")
        if not name:
            continue
        out.append(
            {
                "corpus": name,
                "principal": row.get("principal") or "",
                "sessions": graph().count(schema.SESSION, corpus=name),
                "turns": graph().count(schema.TURN, corpus=name),
                "facts": graph().count(schema.FACT, corpus=name),
                "entities": graph().count(schema.ENTITY, corpus=name),
            }
        )
    return {"corpora": out}


@app.get("/stats")
def stats(corpus: str | None = None) -> dict[str, Any]:
    name = _corpus(corpus)
    client = graph()
    quarantined = client.run(
        "MATCH (f:Fact) WHERE f.corpus = $corpus AND f.status = $status RETURN count(*) AS n",
        corpus=name,
        status=schema.QUARANTINED,
    )
    superseded = client.run(
        "MATCH (f:Fact) WHERE f.corpus = $corpus AND f.status = $status RETURN count(*) AS n",
        corpus=name,
        status=schema.SUPERSEDED,
    )
    return {
        "corpus": name,
        "sessions": client.count(schema.SESSION, corpus=name),
        "turns": client.count(schema.TURN, corpus=name),
        "facts": client.count(schema.FACT, corpus=name),
        "entities": client.count(schema.ENTITY, corpus=name),
        "rejections": client.count(schema.REJECTION, corpus=name),
        "quarantined": int(quarantined[0]["n"]) if quarantined else 0,
        "superseded": int(superseded[0]["n"]) if superseded else 0,
        "answers": client.count(schema.ANSWER, corpus=name),
    }


# --------------------------------------------------------------------------- #
# the core loop
# --------------------------------------------------------------------------- #


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    name = _corpus(req.corpus)
    started = time.perf_counter()
    verdict = gate_for(name).ask(req.question, as_of=req.as_of, record=req.record)
    payload = verdict.as_dict()
    payload["corpus"] = name
    payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return payload


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, Any]:
    from custodia.ingest import ingest_sessions

    name = _corpus(req.corpus)
    sessions = [
        {
            "sid": s.sid,
            "ts": s.ts,
            "title": s.title,
            "turns": [{"role": t.role, "text": t.text, "origin": t.origin} for t in s.turns],
        }
        for s in req.sessions
    ]
    report = ingest_sessions(graph(), name, sessions)
    invalidate(name)
    return report.as_dict()


@app.get("/fact/{fact_id}")
def fact(fact_id: int, corpus: str | None = None) -> dict[str, Any]:
    from custodia.audit import Auditor

    chain = Auditor(graph(), _corpus(corpus)).explain(fact_id)
    if not chain.get("found"):
        raise HTTPException(status_code=404, detail="no such fact in this corpus")
    return chain


# --------------------------------------------------------------------------- #
# the parts that are usually hidden
# --------------------------------------------------------------------------- #


@app.get("/rejections")
def rejections(corpus: str | None = None, limit: int = Query(default=100, le=1000)) -> dict[str, Any]:
    from custodia.audit import Auditor

    name = _corpus(corpus)
    return {"corpus": name, "rejections": Auditor(graph(), name).rejections(limit)}


@app.get("/history")
def history(corpus: str | None = None, limit: int = Query(default=50, le=500)) -> dict[str, Any]:
    from custodia.audit import Auditor

    name = _corpus(corpus)
    return {"corpus": name, "history": Auditor(graph(), name).history(limit)}


@app.get("/integrity")
def integrity(corpus: str | None = None) -> dict[str, Any]:
    from custodia.audit import Auditor

    name = _corpus(corpus)
    report = Auditor(graph(), name).integrity()
    report["corpus"] = name
    return report


@app.get("/policy")
def policy_rules() -> dict[str, Any]:
    from custodia.policy import Policy

    return {"rules": Policy().describe()}


# --------------------------------------------------------------------------- #
# graph views for the client
# --------------------------------------------------------------------------- #


def _node(vid_: int, label: str, props: dict[str, Any]) -> dict[str, Any]:
    return {"id": str(vid_), "label": label, "props": props}


@app.get("/graph/timeline")
def timeline(corpus: str | None = None) -> dict[str, Any]:
    """Sessions in order, with the facts each one produced - drives the scrubber."""
    name = _corpus(corpus)
    rows = graph().run(
        "MATCH (f:Fact)-[:DERIVED_FROM]->(t:Turn)-[:IN_SESSION]->(s:Session) "
        "WHERE f.corpus = $corpus "
        "RETURN s.sid AS sid, s.ts AS ts, s.title AS title, count(*) AS facts "
        "ORDER BY ts",
        corpus=name,
    )
    return {"corpus": name, "sessions": rows}


@app.get("/graph/neighbourhood")
def neighbourhood(
    corpus: str | None = None,
    entity: str | None = None,
    fact_id: int | None = None,
    max_len: int = Query(default=2, ge=1, le=4),
    count: int = Query(default=120, le=800),
) -> dict[str, Any]:
    """A subgraph around an entity or a fact, as nodes and edges the client can draw."""
    name = _corpus(corpus)
    client = graph()
    if entity:
        rows = client.paths(
            rel_types=schema.RETRIEVAL_RELS,
            source_label=schema.ENTITY,
            source_property="norm",
            source_values=[entity],
            max_len=max_len,
            direction="both",
            count=count,
        )
    elif fact_id is not None:
        rows = client.paths(
            rel_types=schema.RETRIEVAL_RELS,
            source_node=fact_id,
            max_len=max_len,
            direction="both",
            count=count,
        )
    else:
        raise HTTPException(status_code=400, detail="pass either entity or fact_id")
    return _paths_to_graph(rows, name)


def _paths_to_graph(rows: Iterable[dict[str, Any]], corpus: str) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = row.get("path")
        if path is None:
            continue
        for node in path.nodes:
            props = dict(node)
            if props.get("corpus") not in (None, corpus):
                continue
            label = next(iter(node.labels), "Node")
            nodes[node.element_id] = _node(node.element_id, label, props)
        for rel in path.relationships:
            key = f"{rel.start_node.element_id}-{rel.type}-{rel.end_node.element_id}"
            edges[key] = {
                "id": key,
                "type": rel.type,
                "source": str(rel.start_node.element_id),
                "target": str(rel.end_node.element_id),
            }
    return {"corpus": corpus, "nodes": list(nodes.values()), "edges": list(edges.values())}


# --------------------------------------------------------------------------- #
# attack console
# --------------------------------------------------------------------------- #


@app.post("/attack")
def attack(req: AttackRequest) -> dict[str, Any]:
    """Try to poison the memory, live, and report what the policy engine did.

    The answer to the target question is captured before and after the attempted
    write, so the console can show that the injection was stored, quarantined and
    ignored -- rather than simply asserting that it would be.
    """
    from custodia.ingest import Ingestor
    from custodia.policy import Policy
    from custodia.schema import Tier, Turn

    name = _corpus(req.corpus)
    before = gate_for(name).ask(req.question, record=False)

    tier = Tier.parse(req.tier)
    now = int(time.time())
    sid = f"attack-{now}"
    turn = Turn(
        corpus=name,
        sid=sid,
        idx=0,
        sidx=9_999,
        role="tool" if tier <= Tier.TOOL else "user",
        text=req.text,
        ts=now,
        tier=tier,
        origin=req.origin,
    )
    from custodia.demo import extractor

    ingestor = Ingestor(graph(), name, policy=Policy(), extract=extractor(_principal(name)))
    ingestor.stage_session(sid, ts=now, idx=9_999, turns=[turn])
    report = ingestor.flush()
    invalidate(name)

    after = gate_for(name).ask(req.question, record=False)
    return {
        "corpus": name,
        "injected": {"text": req.text, "tier": tier.label, "origin": req.origin, "session": sid},
        "ingest": report.as_dict(),
        "before": before.as_dict(),
        "after": after.as_dict(),
        "answer_changed": before.answer.strip() != after.answer.strip(),
        "quarantined": report.quarantined,
        "rejections": report.rejections,
    }


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #


@app.get("/demo/walkthrough")
def demo_walkthrough() -> dict[str, Any]:
    from custodia.demo import load_walkthrough

    return load_walkthrough()


@app.post("/demo/seed")
def demo_seed(force: bool = False) -> dict[str, Any]:
    from custodia.demo import seed_demo

    report = seed_demo(graph(), force=force)
    invalidate(report["corpus"])
    return report


@app.on_event("shutdown")
def _shutdown() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def main() -> None:
    import uvicorn

    uvicorn.run(
        "custodia.api:app",
        host=os.environ.get("CUSTODIA_HOST", "127.0.0.1"),
        port=int(os.environ.get("CUSTODIA_PORT", "8080")),
        reload=False,
    )


if __name__ == "__main__":
    main()
