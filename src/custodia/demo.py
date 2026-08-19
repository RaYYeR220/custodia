"""The shipped demo corpus and its scripted walkthrough.

`demo/corpus.json` is eight months of assistant sessions with one person, written
by hand so that every behaviour the product claims has a case in it: a fact that
is revised months later, a question nothing in memory supports, an untrusted
document that tries to overwrite a health record, and a legitimate update to that
same record from the person themselves.

`demo/walkthrough.json` states what each question is expected to do, which is
what makes `custodia demo --check` a check rather than a slideshow.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from custodia import schema
from custodia.config import REPO_ROOT
from custodia.hydra import HydraClient
from custodia.schema import Tier, Turn

DEMO_DIR = REPO_ROOT / "demo"
if not DEMO_DIR.exists():  # running straight from a source tree
    DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"

#: turns whose content arrived from outside the conversation are not conversation
_EXTERNAL_ORIGINS = ("http://", "https://", "shared-document://", "file://")


def _iso(value: str) -> int:
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def load_corpus(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or DEMO_DIR / "corpus.json").read_text(encoding="utf-8"))


def load_walkthrough(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or DEMO_DIR / "walkthrough.json").read_text(encoding="utf-8"))


def to_turns(session: dict[str, Any], corpus: str, sidx: int) -> list[Turn]:
    """One session's raw turns, tiered by who actually produced them."""
    base = _iso(session["date"])
    turns: list[Turn] = []
    for idx, raw in enumerate(session["turns"]):
        origin = raw.get("origin", "")
        external = bool(origin) and origin.startswith(_EXTERNAL_ORIGINS)
        tier = schema.tier_for_role(raw["role"], external=external)
        turns.append(
            Turn(
                corpus=corpus,
                sid=session["sid"],
                idx=idx,
                sidx=sidx,
                role=raw["role"],
                text=raw["text"],
                ts=base + idx * 60,
                tier=tier,
                origin=origin,
            )
        )
    return turns


def extractor(principal: str) -> Any:
    """The extractor the demo and the attack console both use.

    Bound to the principal so first-person claims land on one subject key.

    The model is always handed over, even when no credentials are configured:
    ``enabled`` only gates *live* calls, and a shipped cache entry is still a
    hit. That is what lets the walkthrough reproduce model-grade extraction with
    no key at all. Windows the cache cannot serve fall back to the rule-based
    extractor, which is weaker but keeps the path runnable.
    """
    from custodia.extract import extract_session
    from custodia.llm import LLM

    llm = LLM()
    return lambda turns: extract_session(turns, llm=llm, principal=principal)


def seed_demo(client: HydraClient, *, force: bool = False, corpus: str | None = None) -> dict[str, Any]:
    """Ingest the demo corpus. Idempotent - re-seeding rewrites the same vertices."""
    from custodia.ingest import Ingestor
    from custodia.policy import Policy

    data = load_corpus()
    name = corpus or data["corpus"]

    existing = client.count(schema.FACT, corpus=name)
    if existing and not force:
        return {"corpus": name, "seeded": False, "facts": existing, "reason": "already seeded"}

    ingestor = Ingestor(client, name, policy=Policy(), extract=extractor(data.get("principal", "user")))
    for sidx, session in enumerate(data["sessions"]):
        turns = to_turns(session, name, sidx)
        ingestor.stage_session(session["sid"], ts=_iso(session["date"]), idx=sidx, turns=turns)
    report = ingestor.flush()

    principal = data.get("principal", "")
    if principal:
        # whose memory this is, so anything that re-extracts later (the attack
        # console) resolves first-person claims onto the same subject key
        from custodia.ids import corpus_id

        client.run(
            "MATCH (c:Corpus {id: $cid}) SET c.principal = $principal",
            cid=corpus_id(name),
            principal=principal,
        )

    result = report.as_dict()
    result["corpus"] = name
    result["seeded"] = True
    result["principal"] = data.get("principal", "")
    return result


def check_walkthrough(gate: Any, *, walkthrough: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run every scripted step and report pass/fail against its stated expectation.

    The expectations live in the data file rather than in this function so that a
    reviewer reads the same claims the check enforces.
    """
    script = walkthrough or load_walkthrough()
    results: list[dict[str, Any]] = []

    for step in script["steps"]:
        expect = step.get("expect", {})
        as_of = _iso(step["as_of"]) if step.get("as_of") else None
        verdict = gate.ask(step["question"], as_of=as_of, record=False)
        answer = (verdict.answer or "").lower()
        failures: list[str] = []

        if expect.get("answered") is True and not verdict.answered:
            failures.append(f"expected an answer, abstained: {verdict.abstained_because}")
        if expect.get("answered") is False and verdict.answered:
            failures.append(f"expected abstention, answered: {verdict.answer[:120]}")

        for needle in expect.get("answer_contains", []):
            if verdict.answered and needle.lower() not in answer:
                failures.append(f"answer missing {needle!r}")
        for needle in expect.get("answer_excludes", []):
            if verdict.answered and needle.lower() in answer:
                failures.append(f"answer should not mention {needle!r}")

        wanted_sessions = set(expect.get("must_cite_sessions", []))
        if wanted_sessions:
            cited = {
                ev.sid
                for ev in verdict.warrant.evidence
                if ev.fid in set(verdict.citations)
            }
            missing = wanted_sessions - cited
            if missing:
                failures.append(f"did not cite {sorted(missing)}; cited {sorted(cited)}")

        floor = expect.get("quarantined_seen_min")
        if floor is not None and verdict.warrant.quarantined_seen < floor:
            failures.append(
                f"expected at least {floor} quarantined fact(s) in retrieval, "
                f"saw {verdict.warrant.quarantined_seen}"
            )

        results.append(
            {
                "id": step["id"],
                "question": step["question"],
                "as_of": step.get("as_of"),
                "why": step.get("why", ""),
                "passed": not failures,
                "failures": failures,
                "answered": verdict.answered,
                "answer": verdict.answer,
                "citations": verdict.citations,
                "abstained_because": verdict.abstained_because,
                "quarantined_seen": verdict.warrant.quarantined_seen,
            }
        )
    return results
