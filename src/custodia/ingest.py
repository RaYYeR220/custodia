"""Staged, batched, idempotent writing of a whole corpus.

HydraDB commits every statement to object storage, so a writer that inserts as
it goes measures out at roughly a dozen rows a second. The batched forms run two
orders of magnitude faster, which is why nothing here touches the graph until
:meth:`Ingestor.flush`: sessions, turns, facts, entities and rejections are all
staged in memory, reconciled once against each other, and then written label by
label in a fixed order.

That staging is also where the provenance invariant is enforced. A fact is only
accepted if the turn it claims to come from has already been staged, and the
``DERIVED_FROM`` edges go out in the same flush as the fact vertices, so there
is no window in which a fact exists without the evidence that produced it.
``UNWIND ... MERGE (n {id: row.id})`` would happily create a bare vertex -- the
engine does not stop us -- so the guarantee has to be ours, and
``custodia audit --orphans`` exists to check we kept it.

Every id is derived by hashing a namespaced key, so a flush is an upsert: the
same corpus ingested twice leaves exactly the same graph. That is the
crash-recovery story too -- a run that dies halfway is resumed by running it
again. A flush that raises leaves the staging buffers untouched so the caller
can retry without rebuilding them.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from custodia import ids, schema
from custodia.hydra.client import HydraClient
from custodia.policy import Policy, RejectionRecord
from custodia.resolve import (
    Reconciliation,
    canonical_key,
    normalize_entity,
    reconcile,
    resolve_entities,
)
from custodia.schema import (
    ACTIVE,
    OPEN_INTERVAL,
    QUARANTINED,
    Fact,
    Tier,
    Turn,
    tier_for_role,
)

#: written as its own label so a refusal is queryable, not just logged
_REJECTION_TEXT_LIMIT = 2000


@dataclass(slots=True)
class IngestReport:
    """What one flush actually wrote. The numbers go into the proof documents."""

    corpus: str
    sessions: int = 0
    turns: int = 0
    facts: int = 0
    entities: int = 0
    edges: int = 0
    quarantined: int = 0
    superseded: int = 0
    contradicted: int = 0
    rejections: int = 0
    batches: int = 0
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def rows_per_second(self) -> float:
        rows = self.turns + self.facts + self.entities + self.edges
        return rows / (self.elapsed_ms / 1000) if self.elapsed_ms > 0 else 0.0


class Ingestor:
    """Stages a corpus in memory and writes it in one batched flush."""

    __slots__ = (
        "client",
        "corpus",
        "policy",
        "_sessions",
        "_turns",
        "_facts",
        "_provenance",
        "_entities",
        "_rejections",
        "_screened",
    )

    def __init__(self, client: HydraClient, corpus: str, *, policy: Policy | None = None) -> None:
        self.client = client
        self.corpus = corpus
        self.policy = policy or Policy()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._turns: dict[tuple[str, int], Turn] = {}
        self._facts: dict[str, Fact] = {}
        #: fact key -> the (session, turn index) pairs that produced it
        self._provenance: dict[str, list[tuple[str, int]]] = {}
        #: entity key -> the first spelling we saw, kept for display
        self._entities: dict[str, str] = {}
        self._rejections: list[RejectionRecord] = []
        #: memoised content screen of each staged turn
        self._screened: dict[tuple[str, int], tuple[str, str] | None] = {}

    # ----------------------------------------------------------------- staging

    def stage_session(self, sid: str, *, ts: int, idx: int, turns: list[Turn]) -> None:
        """Stage a conversation and its turns.

        Turn tiers are clamped, never raised: a turn's tier may be lower than
        its role warrants (an assistant message quoting a web page is
        ``external``), but it can never be higher, so a caller that forgets to
        set one cannot accidentally hand owner authority to a tool result.
        """
        self._sessions[sid] = {"sid": sid, "ts": int(ts), "idx": int(idx)}
        for turn in turns:
            turn.corpus = self.corpus
            turn.sid = sid
            turn.sidx = int(idx)
            warranted = tier_for_role(turn.role, external=bool(turn.origin))
            turn.tier = Tier(min(int(turn.tier), int(warranted)))
            self._turns[(sid, int(turn.idx))] = turn

    def stage_facts(self, facts: Sequence[Any], *, sid: str) -> None:
        """Stage extracted facts against turns that are already staged.

        Raises ``ValueError`` if a fact does not resolve to a staged turn. This
        is the provenance invariant: a fact that cannot name the turn it came
        from is not a fact we are willing to answer from, so it never reaches
        the graph at all. Nothing is committed to the buffers until every fact
        in the batch has validated, so a rejected batch leaves the Ingestor
        exactly as it was.
        """
        if sid not in self._sessions:
            raise ValueError(f"session {sid!r} has not been staged")

        prepared: list[tuple[Fact, Turn, RejectionRecord | None]] = []
        for item in facts:
            tidx = _attr(item, "turn_idx", "tidx")
            if tidx is None:
                raise ValueError(
                    f"fact {_attr(item, 'text', default='')!r} carries no turn index; "
                    "a fact without provenance is not admissible"
                )
            turn = self._turns.get((sid, int(tidx)))
            if turn is None:
                raise ValueError(
                    f"fact references turn {int(tidx)} of session {sid!r}, "
                    "which has not been staged"
                )
            prepared.append(self._build(item, turn))

        for fact, turn, rejection in prepared:
            self._absorb(fact, turn)
            if rejection is not None:
                self._rejections.append(rejection)

    # -------------------------------------------------------------------- flush

    def flush(self) -> IngestReport:
        """Reconcile the staged corpus and write it in one batched pass."""
        started = time.perf_counter()
        if not self._sessions:
            return IngestReport(corpus=self.corpus)

        facts = list(self._facts.values())
        recon = reconcile(facts, policy=self.policy)
        rejections = self._rejections + self._refusal_records(recon)

        missing = [f.key for f in facts if not self._provenance.get(f.key)]
        if missing:  # pragma: no cover - guarded at stage time
            raise ValueError(f"{len(missing)} staged facts have no provenance: {missing[:3]}")

        cid = ids.corpus_id(self.corpus)
        queries_before = int(self.client.stats["queries"])

        # ---- vertices, in dependency order -------------------------------
        self.client.upsert_nodes(
            schema.CORPUS, [{"id": cid, "corpus": self.corpus, "name": self.corpus}]
        )
        self.client.upsert_nodes(schema.SESSION, self._session_rows())
        self.client.upsert_nodes(schema.TURN, self._turn_rows())
        entity_rows = self._entity_rows()
        if entity_rows:
            self.client.upsert_nodes(schema.ENTITY, entity_rows)
        fact_rows = self._fact_rows(facts)
        if fact_rows:
            self.client.upsert_nodes(schema.FACT, fact_rows)
        rejection_rows, rejection_edges = self._rejection_rows(rejections)
        if rejection_rows:
            self.client.upsert_nodes(schema.REJECTION, rejection_rows)

        # ---- edges --------------------------------------------------------
        edges = 0
        edges += self._edges(schema.IN_CORPUS, schema.SESSION, schema.CORPUS, [
            (ids.session_id(self.corpus, sid), cid) for sid in sorted(self._sessions)
        ])
        edges += self._edges(schema.IN_SESSION, schema.TURN, schema.SESSION, [
            (ids.turn_id(self.corpus, sid, idx), ids.session_id(self.corpus, sid))
            for (sid, idx) in sorted(self._turns)
        ])
        edges += self._edges(schema.DERIVED_FROM, schema.FACT, schema.TURN, [
            (ids.fact_id(self.corpus, key), ids.turn_id(self.corpus, sid, idx))
            for key in sorted(self._provenance)
            for (sid, idx) in sorted(set(self._provenance[key]))
        ])
        edges += self._edges(schema.MENTIONS, schema.FACT, schema.ENTITY, [
            (ids.fact_id(self.corpus, fact.key), ids.entity_id(self.corpus, norm))
            for fact in sorted(facts, key=lambda f: f.key)
            for norm in fact.entities
        ])
        for rel, pairs in (
            (schema.SUPERSEDES, recon.supersedes),
            (schema.CONTRADICTS, recon.contradicts),
            (schema.CORROBORATES, recon.corroborates),
        ):
            edges += self._edges(rel, schema.FACT, schema.FACT, [
                (ids.fact_id(self.corpus, new.key), ids.fact_id(self.corpus, old.key))
                for new, old in pairs
            ])
        edges += self._edges(
            schema.BLOCKED, schema.REJECTION, schema.FACT, rejection_edges["blocked"]
        )
        edges += self._edges(
            schema.RAISED_BY, schema.REJECTION, schema.TURN, rejection_edges["raised_by"]
        )

        elapsed = (time.perf_counter() - started) * 1000
        report = IngestReport(
            corpus=self.corpus,
            sessions=len(self._sessions),
            turns=len(self._turns),
            facts=len(fact_rows),
            entities=len(entity_rows),
            edges=edges,
            quarantined=sum(1 for f in facts if f.status == QUARANTINED),
            superseded=len(recon.supersedes),
            contradicted=len(recon.contradicts),
            rejections=len(rejection_rows),
            batches=int(self.client.stats["queries"]) - queries_before,
            elapsed_ms=elapsed,
        )
        self._reset()
        return report

    # ------------------------------------------------------------------ internals

    def _build(self, item: Any, turn: Turn) -> tuple[Fact, Turn, RejectionRecord | None]:
        """Turn one extracted claim into a staged ``Fact`` and screen it."""
        subject = str(_attr(item, "subject", "subj", default="") or "")
        predicate = str(_attr(item, "predicate", "pred", default="") or "")
        obj = str(_attr(item, "object", "obj", default="") or "")
        key = str(_attr(item, "key", default="") or "") or canonical_key(subject, predicate, obj)
        text = str(_attr(item, "text", default="") or "").strip()
        if not text:
            text = " ".join(part for part in (subject, predicate, obj) if part).strip()

        mentions = [str(m) for m in (_attr(item, "entities", default=None) or ())]
        if not mentions:
            mentions = [p for p in (subject, obj) if p]
        entities = resolve_entities(mentions)
        for raw in mentions:
            norm = normalize_entity(raw)
            if norm:
                self._entities.setdefault(norm, raw.strip() or norm)

        valid_from = int(_attr(item, "valid_from", "vfrom", default=0) or 0) or int(turn.ts)
        fact = Fact(
            corpus=self.corpus,
            key=key,
            text=text,
            subject=subject,
            predicate=predicate,
            object=obj,
            entities=entities,
            # the tier is the turn's, never the content's
            tier=turn.tier,
            status=ACTIVE,
            valid_from=valid_from,
            valid_to=int(_attr(item, "valid_to", "vto", default=OPEN_INTERVAL) or OPEN_INTERVAL),
            ingested_at=int(time.time()),
            conf=float(_attr(item, "conf", "confidence", default=1.0) or 1.0),
            sid=turn.sid,
            sidx=turn.sidx,
            tidx=turn.idx,
        )

        decision = self.policy.admit(fact)
        if not decision.flagged:
            # the payload usually lives in the carrier, not in the tidy triple
            # an extractor lifted out of it, so the source turn is screened too
            carried = self._screen_turn(turn)
            if carried is not None:
                rule, reason = carried
                decision = self.policy.verdict(
                    rule, f"{reason} Carried by turn {turn.idx} of session {turn.sid}."
                )

        rejection: RejectionRecord | None = None
        if decision.flagged:
            if not decision.admitted:
                fact.status = decision.status
                fact.quarantine_reason = decision.reason
            rejection = self.policy.rejection(
                decision,
                fact,
                turn_id=ids.turn_id(self.corpus, turn.sid, turn.idx),
                target_fact_id=ids.fact_id(self.corpus, fact.key),
                ts=int(turn.ts),
            )
        return fact, turn, rejection

    def _screen_turn(self, turn: Turn) -> tuple[str, str] | None:
        cache_key = (turn.sid, int(turn.idx))
        if cache_key not in self._screened:
            self._screened[cache_key] = self.policy.screen(turn.text)
        return self._screened[cache_key]

    def _absorb(self, fact: Fact, turn: Turn) -> None:
        """Merge a fact into the staging set, keeping every turn that produced it."""
        held = self._facts.get(fact.key)
        if held is None:
            self._facts[fact.key] = fact
        else:
            if fact.valid_from and (not held.valid_from or fact.valid_from < held.valid_from):
                held.valid_from = fact.valid_from
            if int(fact.tier) > int(held.tier):
                held.tier = fact.tier
                if held.status == QUARANTINED and fact.status == ACTIVE:
                    held.status, held.quarantine_reason = ACTIVE, ""
            elif fact.status == QUARANTINED and held.status == ACTIVE and fact.tier == held.tier:
                held.status, held.quarantine_reason = QUARANTINED, fact.quarantine_reason
            for norm in fact.entities:
                if norm not in held.entities:
                    held.entities.append(norm)
        provenance = self._provenance.setdefault(fact.key, [])
        pair = (turn.sid, int(turn.idx))
        if pair not in provenance:
            provenance.append(pair)

    def _refusal_records(self, recon: Reconciliation) -> list[RejectionRecord]:
        """Raise a ``Rejection`` for every supersession the policy turned down."""
        records: list[RejectionRecord] = []
        for newer, _older, decision in recon.refusals:
            turn = self._turns.get((newer.sid, newer.tidx))
            records.append(
                self.policy.rejection(
                    decision,
                    newer,
                    turn_id=ids.turn_id(self.corpus, newer.sid, newer.tidx),
                    target_fact_id=ids.fact_id(self.corpus, newer.key),
                    ts=int(turn.ts) if turn is not None else newer.valid_from,
                )
            )
        return records

    # ------------------------------------------------------------------- rows

    def _session_rows(self) -> list[dict[str, Any]]:
        rows = []
        for sid in sorted(self._sessions):
            meta = self._sessions[sid]
            rows.append(
                {
                    "id": ids.session_id(self.corpus, sid),
                    "corpus": self.corpus,
                    "sid": sid,
                    "idx": meta["idx"],
                    "ts": meta["ts"],
                    "turns": sum(1 for (s, _) in self._turns if s == sid),
                }
            )
        return rows

    def _turn_rows(self) -> list[dict[str, Any]]:
        return [
            {"id": ids.turn_id(self.corpus, sid, idx), **self._turns[(sid, idx)].props}
            for (sid, idx) in sorted(self._turns)
        ]

    def _entity_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "id": ids.entity_id(self.corpus, norm),
                "corpus": self.corpus,
                "norm": norm,
                "name": self._entities[norm],
            }
            for norm in sorted(self._entities)
        ]

    def _fact_rows(self, facts: Sequence[Fact]) -> list[dict[str, Any]]:
        return [
            {"id": ids.fact_id(self.corpus, fact.key), **fact.props}
            for fact in sorted(facts, key=lambda f: f.key)
        ]

    def _rejection_rows(
        self, records: Sequence[RejectionRecord]
    ) -> tuple[list[dict[str, Any]], dict[str, list[tuple[int, int]]]]:
        rows: dict[int, dict[str, Any]] = {}
        blocked: list[tuple[int, int]] = []
        raised_by: list[tuple[int, int]] = []
        for record in records:
            rid = ids.rejection_id(self.corpus, record.rule, record.text, record.ts)
            rows[rid] = {
                "id": rid,
                "corpus": self.corpus,
                **record.props,
                "text": record.text[:_REJECTION_TEXT_LIMIT],
            }
            if record.target_fact_id is not None:
                blocked.append((rid, record.target_fact_id))
            if record.turn_id:
                raised_by.append((rid, record.turn_id))
        ordered = [rows[rid] for rid in sorted(rows)]
        return ordered, {"blocked": blocked, "raised_by": raised_by}

    def _edges(
        self,
        rel: str,
        src_label: str,
        dst_label: str,
        pairs: Iterable[tuple[int, int]],
    ) -> int:
        rows: dict[int, dict[str, Any]] = {}
        for src, dst in pairs:
            if src == dst:
                # a claim cannot be its own evidence; identical triples are one
                # vertex by construction, so this only guards against a self-loop
                continue
            rid = ids.edge_id(rel, src, dst)
            rows[rid] = {"s": src, "d": dst, "rid": rid}
        if not rows:
            return 0
        ordered = [rows[rid] for rid in sorted(rows)]
        self.client.merge_edges(rel, src_label, dst_label, ordered)
        return len(ordered)

    def _reset(self) -> None:
        self._sessions.clear()
        self._turns.clear()
        self._facts.clear()
        self._provenance.clear()
        self._entities.clear()
        self._rejections.clear()
        self._screened.clear()


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #


def ingest_sessions(
    client: HydraClient,
    corpus: str,
    sessions: Iterable[Any],
    *,
    extract: Callable[[list[Turn]], Sequence[Any]] | None = None,
    policy: Policy | None = None,
) -> IngestReport:
    """Stage and flush a whole corpus.

    ``extract`` maps a session's turns to extracted facts. It is injected so the
    caller decides where facts come from -- a language model, a fixture, or a
    rule-based stub -- and so this module never imports the extractor unless it
    is actually asked to use the default one.
    """
    ingestor = Ingestor(client, corpus, policy=policy)
    extractor = extract or _default_extractor()
    for idx, raw in enumerate(sessions):
        sid, ts, turns = _session_parts(raw, corpus, idx)
        ingestor.stage_session(sid, ts=ts, idx=idx, turns=turns)
        ingestor.stage_facts(list(extractor(turns)), sid=sid)
    return ingestor.flush()


def _default_extractor() -> Callable[[list[Turn]], Sequence[Any]]:
    """Resolve ``custodia.extract.extract_session`` late, and say so if it is absent."""
    try:
        from custodia.extract import extract_session
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise RuntimeError(
            "no extractor available: pass extract=... to ingest_sessions"
        ) from exc
    return extract_session


def _session_parts(raw: Any, corpus: str, idx: int) -> tuple[str, int, list[Turn]]:
    """Accept a session as a mapping, an object, or a bare list of turns."""
    if isinstance(raw, Mapping):
        sid = str(raw.get("sid") or raw.get("id") or f"s{idx}")
        ts = int(raw.get("ts") or 0)
        turns_raw = list(raw.get("turns") or [])
    elif hasattr(raw, "turns"):
        sid = str(getattr(raw, "sid", None) or getattr(raw, "id", None) or f"s{idx}")
        ts = int(getattr(raw, "ts", 0) or 0)
        turns_raw = list(raw.turns)
    else:
        turns_raw = list(raw)
        sid = str(getattr(turns_raw[0], "sid", "") or f"s{idx}") if turns_raw else f"s{idx}"
        ts = int(getattr(turns_raw[0], "ts", 0) or 0) if turns_raw else 0

    turns = [_as_turn(t, corpus, sid, idx, position) for position, t in enumerate(turns_raw)]
    if not ts and turns:
        ts = turns[0].ts
    return sid, ts, turns


def _as_turn(raw: Any, corpus: str, sid: str, sidx: int, position: int) -> Turn:
    if isinstance(raw, Turn):
        return raw
    def get(name: str, default: Any = None) -> Any:
        if isinstance(raw, Mapping):
            return raw.get(name, default)
        return getattr(raw, name, default)

    role = str(get("role", "user") or "user")
    origin = str(get("origin", "") or "")
    tier_raw = get("tier", None)
    tier = Tier.parse(tier_raw) if tier_raw is not None else tier_for_role(role, external=bool(origin))
    return Turn(
        corpus=corpus,
        sid=sid,
        idx=int(get("idx", position) if get("idx", None) is not None else position),
        sidx=sidx,
        role=role,
        text=str(get("text", "") or ""),
        ts=int(get("ts", 0) or 0),
        tier=tier,
        origin=origin,
    )


def _attr(item: Any, *names: str, default: Any = None) -> Any:
    """Read a field from an extracted fact, whether it is an object or a mapping."""
    for name in names:
        if isinstance(item, Mapping):
            if name in item and item[name] is not None:
                return item[name]
            continue
        value = getattr(item, name, None)
        if value is not None:
            return value
    return default
