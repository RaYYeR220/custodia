"""Write-back, provenance explanations, and the integrity checks that prove them.

Two jobs, and they are the same job seen from either end.

Going in: every question and every answer is written back as
``(:Answer)-[:ANSWERS]->(:Query)`` with ``CITES`` edges to the exact facts the
answer rested on. **Abstentions are written too**, with ``status = "abstained"``
and the check that failed. A record that only kept the successes would be
marketing; the refusals are the part of the trail worth having, because they are
where the system did its job.

Coming out: :meth:`Auditor.explain` walks a fact back to the turn and session it
came from, and :meth:`Auditor.integrity` proves - by querying, not by asserting -
that the invariants the architecture claims actually hold in this graph. Both
work within HydraDB's read subset, which has no ``IN`` and no ``IS NULL``, so
"every Fact has a DERIVED_FROM" is expressed as the difference between two id
sets rather than as an absence predicate.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from custodia import ids, schema
from custodia.hydra.client import HydraClient

log = logging.getLogger("custodia.audit")

ANSWERED = "answered"
ABSTAINED = "abstained"

#: how much answer text is kept on the node; the full text lives in the response
ANSWER_LIMIT = 4000


class Auditor:
    """Records what was asked and answered, and audits the record afterwards."""

    def __init__(self, client: HydraClient, corpus: str) -> None:
        self.client = client
        self.corpus = corpus

    # ------------------------------------------------------------- write-back

    def record(self, question: str, warrant: Any, verdict: Any) -> int:
        """Write the query, the answer and its citations. Returns the answer id.

        Ids are derived from the corpus, the question and the ask time, so a
        replayed request rewrites the same two vertices instead of growing the
        history - the same idempotence the ingest side relies on.
        """
        asked_at = int(getattr(warrant, "asked_at", 0) or time.time())
        qid = ids.query_id(self.corpus, question, asked_at)
        aid = ids.answer_id(self.corpus, qid)

        as_of = getattr(warrant, "as_of", None)
        self.client.upsert_nodes(
            schema.QUERY,
            [
                {
                    "id": qid,
                    "corpus": self.corpus,
                    "text": question,
                    "ts": asked_at,
                    "asof": int(as_of or 0),
                    "wsize": len(getattr(warrant, "evidence", []) or []),
                    "paths": int(getattr(warrant, "paths_examined", 0) or 0),
                    "quarantined": int(getattr(warrant, "quarantined_seen", 0) or 0),
                }
            ],
        )

        answered = bool(getattr(verdict, "answered", False))
        citations = [int(c) for c in (getattr(verdict, "citations", []) or [])]
        self.client.upsert_nodes(
            schema.ANSWER,
            [
                {
                    "id": aid,
                    "corpus": self.corpus,
                    "text": str(getattr(verdict, "answer", ""))[:ANSWER_LIMIT],
                    "ts": asked_at,
                    "status": ANSWERED if answered else ABSTAINED,
                    "reason": str(getattr(verdict, "abstained_because", "") or ""),
                    "model": str(getattr(verdict, "model", "") or ""),
                    "ncit": len(citations),
                    "verified": int(getattr(verdict, "verified", 0) or 0),
                    "latency": int(getattr(verdict, "latency_ms", 0) or 0),
                    "checks": ",".join(getattr(verdict, "checks", []) or []),
                }
            ],
        )

        self.client.merge_edges(
            schema.ANSWERS,
            schema.ANSWER,
            schema.QUERY,
            [{"s": aid, "d": qid, "rid": ids.edge_id(schema.ANSWERS, aid, qid)}],
        )
        if citations:
            self.client.merge_edges(
                schema.CITES,
                schema.ANSWER,
                schema.FACT,
                [
                    {"s": aid, "d": fid, "rid": ids.edge_id(schema.CITES, aid, fid)}
                    for fid in citations
                ],
            )

        seeds = (getattr(warrant, "seeds", None) or {}).get("entities") or []
        if seeds:
            # the edge write MATCHes both ends, so a seed whose entity is no
            # longer in the graph simply does not produce an edge
            self.client.merge_edges(
                schema.ASKED_ABOUT,
                schema.QUERY,
                schema.ENTITY,
                [
                    {
                        "s": qid,
                        "d": ids.entity_id(self.corpus, norm),
                        "rid": ids.edge_id(
                            schema.ASKED_ABOUT, qid, ids.entity_id(self.corpus, norm)
                        ),
                    }
                    for norm in dict.fromkeys(seeds)
                ],
            )
        return aid

    # ----------------------------------------------------------------- reading

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Questions and what was said back, newest first, abstentions included."""
        rows = self.client.run(
            f"MATCH (a:{schema.ANSWER})-[:{schema.ANSWERS}]->(q:{schema.QUERY}) "
            "WHERE a.corpus = $corpus "
            "RETURN a.id AS answer_id, q.id AS query_id, q.text AS question, "
            "q.asof AS as_of, q.wsize AS warrant_size, q.quarantined AS quarantined, "
            "a.text AS answer, a.status AS status, a.reason AS reason, a.ts AS ts, "
            "a.model AS model, a.ncit AS citations, a.verified AS verified, "
            "a.latency AS latency_ms, a.checks AS checks "
            "ORDER BY a.ts DESC LIMIT $limit",
            corpus=self.corpus,
            limit=int(limit),
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["answered"] = item.get("status") == ANSWERED
            item["checks"] = [c for c in str(item.get("checks") or "").split(",") if c]
            item["cited"] = self._citations(int(item["answer_id"]))
            out.append(item)
        return out

    def _citations(self, answer_id: int) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.client.run(
                f"MATCH (a:{schema.ANSWER})-[:{schema.CITES}]->(f:{schema.FACT}) "
                "WHERE a.id = $aid RETURN f.id AS fact_id, f.text AS text, "
                "f.tier AS tier, f.status AS status",
                aid=int(answer_id),
            )
        ]

    def explain(self, fact_id: int) -> dict[str, Any]:
        """A fact's full chain of custody: turn, session, and its lineage in time."""
        fact = self.client.run(
            f"MATCH (f:{schema.FACT}) WHERE f.id = $fid "
            "RETURN f.id AS fact_id, f.corpus AS corpus, f.key AS key, f.text AS text, "
            "f.subj AS subject, f.pred AS predicate, f.obj AS object, f.tier AS tier, "
            "f.status AS status, f.vfrom AS valid_from, f.vto AS valid_to, "
            "f.ing AS ingested_at, f.conf AS confidence, f.sid AS session, "
            "f.sidx AS session_index, f.tidx AS turn_index, f.qreason AS quarantine_reason",
            fid=int(fact_id),
        )
        if not fact:
            return {"fact_id": int(fact_id), "found": False}

        out: dict[str, Any] = {"found": True, **fact[0]}
        out["open_interval"] = int(out.get("valid_to") or 0) == schema.OPEN_INTERVAL

        chain = self.client.run(
            f"MATCH (f:{schema.FACT})-[:{schema.DERIVED_FROM}]->(t:{schema.TURN})"
            f"-[:{schema.IN_SESSION}]->(s:{schema.SESSION}) WHERE f.id = $fid "
            "RETURN t.id AS turn_id, t.text AS turn_text, t.role AS turn_role, "
            "t.ts AS turn_ts, t.idx AS turn_index, t.tier AS turn_tier, "
            "t.origin AS turn_origin, s.id AS session_id, s.sid AS session_name, "
            "s.sidx AS session_index",
            fid=int(fact_id),
        )
        out["provenance"] = dict(chain[0]) if chain else None
        if not chain:
            # the fact is on the graph without the turn it came from, which is
            # the one shape the writer is supposed to make impossible
            turn = self.client.run(
                f"MATCH (f:{schema.FACT})-[:{schema.DERIVED_FROM}]->(t:{schema.TURN}) "
                "WHERE f.id = $fid RETURN t.id AS turn_id, t.text AS turn_text, "
                "t.role AS turn_role, t.ts AS turn_ts",
                fid=int(fact_id),
            )
            out["provenance"] = dict(turn[0]) if turn else None
            out["orphan"] = not turn

        out["mentions"] = [
            str(row["norm"])
            for row in self.client.run(
                f"MATCH (f:{schema.FACT})-[:{schema.MENTIONS}]->(e:{schema.ENTITY}) "
                "WHERE f.id = $fid RETURN e.norm AS norm ORDER BY e.norm",
                fid=int(fact_id),
            )
        ]
        out["supersedes"] = [
            dict(row)
            for row in self.client.run(
                f"MATCH (f:{schema.FACT})-[:{schema.SUPERSEDES}]->(b:{schema.FACT}) "
                "WHERE f.id = $fid RETURN b.id AS fact_id, b.text AS text, "
                "b.vfrom AS valid_from, b.vto AS valid_to",
                fid=int(fact_id),
            )
        ]
        out["superseded_by"] = [
            dict(row)
            for row in self.client.run(
                f"MATCH (a:{schema.FACT})-[:{schema.SUPERSEDES}]->(f:{schema.FACT}) "
                "WHERE f.id = $fid RETURN a.id AS fact_id, a.text AS text, "
                "a.vfrom AS valid_from, a.vto AS valid_to",
                fid=int(fact_id),
            )
        ]
        out["cited_by"] = [
            dict(row)
            for row in self.client.run(
                f"MATCH (a:{schema.ANSWER})-[:{schema.CITES}]->(f:{schema.FACT}) "
                "WHERE f.id = $fid RETURN a.id AS answer_id, a.text AS answer, "
                "a.ts AS ts ORDER BY a.ts DESC LIMIT 20",
                fid=int(fact_id),
            )
        ]
        return out

    def rejections(self, limit: int = 100) -> list[dict[str, Any]]:
        """Writes the policy engine refused, with the rule that fired.

        These are the attacks that were attempted rather than the facts that
        survived, which is the half of the trail a demo usually cannot show.
        """
        rows = self.client.run(
            f"MATCH (r:{schema.REJECTION})-[:{schema.RAISED_BY}]->(t:{schema.TURN}) "
            "WHERE r.corpus = $corpus "
            "RETURN r.id AS rejection_id, r.rule AS rule, r.text AS text, "
            "r.reason AS reason, r.ts AS ts, t.id AS turn_id, t.sid AS session, "
            "t.role AS role, t.tier AS tier "
            "ORDER BY r.ts DESC LIMIT $limit",
            corpus=self.corpus,
            limit=int(limit),
        )
        return [dict(row) for row in rows]

    # --------------------------------------------------------------- integrity

    def integrity(self) -> dict[str, Any]:
        """Prove the invariants by query. Judge-facing, so it must actually look.

        * every Fact has a ``DERIVED_FROM`` edge to the turn it came from;
        * no ``SUPERSEDES`` edge points at something that is not a Fact any more;
        * no quarantined fact was ever warrantable - that is, no answer cites one.
        """
        all_facts = {
            int(row["id"])
            for row in self.client.run(
                f"MATCH (f:{schema.FACT}) WHERE f.corpus = $corpus RETURN f.id AS id",
                corpus=self.corpus,
            )
        }
        derived = {
            int(row["id"])
            for row in self.client.run(
                f"MATCH (f:{schema.FACT})-[:{schema.DERIVED_FROM}]->(t:{schema.TURN}) "
                "WHERE f.corpus = $corpus RETURN DISTINCT f.id AS id",
                corpus=self.corpus,
            )
        }
        orphans = sorted(all_facts - derived)

        # HydraDB has no IS NULL, so a broken edge is found by comparing the
        # endpoints a labelled pattern matches against the endpoints an
        # unlabelled one does: the difference is edges into vanished facts
        endpoints = {
            (int(row["a"]), int(row["b"]))
            for row in self.client.run(
                f"MATCH (a:{schema.FACT})-[:{schema.SUPERSEDES}]->(b) "
                "WHERE a.corpus = $corpus RETURN a.id AS a, b.id AS b",
                corpus=self.corpus,
            )
        }
        intact = {
            (int(row["a"]), int(row["b"]))
            for row in self.client.run(
                f"MATCH (a:{schema.FACT})-[:{schema.SUPERSEDES}]->(b:{schema.FACT}) "
                "WHERE a.corpus = $corpus RETURN a.id AS a, b.id AS b",
                corpus=self.corpus,
            )
        }
        dangling = sorted(endpoints - intact)

        leaked = [
            int(row["id"])
            for row in self.client.run(
                f"MATCH (a:{schema.ANSWER})-[:{schema.CITES}]->(f:{schema.FACT}) "
                "WHERE f.corpus = $corpus AND f.status = $status "
                "RETURN DISTINCT f.id AS id",
                corpus=self.corpus,
                status=schema.QUARANTINED,
            )
        ]

        quarantined = self.client.run(
            f"MATCH (f:{schema.FACT}) WHERE f.corpus = $corpus AND f.status = $status "
            "RETURN count(*) AS n",
            corpus=self.corpus,
            status=schema.QUARANTINED,
        )

        return {
            "corpus": self.corpus,
            "facts": len(all_facts),
            "quarantined": int(quarantined[0]["n"]) if quarantined else 0,
            "orphan_facts": len(orphans),
            "dangling_supersedes": len(dangling),
            "quarantined_warrantable": len(leaked),
            "orphan_ids": orphans[:20],
            "dangling_ids": [list(pair) for pair in dangling[:20]],
            "quarantined_cited": leaked[:20],
            "ok": not orphans and not dangling and not leaked,
        }

    # ------------------------------------------------------------------ counts

    def counts(self) -> dict[str, int]:
        """Node totals for this corpus, for the status line and the demo."""
        out: dict[str, int] = {}
        for label in (
            schema.SESSION,
            schema.TURN,
            schema.FACT,
            schema.ENTITY,
            schema.QUERY,
            schema.ANSWER,
            schema.REJECTION,
        ):
            rows = self.client.run(
                f"MATCH (n:{label}) WHERE n.corpus = $corpus RETURN count(*) AS n",
                corpus=self.corpus,
            )
            out[label.lower()] = int(rows[0]["n"]) if rows else 0
        return out


def summarise(history: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Answered / abstained split over a history slice."""
    answered = sum(1 for h in history if h.get("answered"))
    return {"total": len(history), "answered": answered, "abstained": len(history) - answered}
