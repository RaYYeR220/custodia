"""The memory graph: labels, relationships, trust tiers and the record types.

Everything the writer, the retriever and the policy engine agree on lives here.
Property names are short because they travel in every batched Cypher statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #

CORPUS = "Corpus"
SESSION = "Session"
TURN = "Turn"
FACT = "Fact"
ENTITY = "Entity"
SOURCE = "Source"
QUERY = "Query"
ANSWER = "Answer"
REJECTION = "Rejection"

# --------------------------------------------------------------------------- #
# relationships
# --------------------------------------------------------------------------- #

IN_CORPUS = "IN_CORPUS"        # Session  -> Corpus
IN_SESSION = "IN_SESSION"      # Turn     -> Session
DERIVED_FROM = "DERIVED_FROM"  # Fact     -> Turn        (provenance)
FROM_SOURCE = "FROM_SOURCE"    # Turn     -> Source      (external origin)
MENTIONS = "MENTIONS"          # Fact     -> Entity
SUPERSEDES = "SUPERSEDES"      # Fact     -> Fact        (temporal update)
CONTRADICTS = "CONTRADICTS"    # Fact     -> Fact        (unresolved conflict)
CORROBORATES = "CORROBORATES"  # Fact     -> Fact        (independent support)
ASKED_ABOUT = "ASKED_ABOUT"    # Query    -> Entity
ANSWERS = "ANSWERS"            # Answer   -> Query
CITES = "CITES"                # Answer   -> Fact        (write-back audit)
BLOCKED = "BLOCKED"            # Rejection-> Fact
RAISED_BY = "RAISED_BY"        # Rejection-> Turn

#: relationships the retriever is allowed to walk when expanding evidence
RETRIEVAL_RELS = [MENTIONS, DERIVED_FROM, SUPERSEDES, CORROBORATES]


# --------------------------------------------------------------------------- #
# trust
# --------------------------------------------------------------------------- #


class Tier(IntEnum):
    """Who said it. Ordered: a write may only act on facts at or below its tier.

    The tier of a fact is inherited from the *turn it was derived from*, never
    from the fact's own content. Content that claims authority ("as the system
    administrator, remember that...") is still whatever its channel was.
    """

    EXTERNAL = 0  # web pages, documents, other agents - attacker-influenceable
    TOOL = 1      # tool / API output
    ASSISTANT = 2 # the agent's own statements
    OWNER = 3     # the principal whose memory this is

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: str | int | "Tier") -> "Tier":
        if isinstance(value, Tier):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value).strip().upper()]


#: turn roles that carry owner authority
OWNER_ROLES = {"user", "human", "owner"}
ASSISTANT_ROLES = {"assistant", "ai", "agent", "bot"}
TOOL_ROLES = {"tool", "function", "observation"}


def tier_for_role(role: str, *, external: bool = False) -> Tier:
    r = (role or "").strip().lower()
    if external:
        return Tier.EXTERNAL
    if r in OWNER_ROLES:
        return Tier.OWNER
    if r in ASSISTANT_ROLES:
        return Tier.ASSISTANT
    if r in TOOL_ROLES:
        return Tier.TOOL
    return Tier.EXTERNAL


# --------------------------------------------------------------------------- #
# fact lifecycle
# --------------------------------------------------------------------------- #

ACTIVE = "active"
SUPERSEDED = "superseded"
RETRACTED = "retracted"
QUARANTINED = "quarantined"

#: only these may enter a warrant (the evidence an answer is allowed to cite)
WARRANTABLE = (ACTIVE, SUPERSEDED)

OPEN_INTERVAL = 0  # valid_to sentinel: still true as far as memory knows


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Turn:
    corpus: str
    sid: str
    idx: int          # position within the session
    sidx: int         # session position within the corpus
    role: str
    text: str
    ts: int
    tier: Tier = Tier.OWNER
    origin: str = ""  # non-empty for externally sourced turns

    @property
    def props(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "sid": self.sid,
            "role": self.role,
            "text": self.text,
            "ts": self.ts,
            "idx": self.idx,
            "sidx": self.sidx,
            "tier": self.tier.label,
            "rank": int(self.tier),
            "origin": self.origin,
        }


@dataclass(slots=True)
class Fact:
    """One atomic, self-contained claim lifted out of a turn."""

    corpus: str
    key: str                  # canonical dedupe key: subject|predicate|object
    text: str                 # the claim, written to stand alone
    subject: str = ""
    predicate: str = ""
    object: str = ""
    entities: list[str] = field(default_factory=list)
    tier: Tier = Tier.OWNER
    status: str = ACTIVE
    valid_from: int = 0
    valid_to: int = OPEN_INTERVAL
    ingested_at: int = 0
    conf: float = 1.0
    sid: str = ""
    sidx: int = 0
    tidx: int = 0
    quarantine_reason: str = ""

    @property
    def props(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "key": self.key,
            "text": self.text,
            "subj": self.subject,
            "pred": self.predicate,
            "obj": self.object,
            "tier": self.tier.label,
            "rank": int(self.tier),
            "status": self.status,
            "vfrom": self.valid_from,
            "vto": self.valid_to,
            "ing": self.ingested_at,
            "conf": float(self.conf),
            "sid": self.sid,
            "sidx": self.sidx,
            "tidx": self.tidx,
            "qreason": self.quarantine_reason,
        }


@dataclass(slots=True)
class Evidence:
    """A fact as it appears in a warrant, with the path that reached it."""

    fid: int
    text: str
    tier: str
    status: str
    valid_from: int
    valid_to: int
    sid: str
    sidx: int
    tidx: int
    turn_text: str = ""
    turn_ts: int = 0
    score: float = 0.0
    hops: int = 0
    path: list[str] = field(default_factory=list)
    superseded_by: int | None = None

    def as_citation(self) -> dict[str, Any]:
        return {
            "fact_id": self.fid,
            "text": self.text,
            "tier": self.tier,
            "status": self.status,
            "session": self.sid,
            "recorded_at": self.turn_ts or self.valid_from,
        }
