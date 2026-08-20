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


#: separator for the corpus-scoped entity key. `algo.MSpaths` seeds from a
#: property value across the whole graph, with no way to restrict it to one
#: corpus, so the corpus has to be *inside* the value being matched. Without
#: this, a second tenant's entities seed the walk and the bounded path budget is
#: spent on somebody else's data.
CKEY_SEP = "::"


def corpus_key(corpus: str, norm: str) -> str:
    """The value `Entity.ckey` carries, and the value retrieval seeds on."""
    return f"{corpus}{CKEY_SEP}{norm}"


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
# predicates
# --------------------------------------------------------------------------- #

#: A closed vocabulary for the claims people actually make about themselves.
#:
#: Supersession is only detectable when the same slot is named the same way
#: twice. Left free, an extractor writes ``has_usual_order`` in January and
#: ``prefers`` in April, and the graph never learns that the second replaced the
#: first -- it just holds two unrelated facts and answers with whichever ranks
#: higher. Pinning the common slots is what makes "what is it now" and "what was
#: it in February" different questions with different answers.
#:
#: ``single`` means one value at a time: a new value supersedes the old. ``multi``
#: means values accumulate, so a second one corroborates rather than replaces.
#: Anything outside this table is free-form and treated as ``multi``, which is
#: the safe default -- a wrong supersession silently deletes a true fact from the
#: answerable set, while a missed one only leaves both on the record.
PREDICATES: dict[str, str] = {
    # identity
    "name": "single",
    "birthday": "single",
    "age": "single",
    "pronouns": "single",
    "nationality": "single",
    "blood_type": "single",
    # place
    "lives_in": "single",
    "works_at": "single",
    "job_title": "single",
    "office_location": "single",
    "born_in": "single",
    "travelling_to": "multi",
    "visited": "multi",
    # health
    "allergy": "multi",
    "dietary_restriction": "multi",
    "medical_condition": "multi",
    "medication": "multi",
    # preference
    # one slot for "the thing they always order", because two slots for the
    # same idea is how a supersession gets missed
    "usual_order": "single",
    "preferred_food": "multi",
    "preferred_time": "single",
    "preferred_place": "single",
    "preferred_seat": "single",
    "dislikes": "multi",
    "avoids": "multi",
    # possessions and tools
    "device": "single",
    "vehicle": "single",
    "pet": "multi",
    "uses_tool": "multi",
    # people
    "sibling": "multi",
    "parent": "multi",
    "child": "multi",
    "partner": "single",
    "friend": "multi",
    "colleague": "multi",
    "manager": "single",
    # activity and schedule
    "training_for": "multi",
    "attending": "multi",
    "member_of": "multi",
    "recurring_meeting": "single",
    "working_on": "multi",
    "goal": "multi",
    # contact
    "email": "single",
    "phone": "single",
    "handle": "multi",
    # what an untrusted source asserts, kept distinct from what is true
    "asserts": "multi",
}

#: predicates whose value is replaced rather than accumulated
SINGLE_VALUED = frozenset(k for k, v in PREDICATES.items() if v == "single")


def is_single_valued(predicate: str) -> bool:
    """Whether a new value for this slot supersedes the previous one.

    Unknown predicates are multi-valued: an extractor inventing a slot name must
    not be able to cause a true fact to be marked superseded.
    """
    return (predicate or "").strip().lower() in SINGLE_VALUED


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
