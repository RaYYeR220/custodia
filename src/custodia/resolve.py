"""Entity resolution and temporal reconciliation.

Two jobs live here, and they are the same job seen from two angles.

The first is deciding when two strings name the same thing, because the graph
anchors retrieval on ``Entity.norm`` and a seed that misses by an article or a
possessive finds nothing. The normalisation is deliberately blunt and
deterministic -- no model, no embedding, no network -- so the same corpus always
produces the same entity keys and therefore the same vertex ids. On top of that
sits :func:`fold_aliases`, which handles the drift a long conversation always
has: someone introduces themselves in full in January and is on first-name terms
by March, and unless the two spellings become one key nothing they said in
January ever reconciles with anything they said afterwards.

The second is deciding what a new claim does to the claims already held. "My gym
is Ironworks" followed by "I switched to Fitwell" is a supersession; "I go to
Ironworks" said twice in different sessions is corroboration; "I own a bike" and
"I own a car" are neither, because ``owns`` is not a single-valued predicate and
treating it as one silently deletes memory. Which predicates replace their value
and which accumulate is not decided here: it comes from the shared vocabulary in
:mod:`custodia.schema`, so the extractor, the writer and the retriever cannot
disagree about it.

The rule that matters most is the one that refuses. A supersession is a *write
against an existing fact*, so it goes through :class:`~custodia.policy.Policy`
first. When the policy refuses -- an external page trying to overwrite something
the principal said -- the older fact stays active and the conflict is recorded
as a ``CONTRADICTS`` edge. Memory keeps the disagreement rather than resolving
it in the attacker's favour.

Everything here is order-independent: groups are sorted before they are walked,
so reconciling the same facts in a different input order produces the same
edges, the same statuses and the same intervals.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from custodia import schema
from custodia.policy import Decision, Policy
from custodia.schema import QUARANTINED, SUPERSEDED, Fact

# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

_LEADING_ARTICLES = ("the ", "a ", "an ")
_POSSESSIVE = re.compile(r"['’`]s\b")
_TRAILING_POSSESSIVE = re.compile(r"(\w)['’`](?=\s|$)")
_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_SEPARATORS = re.compile(r"[_\-]+")
_WHITESPACE = re.compile(r"\s+")
_PREDICATE_PUNCT = re.compile(r"[^0-9a-z]+")


def _fold(value: str) -> str:
    """Case-fold and strip diacritics, so ``Café`` and ``cafe`` are one anchor."""
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold()


def normalize_entity(name: str) -> str:
    """The key an entity is stored and seeded under.

    Lowercase, diacritics folded, possessives and punctuation removed,
    underscores and hyphens treated as spaces, leading articles dropped,
    whitespace collapsed. ``"The Ironworks' Gym"`` and ``"ironworks gym"``
    normalise to the same string, which is what makes them the same vertex.
    """
    folded = _fold(name)
    folded = _POSSESSIVE.sub("", folded)
    folded = _TRAILING_POSSESSIVE.sub(r"\1", folded)
    folded = _SEPARATORS.sub(" ", folded)
    folded = _NON_WORD.sub(" ", folded)
    folded = _WHITESPACE.sub(" ", folded).strip()
    changed = True
    while changed:
        changed = False
        for article in _LEADING_ARTICLES:
            if folded.startswith(article) and len(folded) > len(article):
                folded = folded[len(article) :]
                changed = True
                break
    return folded.strip()


def normalize_predicate(predicate: str) -> str:
    """Predicates are snake-cased so ``works at`` and ``works_at`` are one key."""
    return _PREDICATE_PUNCT.sub("_", _fold(predicate)).strip("_")


def resolve_entities(
    names: Iterable[str],
    *,
    aliases: dict[str, str] | None = None,
) -> list[str]:
    """Normalise, alias and de-duplicate a run of entity mentions.

    Aliases map any spelling to a canonical one (``{"ironworks gym":
    "ironworks"}``); both sides are normalised first, so the caller can write
    the table in whatever form reads best. Order is preserved because the first
    mention in a fact is usually its subject, and the seeder weights it.
    """
    table = {
        normalize_entity(key): normalize_entity(value)
        for key, value in (aliases or {}).items()
        if normalize_entity(key) and normalize_entity(value)
    }
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in names:
        norm = normalize_entity(raw)
        if not norm:
            continue
        norm = table.get(norm, norm)
        if norm in seen:
            continue
        seen.add(norm)
        resolved.append(norm)
    return resolved


# --------------------------------------------------------------------------- #
# alias folding
# --------------------------------------------------------------------------- #

#: lowercase particles that legitimately sit inside a proper name
_NAME_PARTICLES = frozenset({"de", "da", "do", "dos", "das", "del", "della", "di",
                             "van", "von", "der", "den", "la", "le", "el", "bin",
                             "al", "of", "and"})

#: longest key we will treat as one person or place
_MAX_ALIAS_TOKENS = 4


def _looks_like_a_proper_name(display: str) -> bool:
    """A blunt test for "this is a name, not a noun phrase".

    ``Nora Salgado`` and ``Alfama, Lisbon`` pass; ``coffee shop`` and ``the
    14-inch macbook pro`` do not. It exists so that folding a longer key onto a
    shorter one only ever happens between things that are plausibly the same
    named entity written two ways.
    """
    tokens = [t for t in re.split(r"[\s,]+", display.strip()) if t]
    if not tokens:
        return False
    for token in tokens:
        if token.casefold() in _NAME_PARTICLES:
            continue
        first = token[0]
        if not first.isalpha() or first != first.upper():
            return False
    return True


def fold_aliases(
    entities: Iterable[str] | dict[str, str],
    *,
    aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map spellings of one entity onto the key the corpus should use for it.

    Conversations drift: someone introduces themselves as *Nora Salgado* in
    January and is *Nora* by March, and the two never reconcile because they are
    two vertices. This folds the longer key onto the shorter one when the
    shorter one is a leading token of it and **nothing else competes for it** --
    with both ``nora salgado`` and ``nora costa`` present, neither folds, because
    merging two people is a worse failure than leaving one person split.

    Pass ``entities`` as a ``{key: display}`` mapping to get the proper-name
    guard, which is what keeps ``coffee shop`` from folding onto ``coffee``.
    ``aliases`` pins pairs explicitly and always wins over anything inferred.

    Returns only the keys that move; identity mappings are left out.
    """
    display: dict[str, str] = {}
    if isinstance(entities, dict):
        for raw_key, raw_display in entities.items():
            key = normalize_entity(raw_key)
            if key:
                display[key] = str(raw_display or raw_key)
        guarded = True
    else:
        for raw_key in entities:
            key = normalize_entity(raw_key)
            if key:
                display.setdefault(key, key)
        guarded = False

    keys = set(display)
    folded: dict[str, str] = {}

    for source, target in (aliases or {}).items():
        src, dst = normalize_entity(source), normalize_entity(target)
        if src and dst and src != dst:
            folded[src] = dst

    # a key -> every longer key that would fold onto it
    contenders: dict[str, list[str]] = {}
    for key in sorted(keys):
        if key in folded:
            continue
        tokens = key.split()
        if not 1 < len(tokens) <= _MAX_ALIAS_TOKENS:
            continue
        if guarded and not _looks_like_a_proper_name(display[key]):
            continue
        for cut in range(len(tokens) - 1, 0, -1):
            head = " ".join(tokens[:cut])
            if head in keys and head != key:
                if guarded and not _looks_like_a_proper_name(display[head]):
                    break
                contenders.setdefault(head, []).append(key)
                break

    for head, longer in contenders.items():
        if len(longer) == 1:
            folded[longer[0]] = head

    return _resolve_chains(folded)


def _resolve_chains(mapping: dict[str, str]) -> dict[str, str]:
    """Collapse ``a -> b -> c`` to ``a -> c``, and drop anything that loops."""
    resolved: dict[str, str] = {}
    for source in mapping:
        target = mapping[source]
        seen = {source}
        while target in mapping and target not in seen:
            seen.add(target)
            target = mapping[target]
        if target != source:
            resolved[source] = target
    return resolved


def canonical_key(subject: str, predicate: str, object: str) -> str:
    """The dedupe key a ``Fact`` is identified by within a corpus.

    Two turns that state the same triple produce the same key and therefore the
    same vertex id, which is what makes a re-ingest idempotent and what lets a
    claim repeated across sessions accumulate provenance instead of duplicates.
    """
    return f"{normalize_entity(subject)}|{normalize_predicate(predicate)}|{normalize_entity(object)}"


# --------------------------------------------------------------------------- #
# predicate arity
# --------------------------------------------------------------------------- #

#: Fallback patterns for predicates outside :data:`custodia.schema.PREDICATES`.
#: The shared vocabulary decides first and always; these only catch free-form
#: slots an extractor invented. Kept small and explicit, because a predicate
#: wrongly treated as single-valued makes memory forget things, which is why
#: anything unrecognised stays multi-valued.
SINGLE_VALUED_PATTERNS: tuple[str, ...] = (
    r"(?:is|was|are|equals|resolves_to)",
    r"(?:has_)?(?:name|full_name|first_name|last_name|nickname|username|handle)",
    r"(?:has_)?(?:age|birthday|birth_date|date_of_birth|dob|gender|pronouns)",
    r"(?:has_)?(?:email|email_address|phone|phone_number|address|home_address)",
    r"(?:lives_in|located_in|based_in|resides_in|home_city|hometown|moved_to)",
    r"(?:works_at|works_for|employed_by|employer|job_title|title|role|position|occupation)",
    r"(?:prefers|preference|goes_to|attends|banks_with|married_to|spouse|partner)",
    r"(?:status|state|weight|height|salary|timezone|locale)",
    r"(?:password|passcode|pin|api_key|access_token|secret)",
    r"(?:current|primary|main|preferred|favorite|favourite|default|usual)_\w+",
)

#: Free-form predicates that are naturally many-to-one and must never supersede.
#: Listed explicitly rather than left to the default so the intent is testable.
MULTI_VALUED: frozenset[str] = frozenset(
    {
        "owns", "own", "owned", "visited", "visits", "knows", "know", "likes",
        "dislikes", "has", "have", "includes", "contains", "member_of",
        "attended", "read", "watched", "met", "bought", "tried", "speaks",
        "plays", "uses", "used", "worked_at", "studied_at", "mentions",
        "related_to", "friend_of", "tagged", "interested_in", "travelled_to",
        "traveled_to", "ate", "subscribes_to", "follows", "collaborated_with",
    }
)

_SINGLE_VALUED = re.compile("|".join(f"(?:{p})" for p in SINGLE_VALUED_PATTERNS))


def is_single_valued(predicate: str) -> bool:
    """True when a newer value for this predicate replaces the older one.

    The shared vocabulary in :mod:`custodia.schema` decides for every slot it
    defines -- extractor, writer and retriever have to agree on arity or the
    same corpus reconciles differently depending on who asked. The patterns
    below are consulted only for free-form predicates the vocabulary does not
    cover, and anything they do not recognise stays multi-valued.
    """
    key = normalize_predicate(predicate)
    if not key:
        return False
    if key in schema.PREDICATES:
        return schema.is_single_valued(key)
    if key in MULTI_VALUED:
        return False
    return _SINGLE_VALUED.fullmatch(key) is not None


def same_object(left: str, right: str) -> bool:
    """Whether two object strings name the same value.

    Exact after normalisation, or one is a proper token-subset of the other --
    ``"Ironworks"`` and ``"Ironworks Gym"`` are the same gym, and treating them
    as different values would turn a corroboration into a false supersession.
    """
    a, b = normalize_entity(left), normalize_entity(right)
    if a == b:
        return bool(a) or left == right
    if not a or not b:
        return False
    ta, tb = set(a.split()), set(b.split())
    return ta < tb or tb < ta


# --------------------------------------------------------------------------- #
# reconciliation
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Reconciliation:
    """What a batch of facts did to the facts already held."""

    supersedes: list[tuple[Fact, Fact]] = field(default_factory=list)   # (newer, older)
    contradicts: list[tuple[Fact, Fact]] = field(default_factory=list)
    corroborates: list[tuple[Fact, Fact]] = field(default_factory=list)
    #: older facts whose status, valid_to or confidence changed and therefore
    #: have to be written back even though the caller did not stage them
    updated: list[Fact] = field(default_factory=list)
    #: every supersession the policy refused, as ``(newer, older, decision)``,
    #: so the writer can raise a ``Rejection`` for each one instead of quietly
    #: downgrading the attempt to a contradiction
    refusals: list[tuple[Fact, Fact, Decision]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "supersedes": len(self.supersedes),
            "contradicts": len(self.contradicts),
            "corroborates": len(self.corroborates),
            "updated": len(self.updated),
            "refusals": len(self.refusals),
        }


#: how much independent support in a second session is worth
CORROBORATION_BONUS = 0.05


def _order(fact: Fact) -> tuple[int, int, int, str]:
    """Total order on a group: later wins, ties broken by position in the corpus."""
    return (fact.valid_from, fact.sidx, fact.tidx, fact.key)


def _bump(fact: Fact, amount: float = CORROBORATION_BONUS) -> bool:
    raised = min(1.0, round(float(fact.conf) + amount, 6))
    if raised > fact.conf:
        fact.conf = raised
        return True
    return False


def _absorb(keeper: Fact, duplicate: Fact) -> None:
    """Fold an identical triple into the vertex it shares.

    Identical ``(subject, predicate, object)`` is the same fact by construction
    -- same key, same id -- so a repeat is extra evidence rather than a second
    claim. The surviving fact keeps the earliest ``valid_from`` (that is when
    the claim started being true) and the highest tier that ever asserted it.
    """
    stamps = [t for t in (keeper.valid_from, duplicate.valid_from) if t]
    keeper.valid_from = min(stamps) if stamps else 0
    if int(duplicate.tier) > int(keeper.tier):
        keeper.tier = duplicate.tier
        if duplicate.status != QUARANTINED and keeper.status == QUARANTINED:
            keeper.status = duplicate.status
            keeper.quarantine_reason = ""
    if duplicate.sid and duplicate.sid != keeper.sid:
        _bump(keeper)


def reconcile(
    facts: list[Fact],
    *,
    policy: Policy,
    existing: list[Fact] | None = None,
) -> Reconciliation:
    """Work out supersession, contradiction and corroboration over a fact set.

    ``existing`` is whatever the graph already holds for the same corpus; those
    facts take part as the older side and come back in ``updated`` when their
    status or interval moved.
    """
    result = Reconciliation()

    pool: dict[str, Fact] = {}
    for fact in list(existing or []) + list(facts):
        if not fact.key:
            fact.key = canonical_key(fact.subject, fact.predicate, fact.object)
        held = pool.get(fact.key)
        if held is None:
            pool[fact.key] = fact
        elif held is not fact:
            # the earlier assertion survives as the vertex, whichever order the
            # caller handed them over in
            keeper, duplicate = (held, fact) if _order(held) <= _order(fact) else (fact, held)
            _absorb(keeper, duplicate)
            pool[fact.key] = keeper

    groups: dict[tuple[str, str], list[Fact]] = {}
    for fact in pool.values():
        key = (normalize_entity(fact.subject), normalize_predicate(fact.predicate))
        groups.setdefault(key, []).append(fact)

    changed: dict[int, Fact] = {}

    for group_key in sorted(groups):
        members = sorted(groups[group_key], key=_order)
        if len(members) < 2:
            continue
        single = is_single_valued(group_key[1])
        # facts still standing as the live claim for this (subject, predicate)
        open_heads: list[Fact] = []

        for fact in members:
            quarantined = fact.status == QUARANTINED
            matched = [head for head in open_heads if same_object(fact.object, head.object)]

            for head in matched:
                if fact.sid and head.sid and fact.sid != head.sid:
                    result.corroborates.append((fact, head))
                    _bump(fact)
                    if _bump(head):
                        changed[id(head)] = head

            if matched:
                if not quarantined:
                    open_heads.append(fact)
                continue

            if single:
                for head in list(open_heads):
                    decision = None if quarantined else policy.admit(
                        fact, target=head, op="supersede"
                    )
                    if decision is not None and decision.admitted:
                        result.supersedes.append((fact, head))
                        head.status = SUPERSEDED
                        head.valid_to = max(fact.valid_from, head.valid_from)
                        changed[id(head)] = head
                        open_heads.remove(head)
                        continue
                    # Refused, or attempted by a fact that was already
                    # quarantined at ingest. The older fact stays live and the
                    # conflict is recorded, so retrieval can show both sides
                    # instead of memory resolving it in the attacker's favour.
                    result.contradicts.append((fact, head))
                    if decision is not None:
                        result.refusals.append((fact, head, decision))
                        if decision.status == QUARANTINED and not quarantined:
                            fact.status = QUARANTINED
                            fact.quarantine_reason = decision.reason
                            quarantined = True

            if not quarantined:
                open_heads.append(fact)

    result.updated = [fact for _, fact in sorted(changed.items(), key=lambda kv: _order(kv[1]))]
    return result
