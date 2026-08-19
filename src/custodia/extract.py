"""Turns into facts.

Extraction is where provenance is either earned or lost, so two constraints shape
this module more than accuracy does.

*Windowing with overlap, attribution without it.* A claim is often spread over two
messages ("push my standing 9am to 9:30" means nothing without the turn that set
it), so the model is shown a sliding window with a few turns of context. But a
fact belongs to exactly one turn, and a turn is claimable in exactly one window -
the context prefix is labelled and its indices are refused by the parser. Facts
therefore cannot be attributed twice, and cannot be attributed to a turn the model
never saw.

*The predicate is a slot, not a phrase.* A model asked for a "snake_case verb
phrase" will write ``holds_title`` one day and ``job_title`` the next, and two
names for one slot means the graph holds two unrelated facts where it should
hold a supersession - which is most of what this product claims to do. So the
prompt carries the vocabulary from :data:`custodia.schema.PREDICATES`, and
whatever comes back is folded onto it here. Folding never invents: a predicate
the schema does not know survives as free-form, where it is merely unqueryable
rather than wrong, because unknown predicates are multi-valued and a wrong slot
would make memory retire a fact that is still true.

*The model is never the last word.* Anything it returns is filtered against the
schema (unknown keys dropped, turn index range-checked, timestamps re-derived from
the turn when unstated) because the reply is generated from captured content, and
captured content includes documents that try to give orders. A rule-based
extractor stands behind it so a run without credentials still produces a graph;
it is deliberately narrow - the principal's own first-person statements - rather
than an imitation of the model, and it labels every fact it makes ``rules`` at a
confidence that says so.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from custodia import config, prompts
from custodia.llm import LLM
from custodia.schema import OPEN_INTERVAL, OWNER_ROLES, PREDICATES, Tier, Turn

log = logging.getLogger("custodia.extract")

#: a window of six turns with a long quoted document in it still fits comfortably
EXTRACT_MAX_TOKENS = 3000

#: the rules extractor is a floor, not a model. Its facts say so.
RULES_CONF = 0.45
#: what a model-extracted fact scores when the model gave no confidence
LLM_CONF = 0.7

MAX_ENTITIES = 12
MAX_RULE_FACTS = 6

#: Spellings that mean a slot the schema already names. Kept short on purpose:
#: the generic rules below (snake-casing, dropping a leading ``has_``/``is_``)
#: cover most drift, and every entry here is a judgement about meaning that the
#: vocabulary in ``schema.PREDICATES`` should eventually make unnecessary.
#: Nothing here invents a slot - a fold is only taken when the target exists.
PREDICATE_SYNONYMS: dict[str, str] = {
    "has_usual_order": "usual_order",
    "has_usual": "usual_order",
    "usual": "usual_order",
    "orders": "usual_order",
    "prefers_drink": "usual_order",
    # a slot the vocabulary used to carry: two single-valued homes for one idea
    # is how a supersession goes missing, so anything still spelling it the old
    # way folds onto the one that survived
    "preferred_drink": "usual_order",
    "preferred_order": "usual_order",
    "holds_title": "job_title",
    "title": "job_title",
    "role": "job_title",
    "position": "job_title",
    "allergic_to": "allergy",
    "has_allergy": "allergy",
    "works_for": "works_at",
    "employed_by": "works_at",
    "employer": "works_at",
    "has_sibling": "sibling",
    "uses_device": "device",
    "is_using": "uses_tool",
    "uses": "uses_tool",
    "based_in": "lives_in",
    "located_in": "lives_in",
    "lives_at": "lives_in",
    "is_in": "lives_in",
}

#: A predicate naming a value that has already ended. The end of an interval is
#: ``valid_to`` on the fact that is ending, which reconciliation derives from the
#: value that replaced it; a slot of its own would hide the supersession behind a
#: name no question ever asks for.
_ENDED_PREFIXES = ("previously_", "formerly_", "former_", "used_to_", "past_", "ex_", "old_")
_ENDED_SUFFIXES = ("_previously", "_before", "_until", "_formerly")

#: subjects that mean "the person whose memory this is"
_SELF_SUBJECTS = frozenset(
    {"i", "me", "my", "mine", "myself", "user", "the user", "principal", "the principal",
     "owner", "the owner", "self"}
)


@dataclass(slots=True)
class ExtractedFact:
    text: str
    subject: str
    predicate: str
    object: str
    entities: list[str]
    turn_idx: int
    valid_from: int
    valid_to: int
    conf: float = LLM_CONF
    extractor: str = "llm"

    @property
    def triple(self) -> str:
        """Within-session dedupe key.

        Deliberately not called ``key``: the key a fact is *stored* under is
        ``resolve.canonical_key``, which folds diacritics and articles that this
        stage has no business deciding about. Extraction only needs to know when
        it has said the same thing twice in one session.
        """
        triple = f"{self.subject}|{self.predicate}|{self.object}"
        return triple if triple.strip("|") else f"|text|{_norm(self.text)}"


@dataclass(slots=True)
class _Window:
    turns: list[Turn]
    claim: list[Turn] = field(default_factory=list)

    @property
    def claimable(self) -> set[int]:
        return {t.idx for t in self.claim}


# ---- public --------------------------------------------------------------- #


def extract_session(
    turns: list[Turn],
    *,
    llm: LLM | None = None,
    settings: config.Settings | None = None,
    window: int = 6,
    overlap: int = 2,
    principal: str = "user",
) -> list[ExtractedFact]:
    """Facts asserted by one session, deduplicated and ordered by turn."""
    return extract_corpus(
        [turns], llm=llm, settings=settings, window=window, overlap=overlap, principal=principal
    )[0]


def extract_corpus(
    sessions: list[list[Turn]],
    *,
    llm: LLM | None = None,
    settings: config.Settings | None = None,
    window: int = 6,
    overlap: int = 2,
    principal: str = "user",
) -> list[list[ExtractedFact]]:
    """Facts for every session, one list per session, in the order given.

    Windows from every session go out as a single batch so concurrency is shared
    across the corpus rather than rationed per session, which matters because a
    corpus is usually many short sessions rather than one long one.
    """
    cfg = settings or getattr(llm, "settings", None) or config.settings()
    plans: list[tuple[int, _Window]] = [
        (index, win)
        for index, turns in enumerate(sessions)
        for win in _windows(turns, window, overlap)
    ]

    replies: list[dict[str, Any] | None] = [None] * len(plans)
    if llm is not None and plans:
        batches = [
            prompts.build_extract_messages(win.turns, win.claimable, principal=principal)
            for _, win in plans
        ]
        replies = llm.batch_json(batches, model=cfg.extract_model, max_tokens=EXTRACT_MAX_TOKENS)

    out: list[list[ExtractedFact]] = [[] for _ in sessions]
    for (index, win), reply in zip(plans, replies):
        if isinstance(reply, dict):
            # a model that legitimately found nothing is not a failed window
            out[index].extend(_parse(reply, win, principal))
        else:
            out[index].extend(extract_rules(win.claim, principal=principal))
    return [_dedupe(facts) for facts in out]


def extract_rules(turns: Iterable[Turn], *, principal: str = "user") -> list[ExtractedFact]:
    """First-person facts, no model involved.

    Only the principal's own turns are mined. An assistant paraphrase would be
    recorded as if the principal had said it, and quoted external material is
    exactly what must never be read as an assertion, so both are left alone: this
    extractor exists to keep the pipeline honest offline, not to cover the corpus.
    """
    facts: list[ExtractedFact] = []
    for turn in turns:
        if (turn.role or "").strip().lower() not in OWNER_ROLES:
            continue
        found = 0
        for sentence in _sentences(turn.text):
            if found >= MAX_RULE_FACTS:
                break
            fact = _rule_fact(sentence, turn, principal)
            if fact is not None:
                facts.append(fact)
                found += 1
    return facts


# ---- windows -------------------------------------------------------------- #


def _windows(turns: Sequence[Turn], window: int, overlap: int) -> list[_Window]:
    """Slide over the session so every turn is claimable in exactly one window."""
    ordered = list(turns)
    if not ordered:
        return []
    size = max(1, int(window))
    context = min(max(0, int(overlap)), size - 1)
    step = size - context
    out: list[_Window] = []
    start = 0
    while start < len(ordered):
        slice_ = ordered[start : start + size]
        claim = slice_ if start == 0 else slice_[context:]
        if claim:
            out.append(_Window(turns=slice_, claim=list(claim)))
        start += step
    return out


# ---- model replies -------------------------------------------------------- #


def canonical_predicate(raw: str) -> str:
    """Fold a predicate onto the schema's vocabulary where it belongs there.

    Three passes, cheapest first: snake-case it, look it up in the synonym table,
    and - only if that found nothing - try it without a leading ``has_``/``is_``.
    A predicate the vocabulary does not know survives unchanged rather than being
    forced onto the nearest slot; an unknown predicate is merely unqueryable,
    while a wrong one makes memory supersede a fact that is still true.
    """
    predicate = _snake(raw)
    if not predicate:
        return ""
    predicate = PREDICATE_SYNONYMS.get(predicate, predicate)
    if predicate in PREDICATES:
        return predicate
    for prefix in ("has_", "is_", "have_"):
        if predicate.startswith(prefix):
            stem = predicate[len(prefix) :]
            stem = PREDICATE_SYNONYMS.get(stem, stem)
            if stem in PREDICATES:
                return stem
    return predicate


def _has_ended(predicate: str) -> bool:
    return predicate.startswith(_ENDED_PREFIXES) or predicate.endswith(_ENDED_SUFFIXES)


def _principal_subject(subject: str, principal: str) -> str:
    """One key for the principal, whatever the message called them."""
    key = _norm(principal)
    if not subject or not key:
        return subject
    if subject == key or subject in _SELF_SUBJECTS:
        return key
    # "nora salgado" is the same person as "nora"; "nora's manager" is not
    words = subject.split()
    if len(words) <= 4 and key in words:
        return key
    return subject


def _untrusted(turn: Turn) -> bool:
    """Whether the turn is quoted material rather than a party to the conversation."""
    return bool(turn.origin) or int(turn.tier) <= int(Tier.TOOL)


def _source_key(turn: Turn) -> str:
    return _norm(turn.origin) or _norm(turn.role) or "external source"


def _parse(reply: dict[str, Any], win: _Window, principal: str) -> list[ExtractedFact]:
    items = reply.get("facts")
    if not isinstance(items, list):
        # `LLM.json` wraps a bare array under `items`
        items = reply.get("items")
    if not isinstance(items, list):
        log.debug("extraction reply carried no fact list: %s", sorted(reply)[:8])
        return []

    by_index = {t.idx: t for t in win.claim}
    facts: list[ExtractedFact] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fact = _fact_from(item, by_index, principal)
        if fact is not None:
            facts.append(fact)
    return facts


def _fact_from(
    item: dict[str, Any],
    claimable: dict[int, Turn],
    principal: str,
) -> ExtractedFact | None:
    # only the documented fields are read; a reply that invents keys - including
    # one lifted out of an injected instruction - contributes nothing through them
    fields = {k: v for k, v in item.items() if k in prompts.EXTRACT_FIELDS}

    text = _text(fields.get("text"))
    if not text:
        return None

    turn = claimable.get(_int(fields.get("turn"), -1))
    if turn is None:
        # provenance is the product; a claim we cannot pin to a turn is not a fact
        log.debug("dropping fact attributed outside the window: %r", text[:60])
        return None

    predicate = canonical_predicate(_text(fields.get("predicate")))
    if _has_ended(predicate):
        log.debug("dropping ended-value predicate %r: %r", predicate, text[:60])
        return None

    subject = _principal_subject(_norm(_text(fields.get("subject"))), principal)
    obj = _norm(_text(fields.get("object")))
    if _untrusted(turn):
        # one shape for "an untrusted source claimed X", whatever the reply said:
        # the source is the subject and the claim is the object, so nothing lifted
        # out of a document can be shelved next to what the principal said
        subject = _source_key(turn)
        predicate = "asserts"
        obj = _norm(text)
        if subject not in text.lower():
            text = f"{subject} asserts: {text}"

    return ExtractedFact(
        text=text,
        subject=subject,
        predicate=predicate,
        object=obj,
        entities=_entities(fields.get("entities"), subject, principal),
        turn_idx=turn.idx,
        valid_from=_epoch(fields.get("valid_from"), turn.ts),
        valid_to=_epoch(fields.get("valid_to"), OPEN_INTERVAL),
        conf=_conf(fields.get("conf"), LLM_CONF),
        extractor="llm",
    )


def _entities(raw: Any, subject: str, principal: str) -> list[str]:
    out: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            norm = _norm(_text(item))
            if norm and norm not in out:
                out.append(norm)
    # the subject is what retrieval seeds on, so it has to be reachable through
    # MENTIONS - except when it is the principal, who appears in every fact and
    # so seeds nothing
    skip = _PRONOUN_SUBJECTS | {_norm(principal)}
    if subject and subject not in out and subject not in skip:
        out.insert(0, subject)
    return out[:MAX_ENTITIES]


# ---- rules ---------------------------------------------------------------- #

_PREF_VERBS = "like|love|prefer|hate|enjoy|use|need|own|have|avoid|drink|play|drive|read|watch"
_MOVE_VERBS = "live|work|study|train|volunteer|stay"
_MOVE_PREPS = "in|at|for|on|with"
_DID_VERBS = "signed up for|picked up|moved to|switched to|joined|started"
_HEDGES = r"(?:really\s+|usually\s+|always\s+|currently\s+|still\s+|now\s+|only\s+)?"

_PAT_MY = re.compile(r"^my\s+([a-z][a-z '\-]{0,28}?)\s+is\s+(.+)$", re.I)
_PAT_MOVE = re.compile(rf"^i\s+{_HEDGES}({_MOVE_VERBS})\s+({_MOVE_PREPS})\s+(.+)$", re.I)
_PAT_DID = re.compile(rf"^i(?:['’]ve|\s+have)?\s+({_DID_VERBS})\s+(.+)$", re.I)
_PAT_AT = re.compile(r"^i(?:'m|\s+am)\s+(in|at|from)\s+(.+)$", re.I)
_PAT_ING = re.compile(r"^i(?:'m|\s+am)\s+(\w+ing)\s+(.+)$", re.I)
_PAT_PREF = re.compile(rf"^i\s+{_HEDGES}({_PREF_VERBS})\s+(.+)$", re.I)
_PAT_ALLERGIC = re.compile(r"^i(?:'m|\s+am)\s+allergic\s+to\s+(.+)$", re.I)
_PAT_IS = re.compile(r"^i(?:'m|\s+am)\s+(.+)$", re.I)

_LEAD = re.compile(r"^(?:and|but|so|also|plus|then|ok|okay|yes|no|well|oh)[,\s]+", re.I)
_CLAUSE = re.compile(
    r"\s+[-–—]\s+|;\s+"
    r"|,\s+(?=(?:i|it|he|she|they|we|you|which|so|and|but|then|not|the|a|an)\b)",
    re.I,
)
_TRAIL = re.compile(r"\s+(?:now|today|as well|too|any\s?more|these days|from now on)$", re.I)
_CAP = re.compile(r"\b([A-Z][\w&.'’-]*(?:\s+[A-Z][\w&.'’-]*)*)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_ANCHOR = re.compile(r"\b(?:i|my)\b", re.I)
#: words in front of the pronoun that make the claim hypothetical or reported
_SUBORD = re.compile(
    r"\b(?:when|if|whether|unless|because|that|ask|asks|asked|tell|tells|told"
    r"|think|thinks|wonder|maybe|might|would|could|should)\b",
    re.I,
)
#: how far in front of the pronoun an adverbial phrase may run
LEAD_WINDOW = 40
_IRREGULAR = {"have": "has", "do": "does", "go": "goes", "be": "is"}
#: relation nouns that name a slot, so "my sister Iris" lands on `sibling`
_RELATIONS = {
    "sister": "sibling",
    "brother": "sibling",
    "sibling": "sibling",
    "mother": "parent",
    "father": "parent",
    "mum": "parent",
    "dad": "parent",
    "son": "child",
    "daughter": "child",
    "wife": "partner",
    "husband": "partner",
    "partner": "partner",
    "manager": "manager",
    "boss": "manager",
}
_PRONOUN_SUBJECTS = {"user", "i", "me", "my", "we"}
_STOPCAPS = {"i", "my", "the", "a", "an", "and", "but", "so", "also", "ok", "okay", "yes", "no"}


def _rule_fact(sentence: str, turn: Turn, principal: str) -> ExtractedFact | None:
    line = _LEAD.sub("", sentence.strip()).strip()
    if not line or line.endswith("?"):
        return None
    fact = _match(line, turn, principal)
    if fact is not None:
        return fact
    # a claim often trails a short adverbial - "As of the first of April I'm
    # design lead" - so retry from the pronoun, but only when the words in front
    # of it are not a subordinating clause that would make the claim reported or
    # hypothetical rather than asserted
    anchor = _ANCHOR.search(line)
    if anchor and 0 < anchor.start() <= LEAD_WINDOW and not _SUBORD.search(line[: anchor.start()]):
        return _match(line[anchor.start() :], turn, principal)
    return None


def _match(line: str, turn: Turn, principal: str) -> ExtractedFact | None:
    match = _PAT_MY.match(line)
    if match:
        noun, value = _clip(match.group(1)), _clip(match.group(2))
        words = noun.split()
        relation = _RELATIONS.get(words[0].lower()) if words else None
        if relation and len(words) > 1:
            # "my sister Iris is up in Porto" - the durable claim is who the
            # sister is; where she is that week is a fact about her, not about
            # the principal, and this extractor has no way to attribute it
            name = " ".join(words[1:])
            return _make(
                f"{principal.capitalize()}'s {words[0].lower()} is {name}.",
                principal,
                relation,
                name,
                line,
                turn,
            )
        if not _usable(value):
            return None
        return _make(
            f"{principal.capitalize()}'s {noun} is {value}.",
            principal,
            f"has_{_snake(noun)}",
            value,
            line,
            turn,
        )

    match = _PAT_MOVE.match(line)
    if match:
        verb, prep, value = match.group(1).lower(), match.group(2).lower(), _clip(match.group(3))
        if not _usable(value):
            return None
        return _make(
            f"{principal.capitalize()} {_third(verb)} {prep} {value}.",
            principal,
            f"{_third(verb)}_{prep}",
            value,
            line,
            turn,
        )

    match = _PAT_DID.match(line)
    if match:
        verb, value = match.group(1).lower(), _clip(match.group(2))
        if not _usable(value):
            return None
        return _make(
            f"{principal.capitalize()} {verb} {value}.",
            principal,
            verb,
            value,
            line,
            turn,
        )

    match = _PAT_AT.match(line)
    if match:
        prep, value = match.group(1).lower(), _clip(match.group(2))
        if not _usable(value):
            return None
        return _make(
            f"{principal.capitalize()} is {prep} {value}.",
            principal,
            # "I'm in Lisbon" is where they live; "at" and "from" are not that
            "lives_in" if prep == "in" else f"is_{prep}",
            value,
            line,
            turn,
        )

    match = _PAT_ING.match(line)
    if match:
        verb, value = match.group(1).lower(), _clip(match.group(2))
        if not _usable(value):
            return None
        return _make(
            f"{principal.capitalize()} is {verb} {value}.",
            principal,
            f"is_{verb}",
            value,
            line,
            turn,
        )

    match = _PAT_PREF.match(line)
    if match:
        verb, value = match.group(1).lower(), _clip(match.group(2))
        if not _usable(value):
            return None
        return _make(
            f"{principal.capitalize()} {_third(verb)} {value}.",
            principal,
            verb,
            value,
            line,
            turn,
        )

    match = _PAT_ALLERGIC.match(line)
    if match:
        value = _clip(match.group(1))
        if not _usable(value):
            return None
        return _make(
            f"{principal.capitalize()} is allergic to {value}.",
            principal,
            "allergy",
            value,
            line,
            turn,
        )

    match = _PAT_IS.match(line)
    if match:
        value = _clip(match.group(1))
        if not _usable(value):
            return None
        return _make(f"{principal.capitalize()} is {value}.", principal, "is", value, line, turn)

    return None


def _make(
    text: str,
    subject: str,
    predicate: str,
    obj: str,
    source: str,
    turn: Turn,
) -> ExtractedFact:
    return ExtractedFact(
        text=text,
        subject=_norm(subject),
        predicate=canonical_predicate(predicate),
        object=_norm(obj),
        entities=_entities(_capitalised(source), _norm(subject), subject),
        turn_idx=turn.idx,
        valid_from=turn.ts,
        valid_to=OPEN_INTERVAL,
        conf=RULES_CONF,
        extractor="rules",
    )


def _capitalised(sentence: str) -> list[str]:
    """Proper-noun-ish spans, minus the grammatically capitalised first word."""
    out: list[str] = []
    for match in _CAP.finditer(sentence):
        if match.start() == 0:
            continue
        span = match.group(1).strip(".'’")
        if span.lower() in _STOPCAPS or len(span) < 2:
            continue
        out.append(span)
    return out


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(text or "") if part.strip()]


def _clip(value: str) -> str:
    """Drop a trailing independent clause, so the object stays the claim."""
    head = _CLAUSE.split(value.strip(), maxsplit=1)[0].strip().strip(".,!’'\" ")
    left, sep, right = head.partition(", ")
    # a comma joining two substantial phrases is joining two clauses; a short one
    # ("Lisbon, Alfama", "flat white, no sugar") is part of the same value
    tail = right.split()
    if sep and (
        (len(left.split()) >= 3 and len(tail) >= 3)
        or (len(tail) == 1 and right.lower().endswith("ly"))  # "shellfish, badly"
    ):
        head = left.strip()
    for _ in range(2):  # "the MacBook Pro now" and "a sesame allergy as well"
        head = _TRAIL.sub("", head).strip()
    return head


def _usable(value: str) -> bool:
    return 2 <= len(value) <= 90


def _third(verb: str) -> str:
    if verb in _IRREGULAR:
        return _IRREGULAR[verb]
    if verb.endswith(("s", "x", "z", "ch", "sh")):
        return verb + "es"
    if verb.endswith("y") and verb[-2:-1] not in ("a", "e", "i", "o", "u"):
        return verb[:-1] + "ies"
    return verb + "s"


# ---- normalisation -------------------------------------------------------- #

_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z0-9]+")
_OPEN_WORDS = {"", "none", "null", "present", "now", "ongoing", "open", "still true", "-"}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return _WS.sub(" ", value).strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _norm(value: str) -> str:
    return _WS.sub(" ", value or "").strip().strip(".,;:!\"'").lower()


def _snake(value: str) -> str:
    return _NONWORD.sub("_", (value or "").lower()).strip("_")


def _int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("#").isdigit():
        return int(value.strip().lstrip("#"))
    return default


def _conf(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, conf))


def _epoch(value: Any, default: int) -> int:
    """A stated time as unix seconds, falling back to the turn's own clock.

    Anything the model leaves open stays open: ``valid_to`` defaults to the
    sentinel rather than to a guess, because a wrong end date silently retires a
    fact that is still true.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else default
    if not isinstance(value, str):
        return default

    raw = value.strip()
    if raw.lower() in _OPEN_WORDS:
        return default
    if raw.isdigit() and len(raw) > 4:
        return int(raw)

    text = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    if re.fullmatch(r"\d{4}", text):
        text += "-01-01"
    elif re.fullmatch(r"\d{4}-\d{2}", text):
        text += "-01"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        log.debug("unparseable timestamp %r", raw)
        return default
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp())


# ---- dedupe --------------------------------------------------------------- #


def _dedupe(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """One fact per triple, attributed to the turn that said it first."""
    ordered = sorted(facts, key=lambda f: f.turn_idx)
    seen: dict[str, ExtractedFact] = {}
    for fact in ordered:
        seen.setdefault(fact.triple, fact)
    return sorted(seen.values(), key=lambda f: f.turn_idx)
