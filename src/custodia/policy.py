"""Trust admission: the boundary a claim has to cross to become memory.

The whole point of Custodia is that an answer is warranted by evidence, so the
interesting question is not "what did the model say" but "what was allowed to
become evidence in the first place". That decision is made here, at *write*
time, before anything is retrievable, and it is made from the channel a claim
arrived on rather than from what the claim says about itself.

Three properties are deliberate:

* **It fails closed.** A rule that fires quarantines the write. ``strict=False``
  exists so a research run can measure how often the rules trip without changing
  what is stored, but the default -- the one every surface constructs -- refuses.
* **It keeps the refusal.** A quarantined fact is still written to the graph
  with ``status = quarantined`` and the reason that put it there, and the
  refusal itself becomes a ``Rejection`` node. Dropping the attack would destroy
  exactly the audit trail the product exists to produce.
* **It is data, not code.** Every rule is a :class:`Rule` with an id, a compiled
  pattern or a structural predicate, and a reason template, so the CLI can print
  the ruleset and the UI can name the rule that fired.

The rules are tuned against a corpus of ordinary conversation as much as against
attacks. Over-blocking is a measured failure mode: "I forgot my keys", "ignore
the noise" and "my new job" have to pass, and there are tests that say so.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from custodia.schema import ACTIVE, QUARANTINED, Fact, Tier

# --------------------------------------------------------------------------- #
# text normalisation
# --------------------------------------------------------------------------- #

#: characters an attacker sprinkles through a payload to break naive matching:
#: zero-width joiners, soft hyphens, bidi overrides -- all unicode category Cf.
_TYPOGRAPHY = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "′": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
        "　": " ",
        "ʼ": "'",
        "`": "'",
    }
)

_HORIZONTAL_WS = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{2,}")


def normalize_text(text: str) -> str:
    """Fold a payload to the form the rules are written against.

    Compatibility-normalises (so fullwidth and ligature spellings collapse),
    drops format characters, unifies smart punctuation and squeezes horizontal
    whitespace. Newlines survive, because several rules anchor on a line start:
    ``SYSTEM:`` at the head of a line inside a scraped web page is the single
    most common injection framing we see.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", str(text))
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")
    folded = folded.translate(_TYPOGRAPHY)
    folded = _HORIZONTAL_WS.sub(" ", folded)
    return _BLANK_LINES.sub("\n", folded).strip()


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Rule:
    """One admission rule, declared rather than implemented.

    ``pattern`` is set for content rules -- the ones :meth:`Policy.screen`
    evaluates against raw text. Structural rules (tier floor, identity forgery)
    carry no pattern because they are decided from the fact's tier and shape,
    not from its wording; they are listed here so that the full ruleset is one
    printable object.
    """

    id: str
    kind: str          # "content" | "structural"
    summary: str       # what the rule defends against, for `custodia policy --rules`
    reason: str        # one-sentence template, formatted with the match context
    pattern: re.Pattern[str] | None = None

    def render(self, **ctx: Any) -> str:
        try:
            return self.reason.format(**ctx)
        except (KeyError, IndexError, ValueError):
            return self.reason

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "summary": self.summary,
            "reason": self.reason,
            "pattern": self.pattern.pattern.strip() if self.pattern else "",
        }


_FLAGS = re.IGNORECASE | re.VERBOSE | re.MULTILINE | re.UNICODE

# "forget" is the single highest false-positive risk in the set, so both of its
# alternatives are guarded against the negated forms people actually write
# ("don't forget", "do not forget"). Both lookbehinds are fixed width.
_SELF_ELEVATION = re.compile(
    r"""
      \b as \s+ (?:the|your) \s+ (?:system\s+)?
          (?:administrator|admin|owner|principal|root|superuser|operator) \b
          (?! \s+ of \s+ (?:a|an|my|his|her|their|our) \b )
    | \b i \s+ am \s+ (?:the|your) \s+ (?:system\s+)?
          (?:owner|administrator|admin|principal|operator|root|superuser) \b
          (?! \s+ of \s+ (?:a|an|my|his|her|their|our) \b )
    | \b (?:trust|authority|clearance|privilege|permission) [\s_-]*
          (?:level|tier|rank)? \s* (?: [:=] | \b is \b ) \s*
          (?:owner|admin|administrator|system|root|high|highest|max|maximum|trusted|full|3) \b
    | \b (?:tier|priority) \s* [:=] \s*
          (?:owner|admin|administrator|system|high|max|maximum) \b
    | \b (?:this|that|the\s+following) \s+ is \s+ an? \s+
          (?:verified|trusted|confirmed|authoritative|authenticated|official|system) \s+
          (?:fact|statement|message|memory|instruction|note|update) \b
    | \b verified \s+ (?:fact|statement|information|message|note) \s+ from \s+ the \s+
          (?:user|owner|principal|system|administrator|admin) \b
    | \b (?:speaking|writing|acting) \s+ (?:as|on \s+ behalf \s+ of) \s+ the \s+
          (?:owner|user|principal|administrator|admin|system) \b
    | \b (?:message|instruction|note|memo|directive) \s+ from \s+ the \s+
          (?:owner|principal|administrator|admin|system \s+ administrator) \b
    | \b (?:elevate|escalate|raise|upgrade|grant) \s+ (?:my|this|the) \s+
          (?:trust|privilege|privileges|permission|permissions|authority|tier|clearance|access) \b
    | \b mark \s+ (?:this|the \s+ following) \s+ as \s+
          (?:owner|trusted|verified|authoritative) \b
    | \b treat \s+ (?:this|me) \s+ as \s+ (?:the \s+)?
          (?:owner|principal|trusted|verified|authoritative) \b
    """,
    _FLAGS,
)

_INSTRUCTION_INJECTION = re.compile(
    r"""
      \b (?:ignore|disregard|discard|override|overwrite) \b [^.\n]{0,40}?
          \b (?:previous|prior|earlier|above|preceding|all|any|the) \b [^.\n]{0,24}?
          \b (?:instruction|instructions|prompt|prompts|rule|rules|direction|directions
               |guideline|guidelines|command|commands|context|memory|memories|fact|facts) \b
    | (?<! n't\ ) (?<! not\ ) \b forget \s+
          (?: everything | all \b (?:\s+ of)? (?:\s+ (?:that|this|the \s+ above))? ) \b
    | (?<! n't\ ) (?<! not\ ) \b forget \s+ (?:that|what|whatever) \s+ you \s+
          (?: know|knew|learned|learnt|remember|were \s+ told|have \s+ stored ) \b
    | \b from \s+ now \s+ on \b [^.\n]{0,40}?
          \b (?:always|never|only|remember|treat|respond|reply|answer|ignore|forget
               |you \s+ (?:must|will|shall|should|are)) \b
    | \b your \s+ new \s+
          (?:instruction|instructions|directive|directives|rule|rules|prompt
            |system \s+ prompt|role|persona|task) \b
    | \b (?:new \s+)? system \s+ prompt \s+ (?: is \b | : )
    | ^ \s* (?:system|assistant|user) \s* :
    | \b system \s* : \s*
          (?:you|ignore|forget|new|always|never|override|update|remember|the \s+ user) \b
    | < \s* /? \s* system \s* >
    | \[ \s* /? \s* INST \s* \]
    | < \| \s* im_ (?:start|end) \s* \| >
    | \b update \s+ your \s+
          (?:memory|memories|record|records|note|notes|knowledge|belief|beliefs|stored \s+ \w+) \b
    | \b (?:add|write|save|store|commit) \s+ (?:this|these|the \s+ following) \s+
          (?:to|into) \s+ your \s+ (?:memory|memories|notes|records) \b
    | \b disregard \s+ (?:all|any|everything|previous|prior|earlier|the \s+ above) \b
    """,
    _FLAGS,
)

_EXFILTRATION = re.compile(
    r"""
      \b (?:send|email|e-?mail|post|upload|forward|transmit|exfiltrate|leak|publish|share|sync) \b
          [^.\n]{0,40}? \b your \s+ (?:entire \s+)?
          (?:memory|memories|facts|notes|records|context|knowledge|history|stored \s+ \w+) \b
    | \b (?:reveal|disclose|dump|output|print|repeat|list|enumerate|export) \b [^.\n]{0,40}?
          (?: \b all \s+ (?:of \s+)? (?:the \s+|your \s+)? (?:stored \s+)?
                 (?:facts?|memories|memory|records?|data|secrets?|credentials?|notes?) \b
            | \b the \s+ stored \s+
                 (?:facts?|memories|memory|data|records?|information|credentials?|secrets?|notes?) \b
            | \b your \s+ (?:entire \s+)?
                 (?:memory|memories|context|system \s+ prompt|instructions) \b )
    | \b (?:send|post|upload|forward|transmit|report) \b [^.\n]{0,50}?
          \b (?:memory|memories|facts?|records?|notes?|data|credentials?|secrets?|history) \b
          [^.\n]{0,40}? https?://
    | \b (?:send|email|e-?mail|forward) \b [^.\n]{0,40}?
          \b (?:memory|memories|facts?|records?|notes?|data|credentials?|secrets?|history) \b
          [^.\n]{0,40}? [\w.+-]+ @ [\w-]+ \. [\w.-]+
    """,
    _FLAGS,
)

RULE_SELF_ELEVATION = Rule(
    id="self-elevation",
    kind="content",
    summary="content that asserts its own authority or trust level",
    reason=(
        "Content claims an authority its channel does not carry "
        '(matched "{match}"), so it is recorded but never warrantable.'
    ),
    pattern=_SELF_ELEVATION,
)

RULE_INSTRUCTION_INJECTION = Rule(
    id="instruction-injection",
    kind="content",
    summary="imperative attempts to rewrite the agent's memory or behaviour",
    reason=(
        "Content tries to rewrite the agent's instructions or stored memory "
        '(matched "{match}"), which no remembered fact ever needs to do.'
    ),
    pattern=_INSTRUCTION_INJECTION,
)

RULE_EXFILTRATION = Rule(
    id="exfiltration",
    kind="content",
    summary="instructions to send, reveal or dump what memory holds",
    reason=(
        "Content instructs an action on stored memory rather than stating a fact "
        '(matched "{match}").'
    ),
    pattern=_EXFILTRATION,
)

RULE_IDENTITY_FORGERY = Rule(
    id="identity-forgery",
    kind="structural",
    summary="a low-tier claim about the principal's identity, preferences or credentials",
    reason=(
        "A {tier}-tier source cannot state the principal's {predicate}; "
        "only the principal can say who they are."
    ),
)

RULE_TIER_FLOOR = Rule(
    id="tier-floor",
    kind="structural",
    summary="a write may only act on facts at or below its own trust tier",
    reason=(
        "A {tier}-tier claim may not {op} an {target_tier}-tier fact; "
        "the acting tier has to outrank what the write touches."
    ),
)

#: evaluated in this order: specific content diagnoses first, then the two
#: structural rules, with the tier floor last because it is the general backstop
#: that catches whatever the content rules did not recognise.
RULES: tuple[Rule, ...] = (
    RULE_SELF_ELEVATION,
    RULE_INSTRUCTION_INJECTION,
    RULE_EXFILTRATION,
    RULE_IDENTITY_FORGERY,
    RULE_TIER_FLOOR,
)

CONTENT_RULES: tuple[Rule, ...] = tuple(r for r in RULES if r.kind == "content")

#: operations that act on an existing fact and therefore face the tier floor
TARGETED_OPS = frozenset({"supersede", "retract", "contradict"})

OPS = ("assert",) + tuple(sorted(TARGETED_OPS))


def describe_rules() -> list[dict[str, Any]]:
    """The ruleset as printable data -- what ``custodia policy --rules`` shows."""
    return [rule.as_dict() for rule in RULES]


# --------------------------------------------------------------------------- #
# identity predicates
# --------------------------------------------------------------------------- #

#: ways an extractor refers to the principal itself
PRINCIPAL_ALIASES: frozenset[str] = frozenset(
    {
        "i",
        "me",
        "my",
        "myself",
        "user",
        "the user",
        "owner",
        "the owner",
        "principal",
        "the principal",
        "account holder",
        "the account holder",
    }
)

#: predicates that say who someone *is*, what they prefer, or how to be them.
#: Anything an unrelated third party has no standing to assert about you.
IDENTITY_PREDICATES: frozenset[str] = frozenset(
    {
        "name", "full_name", "first_name", "last_name", "nickname", "username",
        "handle", "alias", "identity", "is_a", "role", "title", "job_title",
        "occupation", "employer", "works_at", "works_for", "lives_in", "home",
        "home_address", "address", "email", "email_address", "phone",
        "phone_number", "age", "birthday", "birth_date", "date_of_birth", "dob",
        "gender", "pronouns", "nationality", "citizenship", "religion",
        "prefers", "preference", "preferences", "likes", "dislikes", "wants",
        "password", "passcode", "passphrase", "pin", "api_key", "apikey",
        "access_token", "token", "secret", "credential", "credentials",
        "account", "account_number", "bank", "bank_account", "card",
        "card_number", "wallet", "wallet_address", "ssn", "passport",
        "license", "licence", "login", "security_question", "recovery_email",
    }
)

_IDENTITY_PREFIX = re.compile(
    r"^(?:preferred|favorite|favourite|primary|current|default|main|usual|home)_\w+$",
    re.IGNORECASE,
)
_CREDENTIAL_HINT = re.compile(
    r"(?:password|passcode|passphrase|\bpin\b|api_?key|access_token|secret|credential"
    r"|otp|2fa|mfa|seed_phrase|private_key|recovery)",
    re.IGNORECASE,
)
_PREDICATE_PUNCT = re.compile(r"[^0-9a-z]+")


def _predicate_key(predicate: str) -> str:
    folded = unicodedata.normalize("NFKD", str(predicate or "")).casefold()
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return _PREDICATE_PUNCT.sub("_", folded).strip("_")


def is_identity_predicate(predicate: str) -> bool:
    """True for predicates that state identity, preference or a credential."""
    key = _predicate_key(predicate)
    if not key:
        return False
    if key in IDENTITY_PREDICATES:
        return True
    if _IDENTITY_PREFIX.match(key):
        return True
    return bool(_CREDENTIAL_HINT.search(key))


# --------------------------------------------------------------------------- #
# decisions
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Decision:
    """The verdict on one write, in the form the writer and the UI both need."""

    admitted: bool          # False => do not let this fact into the warrantable set
    status: str             # schema.ACTIVE | schema.QUARANTINED
    rule: str               # short rule id, e.g. "tier-floor"
    reason: str             # one human sentence, shown in the UI and the audit trail

    @property
    def flagged(self) -> bool:
        """A rule fired, whether or not this run was strict enough to refuse."""
        return bool(self.rule)

    @classmethod
    def allow(cls) -> "Decision":
        return cls(admitted=True, status=ACTIVE, rule="", reason="")


@dataclass(slots=True)
class RejectionRecord:
    """A refused write, kept so the audit trail shows attempts, not just survivors."""

    rule: str
    reason: str
    text: str
    tier: str
    turn_id: int                  # vertex id of the turn the write came from
    target_fact_id: int | None    # vertex id of the fact it tried to act on
    ts: int

    @property
    def props(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "reason": self.reason,
            "text": self.text,
            "tier": self.tier,
            "ts": self.ts,
        }


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #


class Policy:
    """Decides what may become memory.

    ``strict`` is the only knob, and it only ever loosens: with ``strict=False``
    a rule still fires and still names itself, but the write is admitted and
    stays warrantable. That mode exists to measure the rules against a corpus
    without changing what the corpus looks like afterwards. Every production
    surface constructs the default.
    """

    __slots__ = ("strict", "principal_aliases", "rules")

    def __init__(
        self,
        *,
        strict: bool = True,
        principal_aliases: Iterable[str] | None = None,
    ) -> None:
        self.strict = strict
        extra = {str(a).strip().casefold() for a in (principal_aliases or ()) if str(a).strip()}
        self.principal_aliases = frozenset(PRINCIPAL_ALIASES | extra)
        self.rules = RULES

    # ---------------------------------------------------------------- content

    def screen(self, text: str) -> tuple[str, str] | None:
        """Match text against the content rules; ``(rule_id, reason)`` on a hit.

        Used on fact text and, at ingest time, on the text of the turn a fact
        was derived from -- an injection is usually in the carrier, not in the
        tidy triple an extractor lifted out of it.
        """
        folded = normalize_text(text)
        if not folded:
            return None
        for rule in CONTENT_RULES:
            if rule.pattern is None:
                continue
            hit = rule.pattern.search(folded)
            if hit:
                return rule.id, rule.render(match=_snippet(hit.group(0)))
        return None

    # ------------------------------------------------------------- admission

    def admit(
        self,
        fact: Fact,
        *,
        target: Fact | None = None,
        op: str = "assert",
    ) -> Decision:
        """Decide whether ``fact`` may enter the warrantable set.

        ``op`` is one of ``assert``, ``supersede``, ``retract``, ``contradict``.
        The targeted operations additionally face the tier floor, which is the
        invariant the rest of the system is allowed to assume: nothing ever
        rewrites a fact that outranks it.
        """
        operation = (op or "assert").strip().lower()

        screened = self.screen(fact.text)
        if screened is None and fact.object:
            screened = self.screen(fact.object)
        if screened is not None:
            return self._verdict(*screened)

        forged = self._identity_forgery(fact)
        if forged is not None:
            return self._verdict(*forged)

        floored = self._tier_floor(fact, target, operation)
        if floored is not None:
            return self._verdict(*floored)

        return Decision.allow()

    # ----------------------------------------------------------------- helpers

    def rejection(
        self,
        decision: Decision,
        fact: Fact,
        *,
        turn_id: int,
        target_fact_id: int | None = None,
        ts: int = 0,
    ) -> RejectionRecord:
        """Turn a refusal into the record that becomes a ``Rejection`` node."""
        return RejectionRecord(
            rule=decision.rule,
            reason=decision.reason,
            text=fact.text,
            tier=fact.tier.label,
            turn_id=turn_id,
            target_fact_id=target_fact_id,
            ts=ts or fact.ingested_at or fact.valid_from,
        )

    def _verdict(self, rule: str, reason: str) -> Decision:
        if self.strict:
            return Decision(admitted=False, status=QUARANTINED, rule=rule, reason=reason)
        return Decision(admitted=True, status=ACTIVE, rule=rule, reason=reason)

    def _identity_forgery(self, fact: Fact) -> tuple[str, str] | None:
        if fact.tier > Tier.TOOL:
            return None
        subject = str(fact.subject or "").strip().casefold()
        if subject not in self.principal_aliases:
            return None
        if not is_identity_predicate(fact.predicate):
            return None
        return RULE_IDENTITY_FORGERY.id, RULE_IDENTITY_FORGERY.render(
            tier=fact.tier.label,
            predicate=_predicate_key(fact.predicate).replace("_", " ") or "identity",
        )

    def _tier_floor(
        self,
        fact: Fact,
        target: Fact | None,
        operation: str,
    ) -> tuple[str, str] | None:
        if target is None or operation not in TARGETED_OPS:
            return None
        if int(fact.tier) >= int(target.tier):
            return None
        return RULE_TIER_FLOOR.id, RULE_TIER_FLOOR.render(
            tier=fact.tier.label,
            op=operation,
            target_tier=target.tier.label,
        )


def _snippet(text: str, limit: int = 72) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"
