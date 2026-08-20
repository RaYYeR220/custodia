"""Question to warrant: seeding, path expansion, temporal resolution, ranking.

A *warrant* is the only thing the answering model is ever allowed to see. It is a
bounded set of facts, each carrying the chain that reached it and the interval
that justifies its position in time. Everything in this module exists to make
that set both small and defensible.

Three shapes are forced on us by HydraDB and are worth naming, because they look
like odd choices otherwise:

* **Multi-seed traversal goes through ``algo.MSpaths``.** A variable-length
  ``MATCH`` needs a fixed source id inline, and ``WHERE`` has no ``IN``, so there
  is no pattern that starts from a list of entities. The procedure is not a
  workaround: it returns whole paths, so an evidence *chain* -- fact, the turn it
  came from, the session that turn sat in -- arrives in one round trip instead of
  being reassembled from endpoint rows.
* **``MSpaths`` seeds on a property value, not on a corpus.** Two principals who
  both know a `berlin` entity are seeded together, so every path is filtered
  against the corpus client-side before anything is believed. Retrieval that
  crossed a corpus boundary would be a memory leak between principals, which is
  a worse failure than a slow query.
* **Set membership is a per-id statement.** Fetching thirty facts by id is a
  ``UNION`` of thirty single-id arms, batched, because there is no ``IN``.

One more, and it is measured rather than documented anywhere: an equality
predicate written *inline in the pattern* -- ``MATCH (f:Fact {id: $p})`` -- uses
the vertex index, while the same predicate in a ``WHERE`` clause does not. On the
two-hop provenance read the difference is 1080 ms against 5 ms, and it compounds
across a ``UNION`` of arms. Every equality below is therefore in the pattern, and
``WHERE`` is left for the predicates that cannot be (``STARTS WITH``, ranges).

Retrieval never requires a language model. A model, when present, only widens the
seed set; the deterministic noun-phrase heuristic underneath it is what the
pipeline is actually built on, so a warrant is reproducible and an outage
degrades recall rather than breaking retrieval.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from custodia import config, lexical, schema
from custodia.hydra.client import HydraClient, HydraError
from custodia.lexical import LexicalIndex
from custodia.schema import Evidence, Tier

log = logging.getLogger("custodia.retrieve")

# --------------------------------------------------------------------------- #
# tuning
# --------------------------------------------------------------------------- #

#: how many candidate terms are matched against `Entity.norm` in one UNION
SEED_ARMS = 24
#: how many single-id arms go into one batched read
FETCH_ARMS = 32
#: edge reads emit two arms per id, so they take half as many ids per statement
EDGE_ARMS = 16
#: how far a supersession chain is followed past the facts already in hand
SUPERSESSION_ROUNDS = 4
#: more id-arms than this and a whole-corpus read may be the cheaper shape...
SCAN_ARMS = 8
#: ...but only for a corpus small enough that reading all of it stays bounded
SCAN_BUDGET = 5000
#: shortest candidate term allowed to seed a `STARTS WITH` match
PREFIX_MIN = 4
#: longest noun phrase the deterministic seeder will propose
PHRASE_MAX = 3

# Ranking weights. They sum to 1.0, so a score is directly comparable to
# `Settings.evidence_floor` and reads as a percentage in the UI.
#
# Lexical relevance and graph anchoring are separate terms on purpose, and this
# is the one weighting decision worth arguing about. Anchoring alone does not
# discriminate: when the question resolves to a hub entity - `user`, in a corpus
# that is entirely about one person - every fact hangs off it, every fact scores
# the same on that term, and the ordering collapses onto recency and tier. The
# lexical term is what makes "what does the user drink" rank the drink fact
# above the gym fact, so it carries the larger weight of the two.
#
# Tier carries more weight than any other structural term because memory now
# records what the assistant said as well as what the principal said. Those two
# voices restate each other constantly, and when they disagree the principal is
# right by definition. A tier term that only nudges the ordering lets a stale
# assistant restatement sit above the fact that replaced it.
W_LEXICAL = 0.26
W_ANCHOR = 0.10
W_NEIGHBOUR = 0.14
W_PROXIMITY = 0.10
W_TIER = 0.22
W_CORROBORATION = 0.06
W_RECENCY = 0.12

#: corroboration saturates: the fourth independent witness adds nothing
CORROBORATION_CAP = 3

#: anchor strength for a fact that directly mentions a seeded entity, and for
#: one reached only by walking further out from it
ANCHOR_DIRECT = 1.0
ANCHOR_INDIRECT = 0.5

#: A fact the index found was not reached *through* the graph, so it is scored
#: as one hop out rather than as zero: it has no chain, and it should not
#: out-earn a fact whose chain the warrant can actually show.
LEXICAL_HOPS = 1

#: how many of the strongest BM25 hits are used as graph-expansion anchors, and
#: how strong they have to be. Two is deliberate: the anchor's job is to reach the
#: siblings of a fact the question actually matched, and a weak match is not a
#: foothold, it is a guess with a traversal budget attached.
LEXICAL_ANCHORS = 2
LEXICAL_ANCHOR_FLOOR = 0.35

#: fact -> turn -> session -> turn -> fact. Four hops, because the session is the
#: unit of context a person reasons over: "where did I redeem the coupon" is
#: answered by a turn two messages earlier in the same conversation, and nothing
#: shorter reaches it. Entity seeding never gets there either, since the two
#: claims name no entity in common.
ANCHOR_MAX_LEN = 4

#: the anchor walk needs IN_SESSION, which the entity walk deliberately omits:
#: from an entity, session-mates are noise; from a fact the question matched,
#: they are the surrounding conversation.
ANCHOR_RELS = [*schema.RETRIEVAL_RELS, schema.IN_SESSION]

#: facts below this tier never enter a warrant. External content is
#: attacker-influenceable by definition; it may be recorded and it may be shown
#: in an audit, but an answer may not be built on it.
MIN_TIER = Tier.TOOL

_FACT_FIELDS = (
    "f.id AS id, f.corpus AS corpus, f.key AS key, f.text AS text, f.subj AS subj, "
    "f.pred AS pred, f.obj AS obj, f.tier AS tier, f.rank AS rank, f.status AS status, "
    "f.vfrom AS vfrom, f.vto AS vto, f.ing AS ing, f.conf AS conf, f.sid AS sid, "
    "f.sidx AS sidx, f.tidx AS tidx"
)

_EDGE_FIELDS = "RETURN a.id AS newer, b.id AS older"
_PAIR_FIELDS = "RETURN a.id AS src, b.id AS dst"

_PROV_FIELDS = (
    "f.id AS fid, t.text AS turn_text, t.ts AS turn_ts, t.role AS turn_role, t.sid AS turn_sid"
)


def _exact_entity(i: int) -> str:
    return (
        f"MATCH (e:{schema.ENTITY} {{corpus: $corpus, norm: $v{i}}}) RETURN e.norm AS norm"
    )


def _prefix_entity(i: int) -> str:
    return (
        f"MATCH (e:{schema.ENTITY} {{corpus: $corpus}}) "
        f"WHERE e.norm STARTS WITH $v{i} RETURN e.norm AS norm"
    )


@runtime_checkable
class LanguageModel(Protocol):
    """The slice of :class:`custodia.llm.LLM` retrieval and the gate depend on.

    Declared structurally so a test can pass a twenty-line stub, and so this
    module never imports the provider client at module scope.
    """

    @property
    def enabled(self) -> bool: ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = ...,
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> Any: ...

    def json(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = ...,
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> dict[str, Any]: ...


SEED_SYSTEM = """You pull retrieval keys out of a question about someone's stored memory.

Return JSON only:
{"entities": ["..."], "predicates": ["..."]}

entities   - people, places, organisations, products, projects or topics the
             question is about, lowercased, singular, no articles.
predicates - the relation the question asks about, as one or two words
             ("lives in", "works at", "prefers").

Propose what a keyword index would need. Do not answer the question, do not
invent entities the question does not name, and return no other keys."""


# --------------------------------------------------------------------------- #
# the warrant
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Warrant:
    """The bounded evidence set an answer is allowed to be built from."""

    question: str
    asked_at: int
    as_of: int | None
    #: True when the caller told us when the question was asked. When it is False
    #: `asked_at` is just the wall clock at retrieval, which is not something we
    #: should present to the model as "today" - and putting a moving timestamp in
    #: the prompt would also make every answer uncacheable.
    asked_at_known: bool = False
    evidence: list[Evidence] = field(default_factory=list)
    seeds: dict[str, list[str]] = field(default_factory=dict)
    paths_examined: int = 0
    facts_considered: int = 0
    elapsed_ms: float = 0.0
    #: untrusted evidence that was retrieved and refused - surfaced in the UI,
    #: because "we saw the attack and dropped it" is the demonstrable claim
    quarantined_seen: int = 0

    def ids(self) -> set[int]:
        """The only fact ids a citation may name."""
        return {e.fid for e in self.evidence}

    def __len__(self) -> int:
        return len(self.evidence)

    def __bool__(self) -> bool:
        return bool(self.evidence)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "asked_at": self.asked_at,
            "as_of": self.as_of,
            "seeds": dict(self.seeds),
            "evidence": [evidence_dict(e) for e in self.evidence],
            "paths_examined": self.paths_examined,
            "facts_considered": self.facts_considered,
            "quarantined_seen": self.quarantined_seen,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def evidence_dict(item: Evidence) -> dict[str, Any]:
    """The wire form of one piece of evidence.

    Spelled for readers rather than for the dataclass -- `fact_id`, `session`,
    `turn_index` -- because this shape crosses into the HTTP API, the CLI and the
    MCP surface, and those are read by people who have never seen `Evidence`.
    """
    return {
        "fact_id": item.fid,
        "text": item.text,
        "tier": item.tier,
        "status": item.status,
        "valid_from": item.valid_from,
        "valid_to": item.valid_to,
        "session": item.sid,
        "session_index": item.sidx,
        "turn_index": item.tidx,
        "turn_text": item.turn_text,
        "turn_ts": item.turn_ts,
        "score": item.score,
        "hops": item.hops,
        "path": list(item.path),
        "superseded_by": item.superseded_by,
    }


ROUTE_ENTITY = "entity"
ROUTE_LEXICAL = "lexical"
ROUTE_NEIGHBOUR = "neighbour"


@dataclass(slots=True)
class _Candidate:
    """A fact under consideration, with every route that reached it."""

    fid: int
    props: dict[str, Any] = field(default_factory=dict)
    #: distance along the shortest traversal that reached it; -1 if none did
    hops: int = -1
    #: the chain that traversal walked, or the marker for an index-only hit
    path: list[str] = field(default_factory=list)
    #: how strongly the graph anchors this fact to the question's entities
    anchor: float = 0.0
    #: how well the index matched it against this question, 0..1
    lexical: float = 0.0
    #: how strongly a *neighbouring* fact matched, when this one was reached from it
    neighbour: float = 0.0
    routes: set[str] = field(default_factory=set)

    def by_graph(self, *, hops: int, path: list[str], anchor: float) -> None:
        """Record arriving here along a path from an entity seed."""
        if self.hops < 0 or hops < self.hops:
            self.hops = hops
            self.path = path
        self.anchor = max(self.anchor, anchor)
        self.routes.add(ROUTE_ENTITY)

    def by_index(self, score: float) -> None:
        """Record the index matching this fact's own text.

        The chain is only set if nothing else supplied one: a fact found both
        ways keeps its traversal chain, because "matched the index" is a true
        statement about the retrieval and a useless one to show the person
        asking why the fact is in their warrant.
        """
        self.lexical = max(self.lexical, score)
        if not self.path:
            self.path = ["lexical index"]
        self.routes.add(ROUTE_LEXICAL)

    def by_neighbour(self, strength: float) -> None:
        """Record arriving here from a fact the question matched strongly.

        Kept apart from `anchor`, which measures closeness to an entity the
        question named. This measures closeness to a *statement* the question
        matched - the sibling claim lifted from the same turn, which is where the
        answer lives when the question and the answer share no vocabulary.
        """
        self.neighbour = max(self.neighbour, strength)
        self.routes.add(ROUTE_NEIGHBOUR)

    def absorb(self, other: "_Candidate") -> None:
        """Merge a candidate that resolved onto the same fact as this one."""
        if other.hops >= 0 and (self.hops < 0 or other.hops < self.hops):
            self.hops = other.hops
            self.path = other.path
        elif not self.path:
            self.path = other.path
        self.anchor = max(self.anchor, other.anchor)
        self.lexical = max(self.lexical, other.lexical)
        self.routes |= other.routes


# --------------------------------------------------------------------------- #
# the retriever
# --------------------------------------------------------------------------- #


class Retriever:
    """Turns a question into a warrant against one corpus."""

    def __init__(
        self,
        client: HydraClient,
        corpus: str,
        *,
        index: LexicalIndex | None = None,
        llm: LanguageModel | None = None,
        settings: config.Settings | None = None,
        min_tier: Tier = MIN_TIER,
    ) -> None:
        self.client = client
        self.corpus = corpus
        self.settings = settings or config.settings()
        self.index = index
        self.llm = llm
        self.min_tier = Tier.parse(min_tier)
        self._fact_count: int | None = None

    # ------------------------------------------------------------------ seeding

    def seeds(self, question: str) -> dict[str, list[str]]:
        """Entity keys and lexical terms the question resolves to.

        The heuristic runs first and always; a model, if one is configured, only
        adds to what it produced. Order matters because the entity match is
        capped, and a phrase the question actually contains is a better bet than
        one a model inferred - so the model's candidates go on the end, where
        they widen recall without displacing anything.
        """
        terms = self._phrases(question)
        proposed = self._model_terms(question)
        candidates = _dedupe([*terms, *proposed])
        entities = self._match_entities(candidates)
        return {"entities": entities, "terms": candidates[: self.settings.seed_lexical]}

    def _phrases(self, question: str) -> list[str]:
        """Contiguous runs of non-stopword tokens, longest first.

        A crude noun-phrase reader, and deliberately so: `Entity.norm` is a
        normalised surface form, so the thing that matches it is a normalised
        surface form from the question, not a parse tree.
        """
        tokens = lexical.words(question)
        runs: list[list[str]] = [[]]
        for token in tokens:
            if token in lexical.STOPWORDS or len(token) < 2:
                if runs[-1]:
                    runs.append([])
                continue
            runs[-1].append(token)

        phrases: list[str] = []
        singles: list[str] = []
        for run in runs:
            if not run:
                continue
            for size in range(min(PHRASE_MAX, len(run)), 1, -1):
                for start in range(len(run) - size + 1):
                    phrases.append(" ".join(run[start : start + size]))
            singles.extend(run)
        return _dedupe([*phrases, *singles])

    def _model_terms(self, question: str) -> list[str]:
        """Entity and predicate candidates from a model, if one is available.

        Wrapped whole: a provider outage must cost recall, never the query. The
        deterministic phrases below it are what retrieval actually stands on.
        """
        llm = self.llm
        # not `enabled`: that gates live calls only, and a cached seed expansion
        # is still a hit. The try below turns a miss into an empty list.
        if llm is None:
            return []
        try:
            reply = llm.json(
                [
                    {"role": "system", "content": SEED_SYSTEM},
                    {"role": "user", "content": question},
                ],
                model=self.settings.extract_model,
                max_tokens=256,
            )
        except Exception as exc:  # provider error, bad JSON, cache miss
            log.debug("seed model unavailable, falling back to phrases: %s", exc)
            return []
        out: list[str] = []
        for key in ("entities", "predicates"):
            for item in reply.get(key) or []:
                text = str(item).strip().lower()
                if text and len(text) < 64:
                    out.append(text)
        return out

    def _match_entities(self, candidates: Sequence[str]) -> list[str]:
        """Resolve candidate terms to `Entity.norm` values present in the corpus.

        Equality and ``STARTS WITH`` are the only string predicates the engine
        has, and there is no ``IN``, so this is a ``UNION`` with one arm per
        candidate, batched to keep statements a sane size.

        Two passes rather than one, because ``RETURN`` takes a binding property
        or ``count(*)`` and nothing else: there is no literal column to tag an
        arm with, so exact matches and prefix matches are separate statements.
        Exact wins where both hit, which is why the order of the two matters.
        """
        wanted = [c for c in candidates if c]
        if not wanted:
            return []

        limit = max(1, self.settings.seed_entities)
        exact = self._entity_pass(wanted, _exact_entity)
        prefix = self._entity_pass(
            [c for c in wanted if len(c) >= PREFIX_MIN], _prefix_entity
        )
        return _dedupe([*exact, *prefix])[:limit]

    def _entity_pass(self, terms: Sequence[str], arm: Callable[[int], str]) -> list[str]:
        found: list[str] = []
        for chunk in _chunks(terms, SEED_ARMS):
            params: dict[str, Any] = {"corpus": self.corpus}
            arms: list[str] = []
            for i, term in enumerate(chunk):
                params[f"v{i}"] = term
                arms.append(arm(i))
            for row in self.client.run(" UNION ".join(arms), **params):
                norm = str(row.get("norm") or "")
                if norm:
                    found.append(norm)
        return found

    # ---------------------------------------------------------------- expansion

    def warrant(
        self,
        question: str,
        *,
        as_of: int | None = None,
        asked_at: int | None = None,
        k: int | None = None,
    ) -> Warrant:
        """Seed, expand, resolve in time, rank, and cut to a bounded set.

        ``as_of`` and ``asked_at`` are different things and only look alike.
        ``as_of`` rewinds the memory - answer as it stood then. ``asked_at`` only
        says when the asking happened, which is what "three months ago" is
        counted from; it filters nothing.
        """
        started = time.perf_counter()
        asked_at_known = asked_at is not None
        asked_at = int(asked_at if asked_at is not None else time.time())
        size = k if k is not None else self.settings.warrant_size

        seeds = self.seeds(question)
        candidates: dict[int, _Candidate] = {}
        turns: dict[int, dict[str, Any]] = {}

        paths_examined = self._expand(seeds["entities"], candidates, turns)
        anchored = self._lexical(question, candidates)
        paths_examined += self._expand_from_facts(anchored, candidates, turns)
        self._hydrate(candidates)

        facts_considered = len(candidates)
        resolved, quarantined = self._resolve(candidates, as_of)
        ranked = self._rank(resolved)

        keep = [c for c in ranked if c.score >= self.settings.evidence_floor][:size]
        self._attach_provenance(keep, turns)

        return Warrant(
            question=question,
            asked_at=asked_at,
            asked_at_known=asked_at_known,
            as_of=as_of,
            evidence=keep,
            seeds=seeds,
            paths_examined=paths_examined,
            facts_considered=facts_considered,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            quarantined_seen=quarantined,
        )

    def _expand(
        self,
        entities: Sequence[str],
        candidates: dict[int, _Candidate],
        turns: dict[int, dict[str, Any]],
    ) -> int:
        """Walk out from the entity seeds and absorb every fact on every path."""
        if not entities:
            return 0
        try:
            rows = self.client.paths(
                rel_types=schema.RETRIEVAL_RELS,
                source_label=schema.ENTITY,
                # seeding on the corpus-scoped key keeps another tenant's
                # entities out of the walk, so the bounded path budget is spent
                # here and a warrant does not change when a neighbour ingests
                source_property="ckey",
                source_values=[schema.corpus_key(self.corpus, e) for e in entities],
                max_len=self.settings.path_max_len,
                direction="both",
                count=self.settings.path_count,
            )
        except HydraError as exc:
            log.warning("path expansion failed for %s: %s", self.corpus, exc)
            return 0
        for row in rows:
            self._absorb(row.get("path"), candidates, turns)
        return len(rows)

    def _absorb(
        self,
        path: Any,
        candidates: dict[int, _Candidate],
        turns: dict[int, dict[str, Any]],
    ) -> None:
        """Parse one returned path into facts, hop distances and provenance.

        `MSpaths` seeds on a property value across the whole graph, so a node
        belonging to another corpus can and does appear here. Every node is
        checked against the corpus before it is believed.
        """
        if path is None:
            return
        nodes = list(getattr(path, "nodes", ()) or ())
        rels = list(getattr(path, "relationships", ()) or ())
        if not nodes:
            return

        labels = [set(getattr(n, "labels", ()) or ()) for n in nodes]
        props = [dict(n) for n in nodes]
        if props[0].get("corpus") not in (None, self.corpus):
            return

        for i, node in enumerate(nodes):
            if schema.FACT not in labels[i] or props[i].get("corpus") != self.corpus:
                continue
            fid = _vertex_id(node)
            if fid is None:
                continue
            entry = candidates.setdefault(fid, _Candidate(fid=fid))
            entry.props = entry.props or props[i]
            entry.by_graph(
                hops=i,
                path=_readable(labels[: i + 1], props[: i + 1], rels[:i]),
                anchor=ANCHOR_DIRECT if i <= 1 else ANCHOR_INDIRECT,
            )

        # a DERIVED_FROM edge on the path already carries the turn, so the
        # provenance snippet is free for any fact the traversal walked through
        for rel in rels:
            if getattr(rel, "type", "") != schema.DERIVED_FROM:
                continue
            src = _vertex_id(getattr(rel, "start_node", None))
            dst = getattr(rel, "end_node", None)
            if src is None or dst is None:
                continue
            turn = dict(dst)
            if turn.get("corpus") == self.corpus:
                turns.setdefault(src, turn)

    def _lexical(self, question: str, candidates: dict[int, _Candidate]) -> list[int]:
        """Fold in the BM25 hits, and report the strongest as expansion anchors."""
        if self.index is None or not len(self.index):
            return []
        hits = self.index.search(question, self.settings.seed_lexical)
        for fid, score in hits:
            candidates.setdefault(int(fid), _Candidate(fid=int(fid))).by_index(float(score))
        return [
            (int(fid), float(score))
            for fid, score in hits[:LEXICAL_ANCHORS]
            if score >= LEXICAL_ANCHOR_FLOOR
        ]

    def _expand_from_facts(
        self,
        anchors: Sequence[tuple[int, float]],
        candidates: dict[int, _Candidate],
        turns: dict[int, dict[str, Any]],
    ) -> int:
        """Walk out from the strongest lexical hits, not only from entity seeds.

        A question and the fact that answers it often share no words. "Where did
        I redeem the coffee creamer coupon" matches the fact recording the
        redemption perfectly and the fact naming the shop not at all - they are
        two claims lifted from one turn, and only the second holds the answer.
        Seeding the walk from a lexical hit reaches its siblings through the turn
        they were both derived from, which is the join the index cannot make and
        the graph makes in one statement.

        Entity seeding stays first: it is the broader net. This is the fallback
        for the questions that net misses, and it is bounded to a couple of
        anchors so a vague question cannot turn into a corpus scan.
        """
        walked = 0
        for fid, strength in anchors:
            try:
                rows = self.client.paths(
                    rel_types=ANCHOR_RELS,
                    source_node=int(fid),
                    max_len=ANCHOR_MAX_LEN,
                    direction="both",
                    count=self.settings.path_count,
                )
            except HydraError as exc:
                log.info("expansion from fact %s failed: %s", fid, exc)
                continue
            before = set(candidates)
            for row in rows:
                self._absorb(row.get("path"), candidates, turns)
            walked += len(rows)
            # everything this walk brought in is a sibling of a fact the question
            # matched, and carries the anchor's strength as its own evidence
            for fid in set(candidates) - before | {int(fid)}:
                candidates[fid].by_neighbour(strength)
        return walked

    def _hydrate(self, candidates: dict[int, _Candidate]) -> None:
        """Fill in properties for candidates that arrived without them."""
        missing = [fid for fid, c in candidates.items() if not c.props]
        for fid, props in self._fetch_facts(missing).items():
            candidates[fid].props = props
        for fid in list(candidates):
            props = candidates[fid].props
            if not props or props.get("corpus") != self.corpus:
                del candidates[fid]

    def _corpus_facts(self) -> int:
        """How many facts this corpus holds. Cached: it only picks a strategy."""
        if self._fact_count is None:
            rows = self.client.run(
                f"MATCH (f:{schema.FACT} {{corpus: $corpus}}) RETURN count(*) AS n",
                corpus=self.corpus,
            )
            self._fact_count = int(rows[0]["n"]) if rows else 0
        return self._fact_count

    def _fetch_facts(self, ids: Iterable[int]) -> dict[int, dict[str, Any]]:
        """Read facts by id. No ``IN``, so this is a batched ``UNION`` of arms.

        Except when it is cheaper not to be. A ``UNION`` arm costs ~20 ms on this
        engine whatever it returns, so twenty-five arms is half a second, while
        reading every fact in a small corpus in one statement is ~30 ms. Below
        :data:`SCAN_BUDGET` facts the whole corpus is read and filtered here;
        above it the per-id form wins and keeps the read proportional to the
        question instead of to the memory.
        """
        wanted = _dedupe_ints(ids)
        out: dict[int, dict[str, Any]] = {}
        if len(wanted) > SCAN_ARMS and self._corpus_facts() <= SCAN_BUDGET:
            keep = set(wanted)
            for row in self.client.run(
                f"MATCH (f:{schema.FACT} {{corpus: $corpus}}) RETURN {_FACT_FIELDS}",
                corpus=self.corpus,
            ):
                if row.get("id") is not None and int(row["id"]) in keep:
                    out[int(row["id"])] = dict(row)
            return out
        for chunk in _chunks(wanted, FETCH_ARMS):
            arms = []
            params: dict[str, Any] = {}
            for i, fid in enumerate(chunk):
                params[f"p{i}"] = int(fid)
                arms.append(
                    f"MATCH (f:{schema.FACT} {{id: $p{i}}}) RETURN {_FACT_FIELDS}"
                )
            for row in self.client.run(" UNION ".join(arms), **params):
                if row.get("id") is not None:
                    out[int(row["id"])] = dict(row)
        return out

    # --------------------------------------------------------------- resolution

    def _resolve(
        self,
        candidates: dict[int, _Candidate],
        as_of: int | None,
    ) -> tuple[list[_Candidate], int]:
        """Collapse supersession chains and drop what may not be warranted.

        Two questions, one graph. Without ``as_of`` the head of each chain is the
        answer; with it, the member whose ``[valid_from, valid_to)`` interval
        contains the instant is. Neither overwrites the other, which is the whole
        point of keeping validity apart from ingestion time.
        """
        if not candidates:
            return [], 0
        newer, older = self._supersession(list(candidates))

        # every member of every candidate's lineage may end up being the fact the
        # question wants, so their properties are fetched once, up front, rather
        # than one id-lookup per chain walk
        lineages = {fid: _lineage(fid, newer, older) for fid in candidates}
        props: dict[int, dict[str, Any]] = {
            fid: dict(cand.props) for fid, cand in candidates.items() if cand.props
        }
        wanted = [m for members in lineages.values() for m in members if m not in props]
        props.update(self._fetch_facts(wanted))

        selected: dict[int, _Candidate] = {}
        for fid, cand in candidates.items():
            target = _select(lineages[fid], props, newer, as_of, fid)
            if target is None:
                continue
            if target != fid:
                cand = _Candidate(
                    fid=target,
                    props=dict(props.get(target) or cand.props),
                    hops=cand.hops,
                    path=[*cand.path, schema.SUPERSEDES, f"Fact:{target}"],
                    anchor=cand.anchor,
                    lexical=cand.lexical,
                    routes=set(cand.routes),
                )
            existing = selected.get(cand.fid)
            if existing is None:
                selected[cand.fid] = cand
            else:
                existing.absorb(cand)

        quarantined = 0
        keep: list[_Candidate] = []
        for cand in selected.values():
            status = str(cand.props.get("status") or schema.ACTIVE)
            rank = int(cand.props.get("rank") or 0)
            if status == schema.QUARANTINED or rank < int(self.min_tier):
                quarantined += 1
                continue
            if status not in schema.WARRANTABLE:
                continue
            cand.props["superseded_by"] = newer.get(cand.fid)
            keep.append(cand)
        return keep, quarantined

    def _supersession(self, ids: Sequence[int]) -> tuple[dict[int, int], dict[int, int]]:
        """``{displaced: replacement}`` and its inverse, for the facts in play.

        Targeted by id rather than scanned over the corpus, and this is the one
        place where the difference is dramatic. A corpus-wide
        ``MATCH (a:Fact {corpus: $c})-[:SUPERSEDES]->(b:Fact)`` measures ~500 ms
        however few edges come back, because the cost is walking every Fact's
        edge list; the same edges fetched by id through a batched ``UNION`` are
        index lookups and measure ~40 ms. It also makes the read scale with the
        question rather than with the memory.

        Both directions go in one statement - the two arms return the same two
        columns - and the frontier is expanded until it closes, so a chain that
        reaches further than the traversal did is still followed to its head.
        """
        newer: dict[int, int] = {}
        older: dict[int, int] = {}
        frontier = _dedupe_ints(ids)
        seen = set(frontier)
        for _ in range(SUPERSESSION_ROUNDS):
            if not frontier:
                break
            reached: list[int] = []
            for chunk in _chunks(frontier, EDGE_ARMS):
                arms: list[str] = []
                params: dict[str, Any] = {}
                for i, fid in enumerate(chunk):
                    params[f"p{i}"] = int(fid)
                    arms.append(
                        f"MATCH (a:{schema.FACT} {{id: $p{i}}})"
                        f"-[:{schema.SUPERSEDES}]->(b:{schema.FACT}) {_EDGE_FIELDS}"
                    )
                    arms.append(
                        f"MATCH (a:{schema.FACT})-[:{schema.SUPERSEDES}]->"
                        f"(b:{schema.FACT} {{id: $p{i}}}) {_EDGE_FIELDS}"
                    )
                for row in self.client.run(" UNION ".join(arms), **params):
                    if row.get("newer") is None or row.get("older") is None:
                        continue
                    up, down = int(row["newer"]), int(row["older"])
                    newer[down] = up
                    older[up] = down
                    reached.extend((up, down))
            frontier = [f for f in _dedupe_ints(reached) if f not in seen]
            seen.update(frontier)
        return newer, older

    # ------------------------------------------------------------------ ranking

    def _rank(self, candidates: Sequence[_Candidate]) -> list[Evidence]:
        """Blend lexical match, anchoring, proximity, tier, corroboration, recency.

        Recency is a *rank* within the candidate set rather than an absolute age,
        which keeps the term meaningful whether the corpus spans a week or a
        decade and stops a single very old fact flattening everything else.
        """
        if not candidates:
            return []

        corroboration = self._corroboration([c.fid for c in candidates])
        stamps = sorted({int(c.props.get("vfrom") or 0) for c in candidates})
        spread = max(1, len(stamps) - 1)
        recency_of = {stamp: i / spread for i, stamp in enumerate(stamps)}

        out: list[Evidence] = []
        for cand in candidates:
            props = cand.props
            distance = cand.hops if cand.hops >= 0 else LEXICAL_HOPS
            proximity = 1.0 / (1.0 + distance)
            tier = int(props.get("rank") or 0) / int(Tier.OWNER)
            corr = min(corroboration.get(cand.fid, 0), CORROBORATION_CAP) / CORROBORATION_CAP
            recency = recency_of.get(int(props.get("vfrom") or 0), 0.0)
            score = (
                W_LEXICAL * min(1.0, cand.lexical)
                + W_ANCHOR * min(1.0, cand.anchor)
                + W_NEIGHBOUR * min(1.0, cand.neighbour)
                + W_PROXIMITY * proximity
                + W_TIER * tier
                + W_CORROBORATION * corr
                + W_RECENCY * recency
            )
            out.append(
                Evidence(
                    fid=cand.fid,
                    text=str(props.get("text") or ""),
                    tier=str(props.get("tier") or Tier.EXTERNAL.label),
                    status=str(props.get("status") or schema.ACTIVE),
                    valid_from=int(props.get("vfrom") or 0),
                    valid_to=int(props.get("vto") or schema.OPEN_INTERVAL),
                    sid=str(props.get("sid") or ""),
                    sidx=int(props.get("sidx") or 0),
                    tidx=int(props.get("tidx") or 0),
                    score=round(score, 6),
                    hops=max(0, int(cand.hops)),
                    path=list(cand.path),
                    superseded_by=props.get("superseded_by"),
                )
            )
        out.sort(key=lambda e: (-e.score, e.hops, -e.valid_from, e.fid))
        return out

    def _corroboration(self, ids: Sequence[int]) -> dict[int, int]:
        """How many independent facts back each candidate, counting both ways.

        Fetched by id for the same reason as the supersession edges: an edge-list
        scan is priced by the size of the corpus and this is priced by the size
        of the question. Edges are de-duplicated client-side because an edge
        between two candidates comes back from both of their arms.
        """
        counts: dict[int, int] = {}
        edges: set[tuple[int, int]] = set()
        for chunk in _chunks(_dedupe_ints(ids), EDGE_ARMS):
            arms: list[str] = []
            params: dict[str, Any] = {}
            for i, fid in enumerate(chunk):
                params[f"p{i}"] = int(fid)
                arms.append(
                    f"MATCH (a:{schema.FACT} {{id: $p{i}}})"
                    f"-[:{schema.CORROBORATES}]->(b:{schema.FACT}) {_PAIR_FIELDS}"
                )
                arms.append(
                    f"MATCH (a:{schema.FACT})-[:{schema.CORROBORATES}]->"
                    f"(b:{schema.FACT} {{id: $p{i}}}) {_PAIR_FIELDS}"
                )
            for row in self.client.run(" UNION ".join(arms), **params):
                if row.get("src") is None or row.get("dst") is None:
                    continue
                edges.add((int(row["src"]), int(row["dst"])))
        for src, dst in edges:
            counts[src] = counts.get(src, 0) + 1
            counts[dst] = counts.get(dst, 0) + 1
        return counts

    # -------------------------------------------------------------- provenance

    def _attach_provenance(
        self,
        evidence: Sequence[Evidence],
        turns: dict[int, dict[str, Any]],
    ) -> None:
        """Give every surviving fact the turn text and timestamp it came from."""
        for item in evidence:
            turn = turns.get(item.fid)
            if turn:
                item.turn_text = str(turn.get("text") or "")
                item.turn_ts = int(turn.get("ts") or 0)

        missing = [e.fid for e in evidence if not e.turn_text]
        if not missing:
            return
        found: dict[int, dict[str, Any]] = {}
        for chunk in _chunks(missing, FETCH_ARMS):
            arms = []
            params: dict[str, Any] = {}
            for i, fid in enumerate(chunk):
                params[f"p{i}"] = int(fid)
                arms.append(
                    f"MATCH (f:{schema.FACT} {{id: $p{i}}})"
                    f"-[:{schema.DERIVED_FROM}]->(t:{schema.TURN}) RETURN {_PROV_FIELDS}"
                )
            for row in self.client.run(" UNION ".join(arms), **params):
                if row.get("fid") is not None:
                    found.setdefault(int(row["fid"]), dict(row))
        for item in evidence:
            row = found.get(item.fid)
            if row:
                item.turn_text = str(row.get("turn_text") or "")
                item.turn_ts = int(row.get("turn_ts") or 0)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _lineage(fid: int, newer: dict[int, int], older: dict[int, int]) -> list[int]:
    """Every fact connected to this one by a chain of SUPERSEDES edges."""
    seen = [fid]
    cursor = fid
    while cursor in newer and newer[cursor] not in seen:
        cursor = newer[cursor]
        seen.append(cursor)
    cursor = fid
    while cursor in older and older[cursor] not in seen:
        cursor = older[cursor]
        seen.append(cursor)
    return seen


def _select(
    lineage: Sequence[int],
    props: dict[int, dict[str, Any]],
    newer: dict[int, int],
    as_of: int | None,
    fid: int,
) -> int | None:
    """Which member of a supersession chain the question is asking for.

    With no ``as_of`` that is the head -- the one nothing supersedes. With an
    ``as_of`` it is the member whose ``[valid_from, valid_to)`` interval covers
    the instant, `valid_to == 0` meaning still open. A question about a moment
    before anything in the chain was true has no answer in it, and says so by
    returning None rather than by falling back to the head.
    """
    if as_of is None:
        head = fid
        seen = {fid}
        while head in newer and newer[head] not in seen:
            head = newer[head]
            seen.add(head)
        return head

    best: tuple[int, int] | None = None
    for member in lineage:
        member_props = props.get(member)
        if not member_props:
            continue
        vfrom = int(member_props.get("vfrom") or 0)
        vto = int(member_props.get("vto") or schema.OPEN_INTERVAL)
        if vfrom > as_of:
            continue
        if vto != schema.OPEN_INTERVAL and as_of >= vto:
            continue
        if best is None or vfrom > best[1]:
            best = (member, vfrom)
    return best[0] if best else None


def _vertex_id(node: Any) -> int | None:
    """HydraDB's vertex id, taken off the driver's element id.

    Path node property maps do not carry `id`, so the element id is the only
    place the identity is available on a traversal result.
    """
    if node is None:
        return None
    raw = getattr(node, "element_id", None)
    if raw is None:
        return None
    text = str(raw)
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    try:
        return int(text)
    except ValueError:
        return None


def _readable(
    labels: Sequence[set[str]],
    props: Sequence[dict[str, Any]],
    rels: Sequence[Any],
) -> list[str]:
    """A hop-by-hop rendering of the chain that reached a fact."""
    out: list[str] = []
    for i, prop in enumerate(props):
        out.append(_node_label(labels[i], prop))
        if i < len(rels):
            out.append(getattr(rels[i], "type", "?"))
    return out


def _node_label(labels: set[str], props: dict[str, Any]) -> str:
    label = next(iter(sorted(labels)), "Node")
    if schema.ENTITY in labels:
        return f"Entity:{props.get('norm') or props.get('name') or '?'}"
    if schema.FACT in labels:
        return f"Fact:{_clip(props.get('key') or props.get('text') or '?')}"
    if schema.TURN in labels:
        return f"Turn:{props.get('sid') or '?'}#{props.get('idx', '?')}"
    if schema.SESSION in labels:
        return f"Session:{props.get('sid') or '?'}"
    return label


def _clip(value: Any, limit: int = 48) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _dedupe_ints(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        number = int(value)
        if number not in seen:
            seen.add(number)
            out.append(number)
    return out


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]
