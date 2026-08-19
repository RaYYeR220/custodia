"""A BM25 index over fact text, living beside the graph.

HydraDB stores scalar properties only, so there is nowhere in the graph to put a
posting list or a vector. The index therefore lives next to it, keyed by the same
integer vertex ids, which is the whole reason ids are deterministic: a rebuilt
index and a re-ingested graph agree without a join table.

Why BM25 rather than embeddings: seeding is a recall problem, not a semantic
one. The entity seeds already carry the meaning of the question; lexical seeds
exist to catch the claims no entity extractor tagged -- a model number, a street
name, a phrase the user only ever used once. BM25 is the right tool for that, it
needs no provider, and it is deterministic, which matters because a warrant that
cannot be reproduced cannot be audited.

Scores come back normalised to 0..1 so the retriever can blend them with graph
proximity without having to know anything about term statistics.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from custodia import config, schema
from custodia.hydra.client import HydraClient

#: BM25 term-frequency saturation
K1 = 1.5
#: BM25 length normalisation
B = 0.75

#: index format version, written into every file so a stale one is detectable
FORMAT = 1

# Unicode word characters minus underscore: keeps accented letters and digits,
# splits on punctuation, and does not glue snake_case identifiers together.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

#: Closed-class words plus the interrogatives. Question words are dropped
#: deliberately: "where does the user live" should seed on `live`, and `where`
#: matches every location fact in the corpus equally, which is no signal at all.
STOPWORDS = frozenset(
    """
    a about after again all also am an and any are as at be because been before
    being both but by can could did do does doing done down during each few for
    from further had has have having he her here hers him his how i if in into is
    it its itself just me might more most must my no nor not now of off on once
    only or other our ours out over own same she should so some such than that
    the their theirs them then there these they this those through to too under
    until up us very was we were what when where which while who whom why will
    with would you your yours
    """.split()
) | {
    # the orphans an apostrophe leaves behind: "Nora's" -> nora, s and "don't"
    # -> don, t. They carry no meaning, they appear in a third of all English
    # text, and left in they let a fact match a question purely on possession
    "s",
    "t",
}


def words(text: str) -> list[str]:
    """Lowercased word tokens, unstemmed - the form entity keys are stored in."""
    return [m.group(0).lower() for m in _WORD.finditer(text or "")]


def tokenize(text: str) -> list[str]:
    """Index/query tokens: lowercased, stopped and lightly stemmed."""
    out: list[str] = []
    for word in words(text):
        if word in STOPWORDS:
            continue
        stem = _stem(word)
        if stem:
            out.append(stem)
    return out


def _stem(word: str) -> str:
    """A deliberately small suffix stripper: plurals, -ing, -ed, trailing -e.

    Full Porter is more accurate and less predictable. What matters here is that
    `like`, `likes`, `liked` and `liking` collapse onto one key and that the
    result is stable across runs; anything cleverer buys accuracy we cannot
    measure and costs reproducibility we can.
    """
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif len(word) > 5 and word.endswith(("ches", "shes", "xes", "zes", "ses")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    elif word.endswith("ing") and len(word) > 5:
        word = _undouble(word[:-3])
    elif word.endswith("ed") and len(word) > 4:
        word = _undouble(word[:-2])
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _undouble(stem: str) -> str:
    """`running` -> `runn` -> `run`, but `falling` -> `fall` keeps its pair."""
    if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
        return stem[:-1]
    return stem


def index_root() -> Path:
    return config.REPO_ROOT / "data" / "index"


def index_path(corpus: str) -> Path:
    """Default on-disk location for a corpus index."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", corpus) or "default"
    return index_root() / f"{safe}.json"


@dataclass(slots=True)
class LexicalIndex:
    """An inverted index over one corpus' facts, keyed by fact vertex id."""

    corpus: str = ""
    #: fact vertex ids, positionally aligned with :attr:`lengths`
    ids: list[int] = field(default_factory=list)
    #: document length in tokens, per document slot
    lengths: list[int] = field(default_factory=list)
    #: term -> {document slot: term frequency}
    postings: dict[str, dict[int, int]] = field(default_factory=dict)
    k1: float = K1
    b: float = B

    # ---------------------------------------------------------------- building

    @classmethod
    def build(cls, client: HydraClient, corpus: str) -> "LexicalIndex":
        """Read every fact in the corpus and index its text and triple parts.

        Quarantined facts are indexed too. They must stay *retrievable* so the
        retriever can drop them and report how many it dropped -- an attack that
        is silently unindexed is an attack nobody can show a judge.
        """
        rows = client.run(
            # the equality goes in the pattern, not in a WHERE: HydraDB uses the
            # vertex index for the first and scans for the second
            f"MATCH (f:{schema.FACT} {{corpus: $corpus}}) "
            "RETURN f.id AS id, f.text AS text, f.subj AS subj, f.pred AS pred, f.obj AS obj",
            corpus=corpus,
        )
        docs = [
            (
                int(row["id"]),
                " ".join(
                    str(row.get(part) or "") for part in ("text", "subj", "pred", "obj")
                ),
            )
            for row in rows
            if row.get("id") is not None
        ]
        return cls.from_documents(corpus, docs)

    @classmethod
    def from_documents(cls, corpus: str, docs: Iterable[tuple[int, str]]) -> "LexicalIndex":
        index = cls(corpus=corpus)
        for fid, text in docs:
            index.add(fid, text)
        return index

    def add(self, fid: int, text: str) -> None:
        """Index one document. Re-adding the same id appends a second slot, so
        callers rebuild rather than patch; the graph is the source of truth."""
        slot = len(self.ids)
        tokens = tokenize(text)
        self.ids.append(int(fid))
        self.lengths.append(len(tokens))
        for token in tokens:
            bucket = self.postings.setdefault(token, {})
            bucket[slot] = bucket.get(slot, 0) + 1

    # --------------------------------------------------------------- searching

    @property
    def avg_length(self) -> float:
        if not self.lengths:
            return 0.0
        return sum(self.lengths) / len(self.lengths)

    def search(self, query: str, k: int = 24) -> list[tuple[int, float]]:
        """Top ``k`` (fact id, score) pairs, scores normalised so the best is 1.

        Normalising against the best hit for *this* query is what makes the
        number blendable: the retriever wants "how well did the index like this
        fact, relative to the alternatives", not an absolute BM25 magnitude that
        moves with corpus size and query length.
        """
        if not self.ids or k <= 0:
            return []
        terms = tokenize(query)
        if not terms:
            return []

        n = len(self.ids)
        avg = self.avg_length or 1.0
        scores: dict[int, float] = {}
        for term in terms:
            bucket = self.postings.get(term)
            if not bucket:
                continue
            df = len(bucket)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for slot, tf in bucket.items():
                norm = self.k1 * (1.0 - self.b + self.b * self.lengths[slot] / avg)
                scores[slot] = scores.get(slot, 0.0) + idf * tf * (self.k1 + 1.0) / (tf + norm)

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.ids[kv[0]]))[:k]
        best = ranked[0][1] or 1.0
        return [(self.ids[slot], round(score / best, 6)) for slot, score in ranked]

    # ------------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format": FORMAT,
            "corpus": self.corpus,
            "k1": self.k1,
            "b": self.b,
            "ids": self.ids,
            "lengths": self.lengths,
            # JSON object keys are strings, so postings are stored as flat
            # [slot, tf, slot, tf, ...] runs: smaller than nested objects and it
            # avoids a str->int key conversion on every term at load time.
            "postings": {term: _flatten(bucket) for term, bucket in self.postings.items()},
        }
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LexicalIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("format", 0)) != FORMAT:
            raise ValueError(f"unsupported lexical index format in {path}")
        return cls(
            corpus=str(payload.get("corpus", "")),
            ids=[int(i) for i in payload.get("ids", [])],
            lengths=[int(i) for i in payload.get("lengths", [])],
            postings={
                str(term): _unflatten(run)
                for term, run in (payload.get("postings") or {}).items()
            },
            k1=float(payload.get("k1", K1)),
            b=float(payload.get("b", B)),
        )

    def store(self) -> Path:
        """Save to the corpus' default location and return where it went."""
        path = index_path(self.corpus)
        self.save(path)
        return path

    @classmethod
    def open(cls, corpus: str) -> "LexicalIndex | None":
        """Load the default on-disk index for a corpus, or None if none exists."""
        path = index_path(corpus)
        if not path.exists():
            return None
        return cls.load(path)

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"LexicalIndex(corpus={self.corpus!r}, docs={len(self)}, terms={len(self.postings)})"


def _flatten(bucket: dict[int, int]) -> list[int]:
    out: list[int] = []
    for slot, tf in bucket.items():
        out.append(int(slot))
        out.append(int(tf))
    return out


def _unflatten(run: Sequence[int]) -> dict[int, int]:
    return {int(run[i]): int(run[i + 1]) for i in range(0, len(run) - 1, 2)}
