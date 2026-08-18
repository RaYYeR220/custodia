"""Two comparison systems that share Custodia's model but not its gate.

The point of these baselines is to isolate one variable. Both get the same
history, the same language model and the same decoding settings as Custodia; what
neither gets is the warrant-and-citation gate. Any difference in abstention
behaviour therefore belongs to the gate rather than to the model, which is the
claim the poison and abstention tables are making.

``FullContextBaseline`` is the "just use a long context window" answer. It is the
honest strong baseline and it is often good -- but a LongMemEval-S haystack is
~122k estimated tokens, so on most models it does not fit, and what happens then
is the interesting part. Truncation is measured and reported per question rather
than averaged away.

``VectorlessRagBaseline`` is the "just retrieve" answer: chunk, BM25, stuff the
top-k into the prompt. It is deliberately lexical rather than embedding-based, so
it can run with no API key beyond the answering call and so its retrieval is
inspectable.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import ChatLLM, TOKEN_ESTIMATOR, estimate_tokens
from .datasets import Instance

__all__ = [
    "BaselineAnswer",
    "Baseline",
    "FullContextBaseline",
    "VectorlessRagBaseline",
    "Bm25Index",
]


@dataclass(slots=True)
class BaselineAnswer:
    """What a baseline produced, and what it cost to produce it.

    ``truncated`` and ``notes`` exist so a report can never present a full-context
    number without saying how much context actually reached the model.
    """

    text: str
    prompt_tokens: int
    latency_ms: float
    truncated: bool = False
    notes: dict[str, Any] = field(default_factory=dict)


class Baseline:
    """Common shape: ``answer(instance) -> BaselineAnswer``."""

    name: str = "baseline"

    def answer(self, instance: Instance) -> BaselineAnswer:  # pragma: no cover - interface
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:  # pragma: no cover - interface
        return {"system": self.name}


_SYSTEM_PROMPT = (
    "You answer questions about a user from their past conversations with you. "
    "Answer only from the material given. If the material does not contain the "
    "answer, say so plainly and do not guess."
)

_QUESTION_TEMPLATE = """\
{material}

---
Today is {question_date}.
Question: {question}

Answer in one or two sentences. If the material above does not contain the
answer, reply that the information is not available.
"""


# --------------------------------------------------------------------------- #
# full context
# --------------------------------------------------------------------------- #


class FullContextBaseline(Baseline):
    """Concatenate the entire history into one prompt and ask.

    ``context_tokens`` is the model's usable input budget. When the haystack does
    not fit, the *oldest* sessions are dropped first -- the standard sliding
    window, and the one that flatters this baseline least dishonestly, since
    recency is the cheapest available prior. Whatever is dropped is counted, and
    the count travels into the record; a question answered on 60% of its history
    is not silently reported as a full-context result.
    """

    name = "fullcontext"

    def __init__(
        self,
        llm: ChatLLM,
        *,
        context_tokens: int = 128_000,
        reserve_tokens: int = 2_000,
        max_answer_tokens: int = 300,
    ) -> None:
        self.llm = llm
        self.context_tokens = context_tokens
        self.reserve_tokens = reserve_tokens
        self.max_answer_tokens = max_answer_tokens

    def provenance(self) -> dict[str, Any]:
        return {
            "system": self.name,
            "context_tokens": self.context_tokens,
            "reserve_tokens": self.reserve_tokens,
            "truncation": "drop oldest sessions first",
            "token_estimator": TOKEN_ESTIMATOR,
        }

    def build_prompt(self, instance: Instance) -> tuple[str, dict[str, Any]]:
        blocks = [
            f"### Session {i + 1} - {s.date}\n{s.text}"
            for i, s in enumerate(instance.sessions)
        ]
        budget = self.context_tokens - self.reserve_tokens - estimate_tokens(instance.question)
        kept: list[str] = []
        used = 0
        dropped = 0
        # walk newest-first so the oldest sessions are the ones that fall off
        for block in reversed(blocks):
            cost = estimate_tokens(block)
            if used + cost > budget and kept:
                dropped += 1
                continue
            kept.append(block)
            used += cost
        kept.reverse()
        material = "\n\n".join(kept)
        notes = {
            "sessions_total": len(blocks),
            "sessions_kept": len(kept),
            "sessions_dropped": dropped,
            "haystack_tokens_estimated": instance.estimated_tokens(),
            "context_budget_tokens": budget,
            "truncation_strategy": "drop oldest sessions first" if dropped else "none",
        }
        prompt = _QUESTION_TEMPLATE.format(
            material=material,
            question_date=instance.question_date,
            question=instance.question,
        )
        return prompt, notes

    def answer(self, instance: Instance) -> BaselineAnswer:
        prompt, notes = self.build_prompt(instance)
        started = time.perf_counter()
        text = self.llm.complete(
            prompt,
            system=_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=self.max_answer_tokens,
        )
        elapsed = (time.perf_counter() - started) * 1000
        return BaselineAnswer(
            text=text.strip(),
            prompt_tokens=estimate_tokens(prompt),
            latency_ms=round(elapsed, 1),
            truncated=notes["sessions_dropped"] > 0,
            notes=notes,
        )


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an the and or but if of to in on at for with from by as is are was were be been
    being it its this that these those i you he she they we my your his her their our me
    him them do does did done have has had will would can could should may might must not
    no so than then there here what which who whom when where why how all any both each
    few more most other some such only own same too very s t just don now""".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS]


@dataclass(slots=True)
class Bm25Index:
    """Okapi BM25 over short chunks.

    A local copy of a scoring function ``custodia.lexical`` also implements. The
    duplication is deliberate: a baseline that shares the system-under-test's
    retriever cannot show that the retriever helps, and the harness has to keep
    running when ``custodia.lexical`` is mid-change. Whichever index actually ran
    is recorded in the result file.
    """

    k1: float = 1.5
    b: float = 0.75
    docs: list[str] = field(default_factory=list)
    meta: list[dict[str, Any]] = field(default_factory=list)
    _freqs: list[Counter[str]] = field(default_factory=list)
    _lengths: list[int] = field(default_factory=list)
    _df: Counter[str] = field(default_factory=Counter)

    def add(self, text: str, meta: dict[str, Any] | None = None) -> None:
        terms = _tokens(text)
        self.docs.append(text)
        self.meta.append(meta or {})
        counts = Counter(terms)
        self._freqs.append(counts)
        self._lengths.append(len(terms))
        for term in counts:
            self._df[term] += 1

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if not self.docs:
            return []
        n = len(self.docs)
        avg = (sum(self._lengths) / n) or 1.0
        query_terms = _tokens(query)
        scores = [0.0] * n
        for term in query_terms:
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, counts in enumerate(self._freqs):
                tf = counts.get(term, 0)
                if not tf:
                    continue
                norm = tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * self._lengths[i] / avg)
                )
                scores[i] += idf * norm
        ranked = sorted(range(n), key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in ranked[:k] if scores[i] > 0]


def _custodia_index() -> Any | None:
    """Use ``custodia.lexical.LexicalIndex`` if it exposes ``add``/``search``."""
    try:
        from custodia.lexical import LexicalIndex  # noqa: PLC0415 - lazy on purpose
    except Exception:
        return None
    try:
        index = LexicalIndex()
    except Exception:
        return None
    if callable(getattr(index, "add", None)) and callable(getattr(index, "search", None)):
        return index
    return None


# --------------------------------------------------------------------------- #
# vectorless RAG
# --------------------------------------------------------------------------- #


class VectorlessRagBaseline(Baseline):
    """Chunk the history, BM25-retrieve top-k, answer from the retrieved chunks.

    Chunks are windows of consecutive turns rather than fixed character spans, so
    a user statement and the assistant reply that confirms it stay together --
    splitting them is a retrieval artefact, not a property of the method being
    compared against.
    """

    name = "rag"

    def __init__(
        self,
        llm: ChatLLM,
        *,
        top_k: int = 12,
        turns_per_chunk: int = 4,
        chunk_overlap: int = 1,
        max_answer_tokens: int = 300,
        prefer_custodia_index: bool = True,
    ) -> None:
        self.llm = llm
        self.top_k = top_k
        self.turns_per_chunk = turns_per_chunk
        self.chunk_overlap = chunk_overlap
        self.max_answer_tokens = max_answer_tokens
        self.prefer_custodia_index = prefer_custodia_index
        self.index_impl = "eval-local-bm25"

    def provenance(self) -> dict[str, Any]:
        return {
            "system": self.name,
            "top_k": self.top_k,
            "turns_per_chunk": self.turns_per_chunk,
            "chunk_overlap": self.chunk_overlap,
            "index": self.index_impl,
            "token_estimator": TOKEN_ESTIMATOR,
        }

    def chunks(self, instance: Instance) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        step = max(1, self.turns_per_chunk - self.chunk_overlap)
        for session in instance.sessions:
            turns = session.turns
            for start in range(0, max(1, len(turns)), step):
                window = turns[start : start + self.turns_per_chunk]
                if not window:
                    continue
                body = "\n".join(f"{t.role}: {t.content}" for t in window)
                out.append(
                    {
                        "sid": session.sid,
                        "date": session.date,
                        "ts": session.ts,
                        "start": start,
                        "text": f"[{session.date} | session {session.sid}]\n{body}",
                    }
                )
        return out

    def retrieve(self, instance: Instance) -> list[dict[str, Any]]:
        chunks = self.chunks(instance)
        if not chunks:
            return []
        index = _custodia_index() if self.prefer_custodia_index else None
        if index is not None:
            self.index_impl = "custodia.lexical"
            try:
                for position, chunk in enumerate(chunks):
                    index.add(chunk["text"], {"i": position})
                hits = index.search(instance.question, self.top_k)
                picked = [_hit_position(hit, chunks) for hit in hits]
                return [chunks[i] for i in picked if 0 <= i < len(chunks)][: self.top_k]
            except Exception:
                # a mid-change index must not silently degrade the baseline's
                # retrieval quality without the report saying so
                self.index_impl = "eval-local-bm25 (custodia.lexical raised)"
        else:
            self.index_impl = "eval-local-bm25"
        local = Bm25Index()
        for chunk in chunks:
            local.add(chunk["text"])
        return [chunks[i] for i, _ in local.search(instance.question, self.top_k)]

    def answer(self, instance: Instance) -> BaselineAnswer:
        retrieved = self.retrieve(instance)
        material = "\n\n".join(c["text"] for c in retrieved)
        prompt = _QUESTION_TEMPLATE.format(
            material=material or "(nothing retrieved)",
            question_date=instance.question_date,
            question=instance.question,
        )
        started = time.perf_counter()
        text = self.llm.complete(
            prompt,
            system=_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=self.max_answer_tokens,
        )
        elapsed = (time.perf_counter() - started) * 1000
        gold_sessions = set(instance.answer_session_ids)
        return BaselineAnswer(
            text=text.strip(),
            prompt_tokens=estimate_tokens(prompt),
            latency_ms=round(elapsed, 1),
            truncated=False,  # retrieval selects rather than truncates
            notes={
                "chunks_indexed": len(self.chunks(instance)),
                "chunks_retrieved": len(retrieved),
                "index": self.index_impl,
                "evidence_session_hit": bool(
                    gold_sessions and gold_sessions & {c["sid"] for c in retrieved}
                )
                if gold_sessions
                else None,
            },
        )


def _hit_position(hit: Any, chunks: Sequence[dict[str, Any]]) -> int:
    """Read a chunk index out of whatever shape an external index returns."""
    if isinstance(hit, (int,)):
        return int(hit)
    if isinstance(hit, tuple) and hit:
        first = hit[0]
        if isinstance(first, int):
            return first
        if isinstance(first, dict) and "i" in first:
            return int(first["i"])
    if isinstance(hit, dict):
        for key in ("i", "index", "doc", "id"):
            if key in hit:
                return int(hit[key])
    raise TypeError(f"unrecognised search hit shape: {type(hit).__name__}")
