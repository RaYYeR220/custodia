"""Deterministic vertex identifiers.

HydraDB vertex ids are non-negative integers and *are* the identity a pattern
matches on, so there is no separate primary key to MERGE against. We therefore
derive every id by hashing a namespaced string key. Two consequences we rely on
throughout:

* re-ingesting the same content produces the same ids, so a crashed ingest can
  simply be replayed -- every write is an idempotent upsert;
* an id can be computed offline, before the graph is touched, which is what lets
  the writer stage a whole session and flush it in batched round trips.
"""

from __future__ import annotations

from hashlib import blake2b

# 62 bits keeps us clear of any signed-64 edge in transport encodings while
# leaving collision probability negligible at our scale (~1e-10 at 10^8 keys).
_MASK = (1 << 62) - 1


def vid(kind: str, *parts: str) -> int:
    """Stable vertex id for a namespaced key."""
    key = kind + "\x1f" + "\x1f".join(parts)
    digest = blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK


def corpus_id(corpus: str) -> int:
    return vid("corpus", corpus)


def session_id(corpus: str, sid: str) -> int:
    return vid("session", corpus, sid)


def turn_id(corpus: str, sid: str, index: int) -> int:
    return vid("turn", corpus, sid, str(index))


def fact_id(corpus: str, key: str) -> int:
    return vid("fact", corpus, key)


def entity_id(corpus: str, norm: str) -> int:
    return vid("entity", corpus, norm)


def source_id(corpus: str, origin: str) -> int:
    return vid("source", corpus, origin)


def query_id(corpus: str, text: str, ts: int) -> int:
    return vid("query", corpus, text, str(ts))


def answer_id(corpus: str, qid: int) -> int:
    return vid("answer", corpus, str(qid))


def rejection_id(corpus: str, rule: str, text: str, ts: int) -> int:
    return vid("rejection", corpus, rule, text, str(ts))


def edge_id(rel: str, src: int, dst: int) -> int:
    """Relationship id, so an edge write is an idempotent MERGE too."""
    return vid("edge", rel, str(src), str(dst))
