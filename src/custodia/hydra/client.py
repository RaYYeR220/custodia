"""A thin, opinionated client for HydraDB.

HydraDB executes a deliberate subset of OpenCypher, and the subset shapes how a
writer has to be built. Three facts drive everything below:

* a statement that writes one node at a time costs a durable object-store commit,
  which measures out at roughly a dozen writes a second. The batch forms --
  ``UNWIND $rows AS row MERGE ...`` -- run two orders of magnitude faster, so the
  writer stages work and flushes it in batches rather than writing as it goes;
* the server admits at most 1024 rows per batch, so batches are chunked;
* ``UNWIND`` needs every row in a batch to carry the same fields, because the
  statement is compiled once from the field names. Rows are therefore grouped by
  their key set before a statement is built.

Reads use the same connection. The path procedures (``algo.MSpaths`` and
friends) are exposed directly because they, not repeated ``MATCH`` round trips,
are how evidence chains come back out of the graph.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterable, Iterator, Sequence

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, TransientError

log = logging.getLogger("custodia.hydra")

#: server-side admission limit is 1024 rows; stay under it with room for retries
MAX_BATCH = 500


class HydraError(RuntimeError):
    """A query HydraDB refused, with the statement that caused it."""

    def __init__(self, message: str, cypher: str = "") -> None:
        super().__init__(message if not cypher else f"{message}\n  cypher: {cypher}")
        self.raw = message
        self.cypher = cypher


def _clean(value: Any) -> Any:
    """Coerce a property to something HydraDB stores: int, float, bool or str."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _row(props: dict[str, Any]) -> dict[str, Any]:
    return {k: _clean(v) for k, v in props.items() if _clean(v) is not None}


def _chunks(rows: Sequence[dict[str, Any]], size: int) -> Iterator[Sequence[dict[str, Any]]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _by_shape(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    """Group rows by their field set - one compiled statement per shape."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for r in rows:
        cleaned = _row(r)
        groups.setdefault(tuple(sorted(cleaned)), []).append(cleaned)
    return groups


class HydraClient:
    """Connection to a HydraDB graph over Bolt."""

    def __init__(
        self,
        uri: str | None = None,
        token: str | None = None,
        *,
        batch_size: int = MAX_BATCH,
        connect_timeout: float = 10.0,
    ) -> None:
        self.uri = uri or os.environ.get("HYDRA_URI", "neo4j://127.0.0.1:7687")
        self.token = token or os.environ.get("HYDRA_TOKEN", "local-development-token-32-bytes")
        self.batch_size = min(batch_size, 1000)
        self._driver: Driver = GraphDatabase.driver(
            self.uri,
            auth=("neo4j", self.token),
            connection_timeout=connect_timeout,
        )
        self.stats = {"queries": 0, "rows_written": 0, "read_ms": 0.0, "write_ms": 0.0}

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "HydraClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ping(self, retries: int = 30, delay: float = 1.0) -> bool:
        """Block until the graph answers, so compose-started stacks just work."""
        last: Exception | None = None
        for _ in range(retries):
            try:
                self.run("MATCH (c:Corpus) RETURN count(*) AS n")
                return True
            except (ServiceUnavailable, OSError, HydraError) as exc:
                last = exc
                time.sleep(delay)
        log.warning("hydradb unreachable at %s: %s", self.uri, last)
        return False

    # ------------------------------------------------------------------- reads

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        started = time.perf_counter()
        try:
            with self._driver.session() as session:
                result = session.run(cypher, **params)
                rows = [dict(record) for record in result]
        except TransientError as exc:
            raise HydraError(str(exc), cypher) from exc
        except Neo4jError as exc:
            raise HydraError(exc.message or str(exc), cypher) from exc
        self.stats["queries"] += 1
        self.stats["read_ms"] += (time.perf_counter() - started) * 1000
        return rows

    def count(self, label: str, **where: Any) -> int:
        clause = " AND ".join(f"n.{k} = ${k}" for k in where)
        cypher = f"MATCH (n:{label})" + (f" WHERE {clause}" if clause else "") + " RETURN count(*) AS n"
        rows = self.run(cypher, **where)
        return int(rows[0]["n"]) if rows else 0

    # ------------------------------------------------------------------ writes

    def upsert_nodes(self, label: str, rows: Sequence[dict[str, Any]]) -> int:
        """Idempotent vertex upsert. Every row needs an integer ``id``.

        Emitted as ``MERGE (n {id: row.id}) SET n:Label, n.prop = row.prop`` --
        the pattern carries identity only, because HydraDB rejects a MERGE whose
        pattern also writes payload.
        """
        written = 0
        for shape, group in _by_shape(rows).items():
            if "id" not in shape:
                raise ValueError(f"{label} rows must carry an integer id")
            assignments = ", ".join(f"n.{k} = row.{k}" for k in shape if k != "id")
            sets = f"n:{label}" + (f", {assignments}" if assignments else "")
            cypher = f"UNWIND $rows AS row MERGE (n {{id: row.id}}) SET {sets}"
            written += self._flush(cypher, group)
        return written

    def merge_edges(
        self,
        rel: str,
        src_label: str,
        dst_label: str,
        rows: Sequence[dict[str, Any]],
    ) -> int:
        """Idempotent edge upsert between vertices that already exist.

        Rows carry ``s`` (source id), ``d`` (destination id), ``rid``
        (deterministic relationship id) and any relationship properties.
        """
        written = 0
        for shape, group in _by_shape(rows).items():
            missing = {"s", "d", "rid"} - set(shape)
            if missing:
                raise ValueError(f"{rel} rows missing {sorted(missing)}")
            extra = [k for k in shape if k not in ("s", "d", "rid")]
            assignments = ", ".join(f"r.{k} = row.{k}" for k in extra)
            cypher = (
                f"UNWIND $rows AS row "
                f"MATCH (s:{src_label} {{id: row.s}}), (d:{dst_label} {{id: row.d}}) "
                f"MERGE (s)-[r:{rel} {{id: row.rid}}]->(d)"
            )
            if assignments:
                cypher += f" SET {assignments}"
            written += self._flush(cypher, group)
        return written

    def _flush(self, cypher: str, rows: Sequence[dict[str, Any]]) -> int:
        written = 0
        started = time.perf_counter()
        for chunk in _chunks(rows, self.batch_size):
            self._flush_chunk(cypher, chunk)
            written += len(chunk)
        self.stats["write_ms"] += (time.perf_counter() - started) * 1000
        self.stats["rows_written"] += written
        return written

    def _flush_chunk(self, cypher: str, chunk: Sequence[dict[str, Any]], depth: int = 0) -> None:
        try:
            with self._driver.session() as session:
                session.run(cypher, rows=list(chunk)).consume()
            self.stats["queries"] += 1
        except TransientError as exc:
            # admission control pushes back on oversized batches; halve and retry
            if depth < 4 and len(chunk) > 1:
                mid = len(chunk) // 2
                self._flush_chunk(cypher, chunk[:mid], depth + 1)
                self._flush_chunk(cypher, chunk[mid:], depth + 1)
                return
            raise HydraError(str(exc), cypher) from exc
        except Neo4jError as exc:
            raise HydraError(exc.message or str(exc), cypher) from exc

    # ------------------------------------------------------- path procedures

    def paths(
        self,
        *,
        rel_types: Sequence[str],
        max_len: int = 3,
        direction: str = "both",
        count: int = 200,
        source_node: int | None = None,
        source_label: str | None = None,
        source_property: str | None = None,
        source_values: Sequence[Any] | None = None,
        target_node: int | None = None,
        with_cost: bool = True,
    ) -> list[dict[str, Any]]:
        """Bounded path search via HydraDB's native procedures.

        Returns whole paths -- alternating node-property maps and relationship
        type names -- which is what makes an evidence *chain* retrievable in one
        round trip instead of reassembled client-side from endpoint rows.
        """
        config: dict[str, Any] = {
            "relTypes": list(rel_types),
            "maxLen": max_len,
            "relDirection": direction,
            "pathCount": count,
        }
        if target_node is not None:
            proc = "algo.SPpaths"
            config["sourceNode"] = source_node
            config["targetNode"] = target_node
        elif source_values is not None:
            proc = "algo.MSpaths"
            config["sourceLabel"] = source_label
            config["sourceProperty"] = source_property
            config["sourceValues"] = list(source_values)
        else:
            proc = "algo.SSpaths"
            config["sourceNode"] = source_node

        yields = "path, pathCost" if with_cost else "path"
        cypher = f"CALL {proc}({_literal(config)}) YIELD {yields} RETURN {yields}"
        return self.run(cypher)


def _literal(config: dict[str, Any]) -> str:
    """Render a procedure config map inline (procedure configs are not params)."""
    parts = []
    for key, value in config.items():
        if value is None:
            continue
        parts.append(f"{key}: {_render(value)}")
    return "{" + ", ".join(parts) + "}"


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render(v) for v in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"
