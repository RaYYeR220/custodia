# Building on HydraDB

Notes from putting a real workload on HydraDB, kept because the engine's shape
changed our design rather than the other way round.

Measurements are from `ghcr.io/hydra-db/hydradb:latest` in Docker on one laptop
with the local object store, against an otherwise idle instance. They are ratios,
not a benchmark: absolute figures move by an order of magnitude once anything
else on the machine is competing for the disk, which is itself worth knowing
before you size an ingest.

## What Custodia uses it for

The memory graph is the whole system state: sessions, turns, facts, entities,
the temporal edges between facts, the audit trail of what was answered, and the
record of what was refused. There is no second store holding the truth. The only
thing that lives outside is the BM25 index, and only because HydraDB property
values are scalars.

## The three engine facts that shaped the design

### 1. Batched writes are the write path, not an optimisation

A single-statement write commits to object storage and returns durable. That is
a good property and it has a price:

| write form | measured |
|---|---|
| one labelled `CREATE` per statement | **11 rows/s** |
| `UNWIND $rows AS row MERGE (n {id: row.id}) SET n:Fact, n.text = row.text, …` | **5 100 rows/s** (2 000-row batch) |
| `UNWIND $rows AS row MATCH (s:Fact {id: row.s}), (d:Turn {id: row.d}) MERGE (s)-[r:DERIVED_FROM {id: row.rid}]->(d)` | **3 600 edges/s** |
| the same edge batch replayed unchanged | 4 900 edges/s — an idempotent retry is not a discount, but it is not a penalty either |

Roughly **460x**. It is the difference between an ingest that finishes and one
that does not.

Concurrency does not help — writes serialise server-side, and 32 parallel
single-statement writers measured the same ~17/s as one. So the ingest pipeline
stages an entire corpus in memory and flushes it as a fixed sequence of batches:
vertices by label, then edges by type. `custodia.hydra.client` is built around
that and nothing else.

**Property values have a length ceiling.** A string over roughly 32 KB fails the
statement with `internal query execution error` -- and because a batch is one
statement, one oversized value takes the whole batch with it. Measured: 32 000
characters stored, 40 000 refused. The client clips at 32 000 with a visible
marker and a warning, rather than dropping the row: a turn we cannot store in
full is still a turn a fact has to point at.

A third property is worth designing around: an edge batch whose endpoint does
not exist fails the whole statement (`MATCH endpoint vertex … does not exist`)
rather than skipping the row. That is the right behaviour — a silently dropped
edge is a fact with no provenance — but it means the writer has to flush all
vertices before any edge that references them, which is why `flush()` is a fixed
sequence rather than an interleaved stream.

Two further constraints come with the batch forms. The server admits **1024 rows per
batch** (`client_query_batch_items rejected by admission control`), so batches
are chunked and a transient rejection halves the chunk and retries. And every row
in a batch must carry the same fields, because the statement is compiled once
from the field names — so rows are grouped by their key set before a statement is
built.

### 2. Vertex ids are the identity, and they are integers

There is no separate primary key to `MERGE` against: the `id` in the pattern *is*
the vertex. That is a constraint with a large payoff once you lean into it.
Custodia derives every id by hashing a namespaced key (`custodia.ids`), which
means:

* the id of a fact can be computed before the graph is touched, which is what
  lets a whole session be staged offline and flushed in two round trips;
* re-ingesting the same content rewrites the same vertices instead of duplicating
  them, so an interrupted ingest is fixed by running it again;
* crash recovery needed no extra machinery. Kill the container mid-ingest, bring
  it back, re-run: the committed prefix is already there and the rest lands on
  top.

`MERGE` also has no `ON CREATE` / `ON MATCH`, and a `MERGE` pattern may not carry
payload properties — the pattern is the identity being matched, so writing extra
properties into it would rewrite what it matched. The upsert form is therefore
always `MERGE` by id followed by `SET`, which is exactly what the batch writer
emits.

### 3. `algo.MSpaths` returns evidence chains, not endpoints

This is why the product is on a graph database at all.

Plain `MATCH` projects endpoint properties. Retrieval needs the *path*: the fact,
the turn it was derived from, the session that turn sat in, and any supersession
that displaced it. HydraDB's native path procedures return the whole path,
bounded, from many sources at once:

```cypher
CALL algo.MSpaths({
  sourceLabel: 'Entity', sourceProperty: 'norm',
  sourceValues: ['shellfish', 'sesame', 'nora'],
  relTypes: ['MENTIONS', 'DERIVED_FROM', 'SUPERSEDES', 'CORROBORATES'],
  maxLen: 3, relDirection: 'both', pathCount: 400
}) YIELD path, pathCost RETURN path, pathCost
```

One statement takes every entity the question resolved to and comes back with
the chains that reach them. Custodia parses each path into an `Evidence` record
whose `path` field is the literal hop sequence the engine walked — which is what
the client renders when you click "chain of custody" on a citation, and what
makes a citation checkable rather than decorative.

The equivalent on a vector store is a similarity list with no structure to
inspect. The equivalent with plain `MATCH` is one round trip per shape of chain.

`SSpaths` (one source) and `SPpaths` (source to target) are used for the
supersession lineage and for "how is this fact connected to that one".

## Working inside the Cypher subset

HydraDB implements a deliberate subset and rejects the rest at parse time with a
reason. Rather than route around it, each gap pushed the model somewhere better:

| not available | what we do instead |
|---|---|
| `IN` in `WHERE` | set membership is a hop through `MENTIONS` — the graph-native form anyway. Multi-value lookups go through `algo.MSpaths` `sourceValues`, or `UNION` arms. |
| `IS NULL` | absence is a sentinel: `valid_to = 0` means "still open". Explicit, indexable, and it survives a scalar-only property model. |
| variable-length `MATCH` without a fixed source id | multi-seed traversal is what the path procedures are for. |
| `min` / `max` aggregates | ordering plus `LIMIT 1`. |
| `CONTAINS` | `STARTS WITH` for prefix work; everything else is BM25's job. |
| explicit transactions | deterministic ids make every write idempotent, so a retry is safe without one. |
| node-only `MATCH` with no predicate | every read is label-scoped, which is good hygiene regardless. |

The one place the subset genuinely costs something: property values are scalars,
so the BM25 posting lists live in a JSON file beside the graph, keyed by vertex
id. It is honest to say that part is not in HydraDB.

## Durability, demonstrated rather than asserted

Writes commit to object storage per statement, so "did it survive" is testable
in ten seconds rather than argued about:

```bash
custodia seed
docker kill custodia-hydradb          # not a graceful shutdown
docker compose up -d hydradb
custodia ask "Is Nora allergic to anything?"
```

The memory is intact, and the answer still cites the January session it came
from. `custodia audit` re-checks the provenance invariant against the recovered
graph.

**One caveat we found by doing it.** On the *local* object store the node comes
back readable but not writable. Reads answer normally; every write returns
`internal query execution error`, and the node log gives the reason:

```
Bolt suppressed internal graph error
error collecting garbage [resource=Manifest,
                          error=ObjectStoreError(NotImplemented { operation: … })]
```

Manifest recovery after an abrupt exit needs an object-store operation the local
backend does not implement. `docker compose restart` does not clear it — only
discarding the store does. We have not run the same sequence against an
S3-compatible backend, which is what the engine is designed for, so this is a
finding about the local development configuration and nothing more. It is worth
knowing before you reach for `docker kill` on a store you care about.

## One integration note worth writing down

The Bolt driver treats `neo4j://` and `bolt://` differently, and with HydraDB the
difference bites. `neo4j://` performs routing discovery: the driver asks the
server where the cluster lives and then reconnects to whatever
`GRAPH_ADVERTISED_BOLT_ADDR` says. Left at its example value of
`127.0.0.1:7687`, every client is told to go and find the graph on its own
loopback — so a second instance on a different port silently serves the first
one's data, and a containerised client looks for the database inside itself.

Two things follow, and both are in this repository:

* connect with `bolt://` when you mean "this endpoint, directly";
* if you do use routing, advertise an address the *client* can resolve —
  `hydradb:7687` on a compose network, not `127.0.0.1:7687`.
