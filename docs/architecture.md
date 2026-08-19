# Architecture

Custodia is an agent memory layer whose central claim is narrow and testable: an
answer may only be produced when the memory graph contains evidence that
warrants it, and that evidence must be traceable back to the exact turn it came
from. Everything here follows from that.

## The graph

The memory of one principal is a `Corpus`. Everything else hangs off it.

```
(:Corpus)
   ^
   | IN_CORPUS
(:Session) <--IN_SESSION-- (:Turn) <--DERIVED_FROM-- (:Fact) --MENTIONS--> (:Entity)
                              ^                        |  ^
                              |                        |  |
                        FROM_SOURCE               SUPERSEDES / CONTRADICTS /
                              |                     CORROBORATES
                          (:Source)                     |
                                                     (:Fact)

(:Query) <--ANSWERS-- (:Answer) --CITES--> (:Fact)
(:Turn)  <--RAISED_BY-- (:Rejection) --BLOCKED--> (:Fact)
```

| Node | What it is |
|---|---|
| `Corpus` | one principal's memory namespace |
| `Session` | a conversation, ordered within the corpus |
| `Turn` | a single message, with its role and timestamp |
| `Fact` | one atomic claim, written to stand alone without its turn |
| `Entity` | a resolved subject or object, the anchor retrieval seeds on |
| `Source` | a non-conversational origin: a tool result, a document, another agent |
| `Query` / `Answer` | what was asked, what was answered, and on which evidence |
| `Rejection` | a write the policy engine refused, kept with its reason |

A `Fact` is bi-temporal. `valid_from` / `valid_to` describe when the claim was
true of the world; `ingested_at` describes when memory learned it. Keeping the
two apart is what lets "what is my current gym?" and "which gym did I use in
March?" be answered from the same graph without either overwriting the other.

### Provenance is a write-shape, not a convention

HydraDB's row-execution `CREATE` accepts relationship paths — a source id, a
relationship, a destination id — and nothing else. A fact and the edge to the
turn it came from are therefore naturally written as one statement, and the
writer keeps that shape end to end: `stage_fact()` will not accept a fact
without the turn it was derived from, and the batch flusher writes the
`DERIVED_FROM` edges in the same flush as the fact vertices.

This is an invariant Custodia enforces, informed by the engine's write model. It
is not one the engine imposes: the batch form `UNWIND $rows AS row MERGE (n {id:
row.id})` can create a bare vertex, so the guarantee is ours to keep, and
`custodia audit` exists to prove we kept it.

## Trust tiers

Every fact inherits its tier from the turn it was derived from, never from its
own content. Content that claims authority is still whatever channel carried it.

| Tier | Rank | Origin |
|---|---|---|
| `owner` | 3 | the principal's own messages |
| `assistant` | 2 | the agent's own statements |
| `tool` | 1 | tool and API results |
| `external` | 0 | web pages, documents, other agents — anything an attacker can influence |

The policy engine admits a write only if the acting tier outranks what the write
touches. A fact at `external` may be recorded, but it cannot supersede, retract
or contradict-resolve a fact at `owner`, and it never enters a warrant. Refused
writes are not dropped: they become `Rejection` nodes carrying the rule that
fired and the text that tripped it, so the audit trail shows attacks that were
attempted, not just facts that survived.

## Retrieval

Retrieval is a three-stage funnel and each stage is graph work.

1. **Seeding.** The question is resolved to entity keys and lexical terms. Entity
   keys match `Entity.norm` directly; lexical terms score against a local BM25
   index over fact text, which is what covers claims no entity extractor caught.
2. **Expansion.** Seeds go into `algo.MSpaths`, HydraDB's native multi-source
   bounded path search, walking `MENTIONS`, `DERIVED_FROM`, `SUPERSEDES` and
   `CORROBORATES`. It returns whole paths, so what comes back is an evidence
   *chain* — fact, the turn that produced it, the session that turn sat in —
   rather than a bag of nodes to be re-joined client-side.
3. **Resolution.** Chains are collapsed per fact. `SUPERSEDES` edges are followed
   to the current head unless the question asked as-of a time, and quarantined
   facts are dropped -- but counted, so an answer can report how many poisoned
   items its retrieval passed over.
4. **Ranking.** Six weighted terms summing to one, so a score is directly
   comparable to the evidence floor: lexical relevance (0.30), graph anchoring to
   a seeded entity (0.15), path proximity (0.15), tier (0.15), recency rank
   (0.15) and corroboration (0.10, saturating at three independent witnesses).

   Lexical relevance and anchoring are separate terms deliberately, and it is the
   one weighting worth arguing about. Anchoring alone does not discriminate: in a
   corpus about one person every fact hangs off that person, every fact scores
   identically on that term, and the ordering collapses onto recency. The lexical
   term is what puts the drink fact above the gym fact for "what do I drink", so
   it carries the larger weight. Recency is a *rank within the candidate set*
   rather than an absolute age, which keeps it meaningful whether the corpus
   spans a week or a decade.

The output is a **warrant**: a bounded set of facts, each with its provenance
chain and the timestamps that justify its position in time.

## The gate

The answering model never sees the conversation. It sees the warrant, and it is
required to return JSON: an answer, the fact ids it used, and whether the
evidence was sufficient.

What makes the refusal real is that the check is code, not instruction:

* every cited id must be present in the warrant that was handed out — a citation
  the model invented fails the lookup;
* an empty warrant, an empty citation list, or `sufficient: false` short-circuits
  to abstention before the answer text is ever used;
* malformed JSON, a timeout, or a provider error resolves to abstention, not to
  a best guess. The failure path and the "I don't know" path are the same path.

An answer that passes is written back as `(:Answer)-[:CITES]->(:Fact)`, so the
graph records not only what memory held but what was said from it.

## Modules

| Module | Responsibility |
|---|---|
| `custodia.ids` | deterministic vertex ids, which make every write an idempotent upsert |
| `custodia.schema` | labels, relationships, tiers, and the record types everything shares |
| `custodia.hydra.client` | batched writer, reader, and the path procedures |
| `custodia.llm` | provider-agnostic chat/JSON client with an on-disk cache |
| `custodia.extract` | turns to facts, entities and temporal hints |
| `custodia.resolve` | entity resolution and supersession detection |
| `custodia.policy` | trust admission, injection detection, rejection records |
| `custodia.ingest` | staging and batched flush of a whole corpus |
| `custodia.lexical` | BM25 index over fact text, used for seeding |
| `custodia.retrieve` | seeds, path expansion, temporal resolution, warrants |
| `custodia.gate` | warrant to answer, with the code-enforced citation check |
| `custodia.audit` | write-back, provenance explanations, integrity checks |
| `custodia.api` / `mcp_server` / `cli` | the surfaces |

## Working within HydraDB's Cypher subset

The engine accepts a deliberate subset, and the design leans on the parts that
are fast rather than around the parts that are missing.

* Writes are batched `UNWIND ... MERGE ... SET`, chunked under the 1024-row
  admission limit. Single-statement writes commit to object storage individually
  and measure 11/s; batched upserts measure 5 100/s on the same idle box. Property
  values are also length-limited — a string over ~32 KB fails the whole statement,
  so the client clips at 32 000 characters with a visible marker.
* Vertex ids are integers and *are* the identity, so ids are derived by hashing a
  namespaced key. Re-running an ingest rewrites the same vertices instead of
  duplicating them, which is also the crash-recovery story.
* Property values are scalars, so embeddings and the BM25 index live beside the
  graph rather than inside it, keyed by vertex id.
* `WHERE` has no `IN` and no `IS NULL`. Set membership is expressed as a path
  through `MENTIONS`, which is the graph-native form anyway; absence is expressed
  with a sentinel (`valid_to = 0` means "still open") rather than a null.
* Variable-length `MATCH` needs a fixed source id, so multi-seed traversal goes
  through `algo.MSpaths` rather than a pattern with a list of starts.
