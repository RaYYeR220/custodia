# Claims, and what backs each one

Every public statement this project makes, tagged by the strongest evidence we
actually have for it. The point of the table is that a reader can tell the
difference without having to read the code first.

| Tier | Means |
|---|---|
| **measured** | a number produced by a command in this repository, reproducible by running it |
| **demonstrated** | behaviour you can watch happen live, but not reduced to a number |
| **reasoned** | follows from the design and the code, argued rather than measured |
| **not claimed** | listed so nobody has to guess whether we meant it |

---

## The product

| Claim | Tier | Backing |
|---|---|---|
| A question memory cannot support is refused rather than answered | **measured** | `custodia verify` — the walkthrough's two abstention steps, plus the abstention metrics in `eval/` |
| The refusal is enforced in code, not by instructing the model | **demonstrated** | `src/custodia/gate.py` — nine checks run after the model replies; `tests/test_gate.py` exercises every abstention branch, including a hallucinated citation id |
| Every answer cites the facts it used, and every citation resolves to a stored fact | **measured** | the citation check rejects any id outside the warrant; `custodia verify` asserts the cited sessions per step |
| Every fact is bound to the turn it came from | **measured** | `custodia audit` → `orphan_facts: 0` on every corpus; `tests/test_ingest.py::test_no_fact_reaches_the_graph_without_its_provenance` |
| A fact that was later revised is superseded, not overwritten, and both remain answerable | **demonstrated** | `custodia ask "…coffee place?"` versus the same question `--as-of 2026-02-15`; three supersession chains in the demo corpus |
| A tool- or external-tier write cannot act on an owner-tier fact | **demonstrated** | `custodia attack --tier tool …` holds the answer; `tests/test_policy.py` covers the tier matrix across all four tiers and four operations |
| A refused write is kept, with the rule that refused it | **demonstrated** | `custodia audit`, the refusal ledger in the web client |
| The guard discriminates rather than blanket-refusing | **demonstrated** | the negative control: the same record updated by the principal is accepted, and the answer moves |
| Retrieval returns evidence chains, not just matching text | **demonstrated** | `custodia evidence "…" --paths` prints the hop sequence the engine walked |
| An advisory answer is labelled as such and still cites what shaped it | **demonstrated** | `Verdict.kind` is `grounded`; `tests/test_gate.py` covers the label, the relevance check, invented citations, and the strict mode that refuses the kind entirely |
| The whole walkthrough runs with no API key | **measured** | `CUSTODIA_LLM_API_KEY= CUSTODIA_CACHE_ONLY=true custodia verify` → 8/8 |

## HydraDB

| Claim | Tier | Backing |
|---|---|---|
| Batched writes are ~460× single-statement writes | **measured** | 11 rows/s versus 5 100 rows/s on an idle instance; method and caveats in `docs/hydradb.md` |
| An idempotent re-ingest costs no more than the original | **measured** | replaying an identical 2 000-edge batch: 4 900 edges/s against 3 600/s for the first write |
| Committed data survives an unclean kill and reads correctly | **measured** | `docker kill`, restart, same answer citing the same fact ids, `custodia audit` still ok. **The node does not return to a writable state on the local object store** — stated in full in PROOF.md §4 |
| Retrieval is HydraDB's own path search rather than a client-side join | **reasoned** + **demonstrated** | `HydraClient.paths()` issues `algo.MSpaths`; the returned `Path` objects are parsed, not reconstructed |
| A property over ~32 KB fails the whole statement | **measured** | 32 000 characters stored, 40 000 refused; the client clips and warns |

## Evaluation

Numbers, sample sizes, seeds and models are in [PROOF.md](PROOF.md). Everything
there is **measured**, on a stated sample rather than the full benchmark, and the
report prints `not measured` wherever a metric was not produced.

The poisoning suite is our own construction, not a published benchmark. Its
generator, seed and full contents are in `eval/poison.py` and
`eval/suites/poison_suite.json`, and it ships with a negative control so that a
system which refuses everything cannot score well on it.

## Not claimed

- **No production hardening.** No multi-tenant isolation beyond the corpus key, no
  access control beyond HydraDB's token, no rate limiting, no key management.
- **No state-of-the-art extraction.** Extraction quality bounds everything
  downstream. A claim the extractor misses is a claim memory does not have, and
  Custodia will correctly decline to answer about it — safe, but still a miss,
  and it is visible in the per-question-type numbers.
- **No claim about attacks outside the six families** in `eval/poison.py`. The
  trust boundary is a tier rule plus a small set of content rules; content rules
  are pattern matching and will always be evadable at the margin. The tier rule
  is not, which is why it is the one doing the load-bearing work.
- **No claim that the rule-based offline extractor matches the model.** It exists
  so the system runs without credentials, and it is labelled wherever it runs.
- **No benchmark leaderboard position.** We ran a stratified sample against two
  baselines we implemented ourselves; that is a comparison, not a ranking.
- **A `grounded` answer is not a recalled fact.** It is advice constrained by the
  facts it cites. Every claim it makes *about the person* is cited, but the advice
  itself is the model's. We do not claim otherwise, which is why the kind is on the
  verdict, in the API payload, in the CLI and in the web client.
- **No claim of a formal guarantee.** The provenance invariant is enforced by our
  writer, not by the engine — `custodia audit` exists precisely because a
  guarantee we hold ourselves should be checkable by someone who doubts it.
