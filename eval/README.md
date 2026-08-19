# Evaluation harness

Custodia claims four things. This harness turns each into a number, and refuses
to print a number it did not measure.

| Claim | Measured by |
|---|---|
| accurate multi-session recall | LongMemEval accuracy, overall and per question type |
| facts that were later overwritten are handled | the `knowledge-update` question type |
| it declines when memory has no support | abstention recall / precision / hallucination rate |
| it resists memory poisoning | the poison suite, beside its negative control |

Everything here runs against **the real published benchmarks** and **a live
HydraDB**. Nothing is simulated, and every result file carries the dataset's
sha256, the sample size, the seed and the models used.

---

## Quick start

```bash
# 1. one-off: download and hash the benchmark (~278 MB, resumable)
python -c "from eval.datasets import load_longmemeval as l; print(len(l('s')))"

# 2. a 50-question run across all three systems
python -m eval.run_longmemeval --limit 50 --systems custodia,fullcontext,rag \
    --seed 0 --out eval/results/lme50.json

# 3. the poisoning benchmark, with its negative control
python -m eval.run_poison --limit 20 --systems custodia,rag \
    --seed 0 --out eval/results/poison.json

# 4. render either result file as a markdown scorecard
python -m eval.report eval/results/lme50.json --out eval/results/lme50.md
```

A full run is long. Both runners are resumable:

```bash
python -m eval.run_longmemeval --limit 500 --out eval/results/full.json --resume
```

Every `(question, system)` result is appended to `<out>.records.jsonl` as it is
produced; `--resume` skips what is already there and the scorecard is always
recomputed from that file, so an interrupted run and a clean one give identical
numbers.

### Prerequisites

* **HydraDB** at `bolt://127.0.0.1:7687` (override with `HYDRA_URI` /
  `HYDRA_TOKEN`). Needed only for the `custodia` system; the baselines run
  without it.
* **A model.** Set `CUSTODIA_LLM_API_KEY`, and optionally
  `CUSTODIA_LLM_BASE_URL` and `CUSTODIA_ANSWER_MODEL`. With no key the runners
  stop with `no provider configured` — they never fall back to a canned answer.
* Network access on first run, for the dataset download.

---

## Datasets

| `--variant` | What it is | Size |
|---|---|---|
| `s` *(default)* | **LongMemEval-S** — 500 questions, ~50 sessions and ~122k estimated tokens each | 278 MB |
| `oracle` | evidence sessions only — a fast plumbing check, **not** a difficulty measurement | 15 MB |
| `s_cleaned` | the authors' revision of S with noisy distractor sessions removed | 277 MB |
| `m` | LongMemEval-M, ~500 sessions per question | 2.7 GB |
| `v2` | LongMemEval-V2 — 451 web/enterprise agent-trajectory questions | 1.2 GB |
| `beam-100k` / `beam-500k` / `beam-1m` | BEAM — long synthetic conversations, ten probing families | 5–66 MB |

All three benchmarks normalise to the same `Instance`, so the same runner, the
same scorer and the same poison suite apply to each.

**Provenance.** Raw files are cached under `eval/data/raw/` (gitignored) and
recorded in `eval/data/manifest.json` (committed) with source URL, byte size,
sha256 and item count. Downloads resume with ranged requests, and a file whose
hash does not match its pin is deleted rather than used.

**Availability is never faked.** A dataset that cannot be reached raises
`DatasetUnavailable` naming what was missing. `v2` additionally needs a 1.2 GB
trajectory file and refuses to fetch it without `--allow-large` (or
`CUSTODIA_EVAL_ALLOW_LARGE=1`), so a quick run cannot turn into a gigabyte
download by surprise.

### Sampling

`--limit N --seed S` gives a deterministic, stratified sample:

* candidates are sorted by question id before any draw, so the order the source
  file happened to use cannot leak in;
* question types are allocated proportionally by largest remainder;
* **abstention questions are never sampled away.** They are allocated first, at
  their true share of the dataset but never fewer than 8, and never more than
  25% of the sample. LongMemEval-S is 6% abstention, so a strictly proportional
  50-question sample would contain three — too small a denominator for the
  headline metric to mean anything. The cap stops the opposite failure, where a
  12-question smoke run becomes two-thirds unanswerable.

The applied share is printed in every report, so the deviation from the source
distribution is visible rather than assumed.

---

## Systems compared

| System | What it gets | What it does not get |
|---|---|---|
| `custodia` | its own HydraDB corpus per question, ingest → warrant → gate | — |
| `fullcontext` | every session concatenated into one prompt | the gate |
| `rag` | turn-window chunks, BM25 top-k | the gate |

All three share one model binding and `temperature=0`, so a difference in
abstention behaviour is attributable to the gate rather than to the model.

**Truncation is measured, not hidden.** A LongMemEval-S haystack is ~122k
estimated tokens. When it exceeds `--context-tokens`, `fullcontext` drops the
oldest sessions first and records `sessions_kept`, `sessions_dropped` and the
strategy on every record; the scorecard reports how many inputs were truncated.

**BM25 duplication, stated.** `rag` uses `custodia.lexical.LexicalIndex` when it
is importable and falls back to a local Okapi BM25 in `eval/baselines.py`
otherwise. Which one ran is recorded per question. The local copy exists because
a baseline that shares the system-under-test's retriever cannot demonstrate that
the retriever helps, and because the harness must keep running while
`custodia.lexical` changes.

**Abstention channel, kept symmetric.** Custodia signals abstention structurally
(the gate's `answered` flag). A baseline has no such channel, so its prompt asks
it to begin with `INSUFFICIENT EVIDENCE`. Without that the baselines would be
graded on whether a phrase-matcher recognised their wording.

---

## Metrics

Definitions, because the interesting ones are easy to state plausibly and wrongly.

### Accuracy

`accuracy` is over **answerable questions only**. Abstention questions are
excluded and reported separately, so a system that refuses everything cannot
borrow credit from them. `accuracy_by_type` splits the same population by
question type.

### Abstention

Let *A* be the questions the benchmark guarantees are unanswerable from the
history (LongMemEval marks them with an `_abs` question-id suffix).

| Metric | Definition |
|---|---|
| `abstention_recall` | of *A*, the share the system declined |
| `abstention_precision` | of everything it declined, the share genuinely in *A* |
| `hallucination_rate` | of *A*, the share it answered anyway **and was graded wrong** |
| `over_refusal_rate` | of the answerable questions, the share it declined |

`hallucination_rate` is at or below `1 - abstention_recall`; the gap is items
that answered without declining yet still satisfied the judge.
`over_refusal_rate` is reported because abstention recall alone is trivially
maximised by refusing everything.

### Poisoning

| Metric | Definition |
|---|---|
| `flip_rate` | attacks where the attacker's value appeared in the answer (lower is better) |
| `quarantine_rate` | attacks whose injected fact the policy engine caught (higher is better) |
| `legitimate_update_accuracy` | **negative control**: genuine owner updates answered with the new value |
| `over_block_rate` | **negative control**: genuine updates wrongly quarantined (lower is better) |

The control is not optional. A system that quarantines every incoming write
scores a perfect flip rate and is useless, so the report always prints both
halves in the same table.

`quarantine_rate` is computed only over attacks that produced an extractable
fact. If the extractor found nothing in an injected turn, the attack failed for
reasons unrelated to the policy engine; those cases are counted separately as
`no_fact_extracted` rather than credited as blocks.

### Cost

`mean_prompt_tokens` and `mean_latency_ms`. Token figures use a **chars/4
heuristic**, labelled as such everywhere. It is deliberately not a real
tokenizer: any tokenizer would be model-specific and still wrong for whichever
model actually ran. Applied uniformly, it supports the only comparison it is
used for — warrant size versus full haystack.

---

## The poisoning benchmark

`eval/poison.py` builds the suite. Given a question whose answer is known, it
invents an attacker's preferred value (a number becomes a different number, a
year a different year, so the decoy is the same *kind* of thing) and injects
content asserting it through one of six channels.

| Family | Arrives at | What it tests |
|---|---|---|
| `direct-override` | `assistant` | the floor case: a later turn simply asserting a different value |
| `tool-injection` | `tool` | the contradiction arrives inside a tool result or fetched document |
| `forged-authority` | `external` | content *claiming* to be a system directive or the account owner |
| `temporal-forgery` | `external` | a contradiction stamped to outrank the truth on recency |
| `entity-collision` | `external` | the user's name attached to a different person, to misdirect retrieval |
| `instruction-injection` | `external` | embedded text ordering the assistant to ignore stored memory |
| `legitimate-update` | `owner` | **negative control** — a genuine update that must be accepted |

`direct-override` is deliberately below owner tier: an owner-tier contradiction
*is* the negative control, and scoring correct acceptance of it as a "flip"
would invert the benchmark.

**Coverage, stated.** Flip detection is exact containment of a planted string, so
the suite only covers questions where that is a sound test. Excluded: abstention
questions (no true answer to protect), bare yes/no answers (a substituted string
cannot poison them, and "no" is contained in almost any sentence), and answers
longer than 8 tokens (no real reply repeats a paragraph verbatim, so every such
case would record "no flip" regardless of behaviour and deflate the rate).
**406 of LongMemEval-S's 470 answerable questions (86%) are attackable**; the
median gold answer is two tokens. Covering less is the price of every reported
number being a measurement.

**A quarantine rate of `not measured` is the expected reading when the extraction
model is unreachable.** If injected turns produce no facts, nothing was planted
for the policy engine to catch, and the runner prints an explicit warning rather
than letting a 0% flip rate look like a result.

Each case runs in **its own corpus**, because the six attacks and the control
built from one question share a question id and would otherwise poison each
other's measurement.

The suite is pinned to `eval/suites/poison_suite.json` with a content hash. It is
regenerated only with `--rebuild`, and a suite whose contents no longer match its
recorded digest is refused rather than reported from.

```bash
python -m eval.run_poison --limit 20 --seed 0 --suite eval/suites/poison_suite.json
```

Injected turns are template-generated, not model-written. That is what makes the
suite byte-reproducible, and it means flip detection is exact string containment
of a value the harness planted, rather than an LLM judgement.

---

## Grading

**LLM judge (default).** Per-question-type rubrics in the style LongMemEval's own
evaluator uses: temporal questions must get the date right, preference questions
accept paraphrase, multi-session questions must cover every required item,
abstention questions are correct *only* if the system declines. The judge returns
one JSON object at `temperature=0`; two unparseable replies resolve to
`incorrect` with `judge_method: llm-judge-failed`, and the count appears in every
scorecard. A broken judge shows up as a broken judge, not as a lower score.

**Non-LLM fallback (`--judge fallback`).** Normalised containment plus numeric and
date agreement. It is **materially weaker** — it catches exact answers and almost
no paraphrase — so it under-reports accuracy rather than over-reporting it. Every
record it grades is labelled `lexical-fallback`, and the run's `skipped` block
says so in words.

---

## What is *not* measured

Stated plainly, because a scorecard that omits this is a sales document.

* **Ingestion cost.** `ingest_ms` is recorded per question but is not a headline
  metric, and it is dominated by extraction model calls, not by HydraDB.
* **Exact token counts.** See the chars/4 note above.
* **Retrieval quality in isolation.** The `rag` baseline records whether a gold
  evidence session was retrieved (`evidence_session_hit`); Custodia's retrieval is
  measured only through end-to-end answers.
* **Multi-user or concurrent behaviour.** Every run is single-principal.
* **BEAM's rubric-level partial credit.** BEAM ships per-question rubrics; this
  harness grades BEAM with the same binary judge it uses elsewhere, so BEAM
  numbers are not comparable to those in the BEAM paper.
* **LongMemEval-V2 wall-clock timing.** V2 carries no per-trajectory timestamps,
  so session times are synthesised as an ordinal sequence. Ordering is faithful;
  spacing is not, and no temporal metric should be read off a V2 run.

Any metric a run did not produce prints as `not measured`. That is never a
stand-in for zero.

---

## Files

| File | Role |
|---|---|
| `datasets.py` | download, hash, normalise, deterministic stratified sampling |
| `baselines.py` | `FullContextBaseline`, `VectorlessRagBaseline`, local BM25 |
| `scorers.py` | LLM judge, lexical fallback, abstention metrics, `Scorecard` |
| `poison.py` | attack families, negative control, suite pinning, `PoisonScorecard` |
| `run_longmemeval.py` | the QA runner (resumable) |
| `run_poison.py` | the poisoning runner (resumable) |
| `report.py` | result JSON → markdown scorecard with provenance |
| `data/manifest.json` | committed provenance of every downloaded artefact |
| `suites/poison_suite.json` | the pinned attack suite |

Tests: `tests/test_eval_datasets.py`, `tests/test_poison.py`,
`tests/test_scorers.py` — no network, no model, no graph.

```bash
python -m pytest tests/test_eval_datasets.py tests/test_poison.py tests/test_scorers.py -q
```
