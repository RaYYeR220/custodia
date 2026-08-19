# Custodia - long-term memory scorecard

## Provenance

| field | value |
|---|---|
| run | `longmemeval-50` |
| kind | `longmemeval` |
| started at | `2026-08-19T18:24:29Z` |
| finished at | `2026-08-19T20:17:56Z` |
| dataset | `s` |
| dataset variant | `s` |
| dataset sha256 | `08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894` |
| dataset source url | `https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s` |
| dataset items available | `500` |
| sample size | `50` |
| seed | `0` |
| stratified | `True` |
| types | `all` |
| systems | `custodia, fullcontext, rag` |
| answer model | `gemini-3-7-flash` |
| judge model | `gemini-3-7-flash` |
| llm binding | `custodia.llm` |
| judge mode | `llm-judge` |

## Dataset

| field | value |
|---|---|
| instances | 50 |
| abstention items | 8 |
| abstention share of sample | 16.0% |
| sessions per instance (mean) | 49.6 |
| sessions per instance (min/max) | 43 / 60 |
| estimated tokens per instance (mean) | 121,953.2 |
| token estimator | chars/4 heuristic (no tokenizer dependency; comparable across systems, not exact) |

**Sample composition**

| question type | instances |
|---|---|
| knowledge-update | 9 |
| multi-session | 13 |
| single-session-assistant | 5 |
| single-session-preference | 3 |
| single-session-user | 7 |
| temporal-reasoning | 13 |

> in a sampled run this share may exceed the source dataset's share: sampling enforces a floor of 8 abstention questions so the abstention rates have a usable denominator, itself capped at 25% of the sample so a small run is not dominated by them

## Results

| system | accuracy | abstention recall | abstention precision | hallucination rate | over-refusal | mean prompt tokens | mean latency (ms) |
|---|---|---|---|---|---|---|---|
| custodia | 61.9% | 87.5% | 35.0% | 12.5% | 30.9% | 260.1 | 16,079.4 |
| fullcontext | 69.0% | 87.5% | 70.0% | 12.5% | 7.1% | 123,090.6 | 10,119.2 |
| rag | 64.3% | 100.0% | 47.1% | 0.0% | 21.4% | 13,144.0 | 6,157.2 |

### custodia

| metric | value |
|---|---|
| questions scored | 50 |
| answerable questions | 42 |
| accuracy (answerable only) | 61.9% |
| mean latency | 16,079.4 ms |
| mean prompt tokens (estimated) | 260.1 |
| truncated inputs | 0 |
| errors | 0 |
| judge failures | 0 |

**Accuracy by question type**

| question type | n | correct | accuracy |
|---|---|---|---|
| knowledge-update | 6 | 4 | 66.7% |
| multi-session | 11 | 8 | 72.7% |
| single-session-assistant | 5 | 0 | 0.0% |
| single-session-preference | 3 | 0 | 0.0% |
| single-session-user | 6 | 6 | 100.0% |
| temporal-reasoning | 11 | 8 | 72.7% |

**Abstention**

| metric | value | meaning |
|---|---|---|
| abstention items | 8 | questions whose answer is genuinely absent from memory |
| abstention recall | 87.5% | of those, the share the system declined |
| abstention precision | 35.0% | of everything it declined, the share that genuinely had no answer |
| hallucination rate | 12.5% | of the unanswerable questions, the share answered anyway and graded wrong |
| over-refusal rate | 30.9% | answerable questions the system declined (the cost of refusing) |

Graded by: `llm-judge` x50. Token counts use the chars/4 heuristic (no tokenizer dependency; comparable across systems, not exact).

### fullcontext

| metric | value |
|---|---|
| questions scored | 50 |
| answerable questions | 42 |
| accuracy (answerable only) | 69.0% |
| mean latency | 10,119.2 ms |
| mean prompt tokens (estimated) | 123,090.6 |
| truncated inputs | 10 |
| errors | 0 |
| judge failures | 0 |

**Accuracy by question type**

| question type | n | correct | accuracy |
|---|---|---|---|
| knowledge-update | 6 | 6 | 100.0% |
| multi-session | 11 | 7 | 63.6% |
| single-session-assistant | 5 | 5 | 100.0% |
| single-session-preference | 3 | 1 | 33.3% |
| single-session-user | 6 | 6 | 100.0% |
| temporal-reasoning | 11 | 4 | 36.4% |

**Abstention**

| metric | value | meaning |
|---|---|---|
| abstention items | 8 | questions whose answer is genuinely absent from memory |
| abstention recall | 87.5% | of those, the share the system declined |
| abstention precision | 70.0% | of everything it declined, the share that genuinely had no answer |
| hallucination rate | 12.5% | of the unanswerable questions, the share answered anyway and graded wrong |
| over-refusal rate | 7.1% | answerable questions the system declined (the cost of refusing) |

Graded by: `llm-judge` x50. Token counts use the chars/4 heuristic (no tokenizer dependency; comparable across systems, not exact).

### rag

| metric | value |
|---|---|
| questions scored | 50 |
| answerable questions | 42 |
| accuracy (answerable only) | 64.3% |
| mean latency | 6,157.2 ms |
| mean prompt tokens (estimated) | 13,144.0 |
| truncated inputs | 0 |
| errors | 0 |
| judge failures | 0 |

**Accuracy by question type**

| question type | n | correct | accuracy |
|---|---|---|---|
| knowledge-update | 6 | 6 | 100.0% |
| multi-session | 11 | 7 | 63.6% |
| single-session-assistant | 5 | 5 | 100.0% |
| single-session-preference | 3 | 1 | 33.3% |
| single-session-user | 6 | 6 | 100.0% |
| temporal-reasoning | 11 | 2 | 18.2% |

**Abstention**

| metric | value | meaning |
|---|---|---|
| abstention items | 8 | questions whose answer is genuinely absent from memory |
| abstention recall | 100.0% | of those, the share the system declined |
| abstention precision | 47.1% | of everything it declined, the share that genuinely had no answer |
| hallucination rate | 0.0% | of the unanswerable questions, the share answered anyway and graded wrong |
| over-refusal rate | 21.4% | answerable questions the system declined (the cost of refusing) |

Graded by: `llm-judge` x50. Token counts use the chars/4 heuristic (no tokenizer dependency; comparable across systems, not exact).

---

`not measured` above means exactly that: the run did not produce the number. It is never a stand-in for zero.
