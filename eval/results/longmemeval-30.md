# Custodia - long-term memory scorecard

## Provenance

| field | value |
|---|---|
| run | `longmemeval-30` |
| kind | `longmemeval` |
| started at | `2026-08-20T15:11:21Z` |
| finished at | `2026-08-20T15:22:51Z` |
| dataset | `s` |
| dataset variant | `s` |
| dataset sha256 | `08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894` |
| dataset source url | `https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s` |
| dataset items available | `500` |
| sample size | `30` |
| seed | `0` |
| stratified | `True` |
| types | `all` |
| systems | `custodia, fullcontext, rag` |
| answer model | `gemini-3-7-flash` |
| judge model | `grok-4-5` |
| llm binding | `custodia.llm` |
| judge mode | `llm-judge` |

## Dataset

| field | value |
|---|---|
| instances | 30 |
| abstention items | 7 |
| abstention share of sample | 23.3% |
| sessions per instance (mean) | 49.2 |
| sessions per instance (min/max) | 44 / 57 |
| estimated tokens per instance (mean) | 122,121.3 |
| token estimator | chars/4 heuristic (no tokenizer dependency; comparable across systems, not exact) |

**Sample composition**

| question type | instances |
|---|---|
| knowledge-update | 7 |
| multi-session | 8 |
| single-session-assistant | 3 |
| single-session-preference | 1 |
| single-session-user | 4 |
| temporal-reasoning | 7 |

> in a sampled run this share may exceed the source dataset's share: sampling enforces a floor of 8 abstention questions so the abstention rates have a usable denominator, itself capped at 25% of the sample so a small run is not dominated by them

## Results

| system | accuracy | abstention recall | abstention precision | hallucination rate | over-refusal | mean prompt tokens | mean latency (ms) |
|---|---|---|---|---|---|---|---|
| custodia | 30.4% | 100.0% | 35.0% | 0.0% | 56.5% | 328.5 | 20,275.5 |
| fullcontext | 65.2% | 85.7% | 54.5% | 14.3% | 21.7% | 123,165.4 | 5,837.0 |
| rag | 56.5% | 100.0% | 50.0% | 0.0% | 30.4% | 13,118.9 | 4,040.4 |

### custodia

| metric | value |
|---|---|
| questions scored | 30 |
| answerable questions | 23 |
| accuracy (answerable only) | 30.4% |
| mean latency | 20,275.5 ms |
| mean prompt tokens (estimated) | 328.5 |
| truncated inputs | 0 |
| errors | 0 |
| judge failures | 0 |

**Accuracy by question type**

| question type | n | correct | accuracy |
|---|---|---|---|
| knowledge-update | 4 | 3 | 75.0% |
| multi-session | 6 | 1 | 16.7% |
| single-session-assistant | 3 | 1 | 33.3% |
| single-session-preference | 1 | 0 | 0.0% |
| single-session-user | 3 | 2 | 66.7% |
| temporal-reasoning | 6 | 0 | 0.0% |

**Abstention**

| metric | value | meaning |
|---|---|---|
| abstention items | 7 | questions whose answer is genuinely absent from memory |
| abstention recall | 100.0% | of those, the share the system declined |
| abstention precision | 35.0% | of everything it declined, the share that genuinely had no answer |
| hallucination rate | 0.0% | of the unanswerable questions, the share answered anyway and graded wrong |
| over-refusal rate | 56.5% | answerable questions the system declined (the cost of refusing) |

Graded by: `llm-judge` x30. Token counts use the chars/4 heuristic (no tokenizer dependency; comparable across systems, not exact).

### fullcontext

| metric | value |
|---|---|
| questions scored | 30 |
| answerable questions | 23 |
| accuracy (answerable only) | 65.2% |
| mean latency | 5,837.0 ms |
| mean prompt tokens (estimated) | 123,165.4 |
| truncated inputs | 8 |
| errors | 0 |
| judge failures | 0 |

**Accuracy by question type**

| question type | n | correct | accuracy |
|---|---|---|---|
| knowledge-update | 4 | 4 | 100.0% |
| multi-session | 6 | 2 | 33.3% |
| single-session-assistant | 3 | 3 | 100.0% |
| single-session-preference | 1 | 0 | 0.0% |
| single-session-user | 3 | 2 | 66.7% |
| temporal-reasoning | 6 | 4 | 66.7% |

**Abstention**

| metric | value | meaning |
|---|---|---|
| abstention items | 7 | questions whose answer is genuinely absent from memory |
| abstention recall | 85.7% | of those, the share the system declined |
| abstention precision | 54.5% | of everything it declined, the share that genuinely had no answer |
| hallucination rate | 14.3% | of the unanswerable questions, the share answered anyway and graded wrong |
| over-refusal rate | 21.7% | answerable questions the system declined (the cost of refusing) |

Graded by: `llm-judge` x30. Token counts use the chars/4 heuristic (no tokenizer dependency; comparable across systems, not exact).

### rag

| metric | value |
|---|---|
| questions scored | 30 |
| answerable questions | 23 |
| accuracy (answerable only) | 56.5% |
| mean latency | 4,040.4 ms |
| mean prompt tokens (estimated) | 13,118.9 |
| truncated inputs | 0 |
| errors | 0 |
| judge failures | 0 |

**Accuracy by question type**

| question type | n | correct | accuracy |
|---|---|---|---|
| knowledge-update | 4 | 3 | 75.0% |
| multi-session | 6 | 2 | 33.3% |
| single-session-assistant | 3 | 3 | 100.0% |
| single-session-preference | 1 | 0 | 0.0% |
| single-session-user | 3 | 2 | 66.7% |
| temporal-reasoning | 6 | 3 | 50.0% |

**Abstention**

| metric | value | meaning |
|---|---|---|
| abstention items | 7 | questions whose answer is genuinely absent from memory |
| abstention recall | 100.0% | of those, the share the system declined |
| abstention precision | 50.0% | of everything it declined, the share that genuinely had no answer |
| hallucination rate | 0.0% | of the unanswerable questions, the share answered anyway and graded wrong |
| over-refusal rate | 30.4% | answerable questions the system declined (the cost of refusing) |

Graded by: `llm-judge` x30. Token counts use the chars/4 heuristic (no tokenizer dependency; comparable across systems, not exact).

---

`not measured` above means exactly that: the run did not produce the number. It is never a stand-in for zero.
