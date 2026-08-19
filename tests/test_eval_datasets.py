"""Dataset normalisation and deterministic sampling.

No network, no model. The fixture below is shaped exactly like the real
LongMemEval file -- the same field names, the same parallel haystack arrays, the
same ``_abs`` id convention -- so a change to the loader that would break on the
real 278 MB download breaks here first, in milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.datasets import (  # noqa: E402
    EvalTurn,
    Instance,
    dataset_stats,
    normalize_beam,
    normalize_longmemeval,
    parse_lme_date,
    sample_instances,
)


def _record(
    qid: str,
    qtype: str,
    *,
    sessions: int = 3,
    answer: str = "Business Administration",
) -> dict:
    """One LongMemEval record, with the real field names."""
    dates = [f"2023/05/{20 + i:02d} (Sat) 0{i}:21" for i in range(sessions)]
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": "What degree did I graduate with?",
        "answer": answer,
        "question_date": "2023/05/30 (Tue) 23:40",
        "answer_session_ids": [f"{qid}-s0"],
        "haystack_dates": dates,
        "haystack_session_ids": [f"{qid}-s{i}" for i in range(sessions)],
        "haystack_sessions": [
            [
                {"role": "user", "content": f"session {i} user text about degrees"},
                {
                    "role": "assistant",
                    "content": f"session {i} assistant reply",
                    **({"has_answer": True} if i == 0 else {}),
                },
            ]
            for i in range(sessions)
        ],
    }


@pytest.fixture()
def raw_records() -> list[dict]:
    types = [
        "single-session-user",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
        "single-session-preference",
    ]
    records: list[dict] = []
    for index in range(40):
        records.append(_record(f"q{index:03d}", types[index % len(types)]))
    for index in range(6):
        records.append(
            _record(
                f"a{index:03d}_abs",
                types[index % len(types)],
                answer="The information provided is not enough.",
            )
        )
    return records


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #


def test_parse_date_ignores_the_weekday() -> None:
    """The weekday in a LongMemEval date is decorative and locale-hostile."""
    assert parse_lme_date("2023/05/30 (Tue) 23:40") == 1685490000
    assert parse_lme_date("2023/05/30 23:40") == 1685490000
    with pytest.raises(ValueError):
        parse_lme_date("last Tuesday")


def test_normalize_maps_the_real_field_names(raw_records: list[dict]) -> None:
    instances = normalize_longmemeval(raw_records)
    assert len(instances) == 46
    first = instances[0]
    assert isinstance(first, Instance)
    assert first.question == "What degree did I graduate with?"
    assert first.answer == "Business Administration"
    assert first.asked_at == parse_lme_date("2023/05/30 (Tue) 23:40")
    assert len(first.sessions) == 3
    assert [s.sid for s in first.sessions] == ["q000-s0", "q000-s1", "q000-s2"]
    assert all(isinstance(t, EvalTurn) for t in first.sessions[0].turns)


def test_has_answer_defaults_to_false_when_absent(raw_records: list[dict]) -> None:
    """Only evidence turns carry ``has_answer`` in the S haystacks."""
    instance = normalize_longmemeval(raw_records)[0]
    evidence = instance.evidence_turns()
    assert len(evidence) == 1
    assert evidence[0][0] == "q000-s0"
    assert instance.sessions[1].turns[1].has_answer is False


def test_sessions_are_ordered_by_time() -> None:
    record = _record("qX", "multi-session", sessions=3)
    record["haystack_dates"] = [
        "2023/05/25 (Thu) 03:00",
        "2023/05/20 (Sat) 03:00",
        "2023/05/22 (Mon) 03:00",
    ]
    instance = normalize_longmemeval([record])[0]
    assert [s.ts for s in instance.sessions] == sorted(s.ts for s in instance.sessions)


def test_mismatched_haystack_arrays_are_rejected() -> None:
    """Silently zipping arrays of different lengths would drop evidence."""
    record = _record("qY", "multi-session", sessions=3)
    record["haystack_dates"] = record["haystack_dates"][:2]
    with pytest.raises(ValueError, match="haystack arrays disagree"):
        normalize_longmemeval([record])


def test_abstention_is_read_off_the_id_suffix(raw_records: list[dict]) -> None:
    instances = normalize_longmemeval(raw_records)
    abstention = [i for i in instances if i.is_abstention]
    assert len(abstention) == 6
    assert all(i.qid.endswith("_abs") for i in abstention)
    assert not any(i.is_abstention for i in instances if not i.qid.endswith("_abs"))


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #


def test_sampling_is_deterministic_under_a_seed(raw_records: list[dict]) -> None:
    instances = normalize_longmemeval(raw_records)
    first = [i.qid for i in sample_instances(instances, limit=15, seed=7)]
    second = [i.qid for i in sample_instances(instances, limit=15, seed=7)]
    assert first == second
    assert len(first) == 15


def test_a_different_seed_gives_a_different_sample(raw_records: list[dict]) -> None:
    instances = normalize_longmemeval(raw_records)
    a = [i.qid for i in sample_instances(instances, limit=15, seed=1)]
    b = [i.qid for i in sample_instances(instances, limit=15, seed=2)]
    assert a != b


def test_input_order_cannot_leak_into_the_sample(raw_records: list[dict]) -> None:
    """Candidates are sorted by qid before any draw, so file order is irrelevant."""
    instances = normalize_longmemeval(raw_records)
    shuffled = list(reversed(instances))
    assert [i.qid for i in sample_instances(instances, limit=12, seed=3)] == [
        i.qid for i in sample_instances(shuffled, limit=12, seed=3)
    ]


def test_abstention_items_are_never_sampled_away(raw_records: list[dict]) -> None:
    """The headline metric must always have a denominator."""
    instances = normalize_longmemeval(raw_records)
    for limit in (5, 10, 20, 30, 45):
        picked = sample_instances(instances, limit=limit, seed=0)
        assert len(picked) == limit
        assert any(i.is_abstention for i in picked), f"no abstention items at limit={limit}"


def test_abstention_floor_is_respected_when_the_sample_allows_it(
    raw_records: list[dict],
) -> None:
    instances = normalize_longmemeval(raw_records)
    picked = sample_instances(instances, limit=20, seed=0, min_abstention=6)
    assert sum(1 for i in picked if i.is_abstention) == 6


def test_stratification_covers_every_question_type(raw_records: list[dict]) -> None:
    instances = normalize_longmemeval(raw_records)
    picked = sample_instances(instances, limit=25, seed=0)
    answerable_types = {i.qtype for i in instances if not i.is_abstention}
    assert {i.qtype for i in picked if not i.is_abstention} == answerable_types


def test_unstratified_sampling_still_returns_the_requested_count(
    raw_records: list[dict],
) -> None:
    instances = normalize_longmemeval(raw_records)
    picked = sample_instances(instances, limit=10, seed=0, stratify=False)
    assert len(picked) == 10


def test_type_filter_narrows_the_pool(raw_records: list[dict]) -> None:
    instances = normalize_longmemeval(raw_records)
    picked = sample_instances(instances, limit=None, types=["multi-session"])
    assert picked and {i.qtype for i in picked} == {"multi-session"}


def test_limit_beyond_the_dataset_returns_everything(raw_records: list[dict]) -> None:
    instances = normalize_longmemeval(raw_records)
    assert len(sample_instances(instances, limit=10_000, seed=0)) == len(instances)


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #


def test_dataset_stats_counts_what_it_claims(raw_records: list[dict]) -> None:
    stats = dataset_stats(normalize_longmemeval(raw_records))
    assert stats["instances"] == 46
    assert stats["abstention"]["items"] == 6
    assert stats["sessions"]["mean"] == 3.0
    assert stats["turns"]["total"] == 46 * 6
    assert "chars/4" in stats["tokens_estimated"]["estimator"]
    assert sum(stats["by_type"].values()) == 46
    assert stats["evidence_turns_labelled"] == 46


def test_dataset_stats_on_nothing_reports_nothing() -> None:
    stats = dataset_stats([])
    assert stats["instances"] == 0
    assert stats["abstention"]["share"] is None


# --------------------------------------------------------------------------- #
# BEAM adapter
# --------------------------------------------------------------------------- #


def test_beam_probing_questions_parse_from_a_python_literal() -> None:
    """BEAM stores ``probing_questions`` as a Python literal, not as JSON."""
    row = {
        "conversation_id": "7",
        "chat": [
            [
                {"role": "user", "content": "hello", "time_anchor": "March-15-2024"},
                {"role": "assistant", "content": "hi"},
            ]
        ],
        "probing_questions": (
            "{'abstention': [{'question': 'What is my dog called?', "
            "'ideal_response': 'There is no information about a dog.'}], "
            "'temporal_reasoning': [{'question': 'When did I start?', "
            "'ideal_response': 'March 2024'}]}"
        ),
    }
    instances = normalize_beam([row], split="100K")
    assert len(instances) == 2
    by_type = {i.qtype: i for i in instances}
    assert by_type["abstention"].is_abstention is True
    assert by_type["temporal_reasoning"].is_abstention is False
    assert by_type["abstention"].qid.endswith("_abs")
    assert len(by_type["abstention"].sessions) == 1
