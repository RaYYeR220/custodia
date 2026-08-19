"""Grading maths, on hand-built records.

The abstention arithmetic is the part worth pinning down. Precision, recall and
hallucination rate are easy to define plausibly and wrongly, and every wrong
definition flatters the system. The cases below fix the intended reading, and the
``not measured`` cases fix the other half of the contract: an empty denominator
must never print as a zero that reads like a good score.

No network and no provider: the judge is exercised against fakes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.report import NOT_MEASURED, build_report, pct  # noqa: E402
from eval.scorers import (  # noqa: E402
    RunRecord,
    contains_answer,
    fallback_judge,
    judge,
    looks_like_abstention,
    normalize_text,
    score_run,
    score_systems,
)


class FakeLLM:
    """Returns canned replies in order and records the prompts it was given."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str, **_: object) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


class ExplodingLLM:
    def complete(self, prompt: str, **_: object) -> str:
        raise RuntimeError("provider is down")


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #


def test_normalisation_drops_case_punctuation_and_articles() -> None:
    assert normalize_text("The  Fitness-First, Gym!") == "fitness first gym"


def test_containment_is_one_directional() -> None:
    """A prediction that is merely a fragment of the gold answer is not correct."""
    assert contains_answer("You graduated in Business Administration.", "Business Administration")
    assert not contains_answer("Business", "Business Administration")


@pytest.mark.parametrize(
    "text",
    [
        "I don't have that information.",
        "The information provided is not enough.",
        "You never mentioned buying cows.",
        "Based on the provided chat, there is no information related to that.",
        "I cannot determine this from our conversations.",
        "",
    ],
)
def test_abstention_phrasings_are_recognised(text: str) -> None:
    assert looks_like_abstention(text) is True


@pytest.mark.parametrize(
    "text",
    ["You graduated with a degree in Business Administration.", "Your gym is Fitness First."],
)
def test_substantive_answers_are_not_abstentions(text: str) -> None:
    assert looks_like_abstention(text) is False


# --------------------------------------------------------------------------- #
# judges
# --------------------------------------------------------------------------- #


def test_llm_judge_returns_a_structured_verdict() -> None:
    llm = FakeLLM('{"correct": true, "reason": "same degree"}')
    verdict = judge("q?", "Business Administration", "You studied Business Administration.",
                    "single-session-user", llm=llm)
    assert verdict.correct is True
    assert verdict.method == "llm-judge"
    assert verdict.is_fallback is False


def test_llm_judge_tolerates_a_fenced_object() -> None:
    llm = FakeLLM('```json\n{"correct": false, "reason": "wrong date"}\n```')
    verdict = judge("q?", "March 2023", "April 2023", "temporal-reasoning", llm=llm)
    assert verdict.correct is False
    assert "wrong date" in verdict.reason


def test_the_rubric_matches_the_question_type() -> None:
    llm = FakeLLM('{"correct": true, "reason": "ok"}')
    judge("q?", "gold", "pred", "temporal-reasoning", llm=llm)
    assert "temporal question" in llm.prompts[0]

    llm = FakeLLM('{"correct": true, "reason": "ok"}')
    judge("q?", "gold", "pred", "single-session-preference", llm=llm)
    assert "Accept paraphrase" in llm.prompts[0]


def test_an_abstention_item_gets_the_abstention_rubric() -> None:
    llm = FakeLLM('{"correct": true, "reason": "declined"}')
    judge("q?", "not enough information", "I don't know.", "abstention", llm=llm)
    assert "ONLY if it declines" in llm.prompts[0]


def test_an_ungradeable_item_is_not_a_passing_item() -> None:
    """Two unparseable replies resolve to incorrect *and* a visible failure mark."""
    llm = FakeLLM("I think it's probably fine", "yeah looks right")
    verdict = judge("q?", "gold", "pred", "multi-session", llm=llm)
    assert verdict.correct is False
    assert verdict.method == "llm-judge-failed"


def test_a_provider_error_is_recorded_not_swallowed() -> None:
    verdict = judge("q?", "gold", "pred", "multi-session", llm=ExplodingLLM())
    assert verdict.correct is False
    assert verdict.method == "llm-judge-failed"
    assert "provider is down" in verdict.reason


def test_fallback_judge_is_labelled_as_weaker() -> None:
    verdict = fallback_judge("q?", "Business Administration",
                             "You studied Business Administration.", "single-session-user")
    assert verdict.correct is True
    assert verdict.method == "lexical-fallback"
    assert verdict.is_fallback is True


def test_fallback_judge_matches_numbers_and_dates() -> None:
    assert fallback_judge("q?", "March 5, 2024", "It was 2024/03/05.", "temporal-reasoning").correct
    assert fallback_judge("q?", "3 miles", "You ran 3 miles.", "single-session-user").correct
    assert not fallback_judge("q?", "3 miles", "You ran 8 miles.", "single-session-user").correct


def test_fallback_judge_treats_abstention_items_correctly() -> None:
    assert fallback_judge("q?", "not enough info", "I don't know.", "abstention").correct
    assert not fallback_judge("q?", "not enough info", "It was Fitness First.", "abstention").correct


def test_fallback_judge_penalises_refusing_an_answerable_question() -> None:
    verdict = fallback_judge("q?", "Fitness First", "I don't have that information.",
                             "single-session-user")
    assert verdict.correct is False


# --------------------------------------------------------------------------- #
# abstention arithmetic
# --------------------------------------------------------------------------- #


def _records() -> list[RunRecord]:
    """Six answerable + four unanswerable, with every interesting combination."""
    return [
        # answerable, answered, right
        RunRecord("q1", "custodia", "single-session-user", False, correct=True,
                  latency_ms=100, prompt_tokens=500),
        RunRecord("q2", "custodia", "single-session-user", False, correct=True,
                  latency_ms=200, prompt_tokens=700),
        # answerable, answered, wrong
        RunRecord("q3", "custodia", "multi-session", False, correct=False,
                  latency_ms=300, prompt_tokens=900),
        # answerable, wrongly declined -> over-refusal
        RunRecord("q4", "custodia", "multi-session", False, abstained=True, correct=False,
                  latency_ms=400, prompt_tokens=100),
        RunRecord("q5", "custodia", "temporal-reasoning", False, correct=True,
                  latency_ms=100, prompt_tokens=100),
        RunRecord("q6", "custodia", "temporal-reasoning", False, correct=False,
                  latency_ms=100, prompt_tokens=100),
        # unanswerable, correctly declined
        RunRecord("a1_abs", "custodia", "multi-session", True, abstained=True, correct=True),
        RunRecord("a2_abs", "custodia", "multi-session", True, abstained=True, correct=True),
        # unanswerable, answered anyway and graded wrong -> hallucination
        RunRecord("a3_abs", "custodia", "temporal-reasoning", True, abstained=False, correct=False),
        # unanswerable, answered without declining but the judge accepted it: counts
        # against abstention recall, but is not a hallucination
        RunRecord("a4_abs", "custodia", "temporal-reasoning", True, abstained=False, correct=True),
    ]


def test_accuracy_covers_answerable_questions_only() -> None:
    """Otherwise a refuse-everything system borrows credit from the abstention set."""
    card = score_run(_records())
    assert card.answerable == 6
    assert card.abstention_items == 4
    assert card.correct == 3
    assert card.accuracy == 0.5


def test_abstention_recall_precision_and_hallucination() -> None:
    card = score_run(_records())
    # 2 of 4 unanswerable questions were declined
    assert card.abstention_recall == 0.5
    # 3 declines in total, 2 of them on genuinely unanswerable questions
    assert card.abstention_precision == pytest.approx(2 / 3, abs=1e-4)
    # 1 of 4 unanswerable questions was answered anyway *and* graded wrong
    assert card.hallucination_rate == 0.25
    # 1 of 6 answerable questions was declined
    assert card.over_refusal_rate == pytest.approx(1 / 6, abs=1e-4)


def test_hallucination_rate_never_exceeds_one_minus_recall() -> None:
    card = score_run(_records())
    assert card.hallucination_rate <= 1 - card.abstention_recall


def test_a_refuse_everything_system_is_visibly_bad() -> None:
    """Perfect abstention recall, and an over-refusal rate that gives it away."""
    rows = [
        RunRecord("q1", "coward", "multi-session", False, abstained=True, correct=False),
        RunRecord("q2", "coward", "multi-session", False, abstained=True, correct=False),
        RunRecord("a1_abs", "coward", "multi-session", True, abstained=True, correct=True),
    ]
    card = score_run(rows)
    assert card.abstention_recall == 1.0
    assert card.hallucination_rate == 0.0
    assert card.accuracy == 0.0
    assert card.over_refusal_rate == 1.0
    assert card.abstention_precision == pytest.approx(1 / 3, abs=1e-4)


def test_an_answer_everything_system_is_visibly_bad() -> None:
    rows = [
        RunRecord("q1", "bluffer", "multi-session", False, correct=True),
        RunRecord("a1_abs", "bluffer", "multi-session", True, correct=False),
        RunRecord("a2_abs", "bluffer", "multi-session", True, correct=False),
    ]
    card = score_run(rows)
    assert card.abstention_recall == 0.0
    assert card.hallucination_rate == 1.0
    assert card.abstention_precision is None  # it declined nothing


def test_metrics_without_a_denominator_are_none_not_zero() -> None:
    rows = [RunRecord("q1", "s", "multi-session", False, correct=True)]
    card = score_run(rows)
    assert card.abstention_recall is None
    assert card.hallucination_rate is None
    assert card.abstention_precision is None
    assert pct(card.abstention_recall) == NOT_MEASURED


def test_per_type_accuracy_splits_by_question_type() -> None:
    card = score_run(_records())
    assert card.accuracy_by_type["single-session-user"] == {"n": 2, "correct": 2, "accuracy": 1.0}
    assert card.accuracy_by_type["multi-session"]["accuracy"] == 0.0
    assert "abstention" not in card.accuracy_by_type


def test_means_and_counters_are_measured() -> None:
    card = score_run(_records())
    assert card.mean_latency_ms == pytest.approx(200.0, abs=0.1)
    assert card.mean_prompt_tokens == pytest.approx(400.0, abs=0.1)
    assert card.judge_failures == 0
    assert card.errors == 0


def test_judge_failures_are_surfaced() -> None:
    rows = [
        RunRecord("q1", "s", "multi-session", False, judge_method="llm-judge-failed"),
        RunRecord("q2", "s", "multi-session", False, judge_method="llm-judge", correct=True),
    ]
    card = score_run(rows)
    assert card.judge_failures == 1
    assert card.judge_methods == {"llm-judge": 1, "llm-judge-failed": 1}


def test_records_round_trip_through_a_dict() -> None:
    original = _records()[0]
    assert RunRecord.from_dict(original.as_dict()) == original


def test_score_systems_splits_by_system() -> None:
    rows = _records() + [RunRecord("q1", "rag", "multi-session", False, correct=False)]
    cards = score_systems([r.as_dict() for r in rows])
    assert set(cards) == {"custodia", "rag"}
    assert cards["rag"].n == 1


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def test_markdown_prints_not_measured_rather_than_a_fake_zero() -> None:
    card = score_run([RunRecord("q1", "s", "multi-session", False, correct=True)])
    markdown = card.as_markdown()
    assert NOT_MEASURED in markdown
    assert "abstention recall" in markdown


def test_report_carries_its_provenance() -> None:
    document = {
        "provenance": {
            "kind": "longmemeval",
            "dataset_sha256": "08d8dad4",
            "sample_size": 50,
            "seed": 0,
            "answer_model": "some-model",
            "judge_mode": "llm-judge",
        },
        "scorecards": {"custodia": score_run(_records()).as_json()},
        "dataset_stats": {"instances": 50, "abstention": {"items": 8, "share": 0.16}},
    }
    text = build_report(document)
    assert "08d8dad4" in text
    assert "sample size" in text
    assert "hallucination rate" in text
    assert "not measured" in text  # the closing note explains the convention
