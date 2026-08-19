"""The gate. Every abstention branch, driven by a stub model.

The claim under test is narrow: no reply from a language model, however
confident, well-formed or insistent, can produce an answer that the warrant does
not justify. So each test here breaks exactly one thing and asserts the same
outcome - `answered is False`, with the shared reason string the surfaces render
and the specific check that failed at the end of the trail.

The deterministic answerer gets the same treatment from the other side: it is
only allowed to speak when no model is reachable *and* the warrant has already
made the choice for it, so most of its tests are about it staying quiet.
"""

from __future__ import annotations

import pytest

from custodia import config, gate
from custodia.gate import Gate, Verdict
from custodia.retrieve import Warrant
from custodia.schema import Evidence

# ---- doubles --------------------------------------------------------------- #


class StubLLM:
    """Replies handed out in order. A reply may be a dict or an exception."""

    def __init__(self, *replies, enabled: bool = True) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def json(self, messages, *, model=None, temperature=0.0, max_tokens=2048):
        self.calls.append(messages)
        reply = self.replies.pop(0) if self.replies else {}
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def chat(self, messages, *, model=None, temperature=0.0, max_tokens=2048):
        raise AssertionError("the gate must go through json(), not chat()")


class StubRetriever:
    """Stands in for the retriever so gate tests need no graph."""

    def __init__(self, warrant: Warrant, settings: config.Settings) -> None:
        self._warrant = warrant
        self.settings = settings
        self.asked: list[tuple[str, int | None]] = []

    def warrant(self, question: str, *, as_of: int | None = None, k: int | None = None):
        self.asked.append((question, as_of))
        return self._warrant


class RecordingAuditor:
    def __init__(self, *, explode: bool = False) -> None:
        self.records: list[tuple[str, Warrant, Verdict]] = []
        self.explode = explode

    def record(self, question, warrant, verdict) -> int:
        if self.explode:
            raise RuntimeError("graph unavailable")
        self.records.append((question, warrant, verdict))
        return 1


def evidence(fid: int, text: str, **kw) -> Evidence:
    base = dict(
        tier="owner",
        status="active",
        valid_from=1_700_000_000,
        valid_to=0,
        sid="s1",
        sidx=0,
        tidx=0,
        turn_text="I switched to Northline Fitness last week.",
        turn_ts=1_700_000_000,
        score=0.8,
        hops=1,
        path=["Entity:gym", "MENTIONS", f"Fact:{fid}"],
    )
    base.update(kw)
    return Evidence(fid=fid, text=text, **base)


def warrant_of(*items: Evidence, **kw) -> Warrant:
    return Warrant(
        question=kw.pop("question", "Which gym does the user go to?"),
        asked_at=1_700_000_100,
        as_of=kw.pop("as_of", None),
        evidence=list(items),
        seeds=kw.pop("seeds", {"entities": ["gym", "northline"], "terms": ["gym"]}),
        paths_examined=kw.pop("paths_examined", 12),
        facts_considered=kw.pop("facts_considered", 4),
        quarantined_seen=kw.pop("quarantined_seen", 0),
    )


GYM = 4_611_686_018_427_387_903
MILK = 1_234_567_890_123_456_789
INVENTED = 999_999_999_999

GYM_TEXT = "The user's gym is Northline Fitness."


def decisive() -> Warrant:
    """One clearly-ahead fact: the deterministic answerer may quote it."""
    return warrant_of(evidence(GYM, GYM_TEXT, score=0.82))


def ambiguous() -> Warrant:
    """Two facts within the margin: nothing but reading can choose between them."""
    return warrant_of(
        evidence(GYM, GYM_TEXT, score=0.80),
        evidence(MILK, "The user drinks oat milk.", score=0.78),
    )


@pytest.fixture()
def settings() -> config.Settings:
    s = config.Settings()
    s.verify_citations = False
    s.answer_model = "stub/answerer"
    return s


def build(warrant: Warrant, llm, settings, **kw) -> Gate:
    return Gate(StubRetriever(warrant, settings), llm=llm, settings=settings, **kw)


def good_reply(*citations: int) -> dict:
    return {
        "answer": "The user goes to Northline Fitness.",
        "citations": list(citations),
        "sufficient": True,
    }


# ---- abstention branches --------------------------------------------------- #


def test_empty_warrant_never_reaches_the_model(settings):
    llm = StubLLM(good_reply(GYM))
    verdict = build(warrant_of(), llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.NO_EVIDENCE
    assert verdict.checks == [gate.CHECK_WARRANT]
    assert llm.calls == []


def test_a_non_mapping_reply_abstains(settings):
    llm = StubLLM(["not", "an", "object"])
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.MALFORMED_RESPONSE
    assert verdict.checks[-1] == gate.CHECK_JSON


def test_a_reply_missing_required_keys_abstains(settings):
    llm = StubLLM({"answer": "Northline.", "sufficient": True})  # no citations key
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)

    assert verdict.abstained_because == gate.MALFORMED_RESPONSE
    assert verdict.checks[-1] == gate.CHECK_SCHEMA


def test_sufficient_false_short_circuits(settings):
    llm = StubLLM({"answer": "Probably Northline.", "citations": [GYM], "sufficient": False})
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.INSUFFICIENT
    assert verdict.checks[-1] == gate.CHECK_SUFFICIENT
    assert "Probably Northline" not in verdict.answer


def test_a_truthy_but_non_boolean_sufficient_is_not_accepted(settings):
    llm = StubLLM({"answer": "Northline.", "citations": [GYM], "sufficient": "yes"})
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)
    assert verdict.checks[-1] == gate.CHECK_SUFFICIENT


def test_an_empty_citation_list_abstains(settings):
    llm = StubLLM(good_reply())
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)

    assert verdict.abstained_because == gate.NO_CITATIONS
    assert verdict.checks[-1] == gate.CHECK_CITED


def test_a_hallucinated_citation_id_abstains(settings):
    llm = StubLLM(good_reply(INVENTED))
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.INVENTED_CITATION
    assert verdict.checks[-1] == gate.CHECK_IN_WARRANT
    assert verdict.citations == []
    assert "Northline" not in verdict.answer


def test_one_invented_id_among_real_ones_discards_the_whole_answer(settings):
    llm = StubLLM(good_reply(GYM, INVENTED))
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)
    assert verdict.abstained_because == gate.INVENTED_CITATION


def test_citations_that_are_not_ids_at_all_abstain(settings):
    llm = StubLLM({"answer": "Northline.", "citations": ["fact one"], "sufficient": True})
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)
    assert verdict.abstained_because == gate.NO_CITATIONS


def test_a_citation_id_sent_as_a_string_is_accepted(settings):
    llm = StubLLM({"answer": "Northline.", "citations": [str(GYM)], "sufficient": True})
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)

    assert verdict.answered is True
    assert verdict.citations == [GYM]


def test_an_empty_answer_abstains(settings):
    llm = StubLLM({"answer": "   ", "citations": [GYM], "sufficient": True})
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)

    assert verdict.abstained_because == gate.MALFORMED_RESPONSE
    assert verdict.checks[-1] == gate.CHECK_TEXT


@pytest.mark.parametrize(
    "text",
    [
        "I don't know which gym the user goes to.",
        "I do not have information about that.",
        "There is no record of the user's gym.",
        "Insufficient evidence to answer.",
        "Unable to determine the user's gym.",
        "I have no idea.",
    ],
)
def test_an_answer_that_is_itself_a_refusal_abstains(settings, text):
    llm = StubLLM({"answer": text, "citations": [GYM], "sufficient": True})
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.INSUFFICIENT
    assert verdict.checks[-1] == gate.CHECK_NOT_REFUSAL


def test_a_hedged_but_real_answer_is_not_mistaken_for_a_refusal(settings):
    llm = StubLLM(
        {
            "answer": "I don't know the exact date, but the gym is Northline Fitness.",
            "citations": [GYM],
            "sufficient": True,
        }
    )
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)
    assert verdict.answered is False  # opening clause is a refusal, so it is caught

    llm = StubLLM(
        {
            "answer": "The gym is Northline Fitness, though I don't know when they joined.",
            "citations": [GYM],
            "sufficient": True,
        }
    )
    verdict = build(ambiguous(), llm, settings).ask("q", record=False)
    assert verdict.answered is True


def test_a_timeout_abstains(settings):
    llm = StubLLM(TimeoutError("read timed out"))
    verdict = build(decisive(), llm, settings).ask("q", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.MODEL_UNAVAILABLE
    assert verdict.checks[-1] == gate.REASON_TIMEOUT


def test_an_unexpected_transport_error_abstains(settings):
    llm = StubLLM(ConnectionResetError("peer reset"))
    verdict = build(decisive(), llm, settings).ask("q", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.MODEL_UNAVAILABLE
    assert verdict.checks[-1] == gate.REASON_ERROR


# ---- the happy path -------------------------------------------------------- #


def test_a_warranted_answer_is_served(settings):
    llm = StubLLM(good_reply(GYM))
    verdict = build(decisive(), llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert verdict.answer == "The user goes to Northline Fitness."
    assert verdict.citations == [GYM]
    assert verdict.abstained_because == ""
    assert verdict.latency_ms >= 0
    assert verdict.model == "stub/answerer"
    assert gate.CHECK_IN_WARRANT in verdict.checks


def test_the_model_sees_the_warrant_and_not_the_conversation(settings):
    llm = StubLLM(good_reply(GYM))
    build(decisive(), llm, settings).ask("Which gym?", record=False)

    prompt = llm.calls[0][-1]["content"]
    assert str(GYM) in prompt
    assert GYM_TEXT in prompt
    assert "I switched to Northline Fitness last week." in prompt  # its own snippet only
    assert "conversation" not in prompt.lower()


def test_as_of_is_passed_through_to_retrieval(settings):
    warrant = warrant_of(evidence(GYM, "Northline."), as_of=1_600_000_000)
    retriever = StubRetriever(warrant, settings)
    Gate(retriever, llm=StubLLM(good_reply(GYM)), settings=settings).ask(
        "Which gym in March?", as_of=1_600_000_000, record=False
    )
    assert retriever.asked == [("Which gym in March?", 1_600_000_000)]


# ---- the second pass ------------------------------------------------------- #


def test_the_verifier_confirms_a_good_citation(settings):
    settings.verify_citations = True
    llm = StubLLM(good_reply(GYM), {"supports": True})
    verdict = build(decisive(), llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert verdict.verified == 1
    assert gate.CHECK_SUPPORTED in verdict.checks


def test_the_verifier_drops_a_citation_that_does_not_support_the_answer(settings):
    settings.verify_citations = True
    llm = StubLLM(good_reply(GYM, MILK), {"supports": True}, {"supports": False})
    verdict = build(ambiguous(), llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert verdict.citations == [GYM]
    assert verdict.verified == 1


def test_the_verifier_rejecting_every_citation_abstains(settings):
    settings.verify_citations = True
    warrant = warrant_of(evidence(GYM, "The user drinks oat milk."))
    llm = StubLLM(good_reply(GYM), {"supports": False})
    verdict = build(warrant, llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.UNVERIFIED_CITATION
    assert verdict.checks[-1] == gate.CHECK_SUPPORTED
    assert verdict.verified == 0


def test_a_verifier_that_errors_drops_the_citation_rather_than_trusting_it(settings):
    from custodia.llm import LLMUnavailable

    settings.verify_citations = True
    llm = StubLLM(good_reply(GYM), LLMUnavailable("provider down"))
    verdict = build(decisive(), llm, settings).ask("Which gym?", record=False)

    # the model already answered, so the deterministic path may not rescue it
    assert verdict.answered is False
    assert verdict.abstained_because == gate.UNVERIFIED_CITATION


def test_verification_is_skipped_when_disabled(settings):
    settings.verify_citations = False
    llm = StubLLM(good_reply(GYM))
    verdict = build(decisive(), llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert gate.CHECK_SUPPORTED not in verdict.checks
    assert len(llm.calls) == 1


# ---- the deterministic answerer -------------------------------------------- #


def test_with_no_model_a_decisive_warrant_is_quoted_verbatim(settings):
    verdict = build(decisive(), None, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert verdict.answer == GYM_TEXT  # exactly what the graph holds, nothing added
    assert verdict.citations == [GYM]
    assert verdict.model == gate.EXTRACTIVE_MODEL
    assert verdict.verified == 0
    assert gate.CHECK_EXTRACTIVE in verdict.checks


def test_a_disabled_model_takes_the_same_path(settings):
    llm = StubLLM(good_reply(GYM), enabled=False)
    verdict = build(decisive(), llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert verdict.model == gate.EXTRACTIVE_MODEL
    assert llm.calls == []


def test_a_provider_that_cannot_complete_falls_back_to_quoting(settings):
    from custodia.llm import LLMUnavailable

    llm = StubLLM(LLMUnavailable("cache miss under cache_only"))
    verdict = build(decisive(), llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert verdict.answer == GYM_TEXT
    assert verdict.model == gate.EXTRACTIVE_MODEL
    assert verdict.checks[-1] == gate.CHECK_EXTRACTIVE


def test_two_close_facts_are_not_quoted(settings):
    verdict = build(ambiguous(), None, settings).ask("Which gym?", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.MODEL_UNAVAILABLE
    assert gate.CHECK_EXTRACTIVE in verdict.checks
    assert verdict.citations == []


def test_weak_evidence_is_not_quoted_even_when_it_stands_alone(settings):
    weak = warrant_of(evidence(GYM, GYM_TEXT, score=gate.EXTRACTIVE_FLOOR - 0.01))
    verdict = build(weak, None, settings).ask("Which gym?", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.MODEL_UNAVAILABLE


def test_a_clear_winner_beside_an_unrelated_fact_is_still_quoted(settings):
    warrant = warrant_of(
        evidence(GYM, GYM_TEXT, score=0.82),
        evidence(MILK, "The user's dog is called Pip.", score=0.50),
    )
    verdict = build(warrant, None, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert verdict.answer == GYM_TEXT


def test_the_top_fact_is_not_quoted_when_it_is_off_the_subject(settings):
    """Ranking first is not the same as being the answer to this question."""
    warrant = warrant_of(
        evidence(GYM, GYM_TEXT, score=0.82),
        evidence(MILK, "The user's dog is called Pip.", score=0.50),
        question="Who is the user's dentist?",
    )
    verdict = build(warrant, None, settings).ask("Who is the user's dentist?", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.MODEL_UNAVAILABLE
    assert gate.CHECK_EXTRACTIVE in verdict.checks


def test_a_question_of_only_stopwords_is_never_quoted(settings):
    warrant = warrant_of(evidence(GYM, GYM_TEXT, score=0.90), question="what about it")
    verdict = build(warrant, None, settings).ask("what about it", record=False)
    assert verdict.answered is False


def test_the_deterministic_floor_is_stricter_than_the_warrant_floor(settings):
    assert gate.EXTRACTIVE_FLOOR > settings.evidence_floor


def test_an_empty_warrant_is_never_quoted(settings):
    verdict = build(warrant_of(), None, settings).ask("Which gym?", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.NO_EVIDENCE
    assert gate.CHECK_EXTRACTIVE not in verdict.checks


def test_the_deterministic_answerer_can_be_turned_off(settings):
    verdict = build(decisive(), None, settings, extractive=False).ask("q", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.MODEL_UNAVAILABLE
    assert gate.CHECK_EXTRACTIVE not in verdict.checks


def test_quoting_never_rescues_a_model_that_did_answer(settings):
    """Every model-path failure still abstains, decisive warrant or not."""
    failures = [
        StubLLM(good_reply(INVENTED)),
        StubLLM({"answer": "Northline.", "citations": [GYM], "sufficient": False}),
        StubLLM({"nonsense": True}),
        StubLLM(["not a mapping"]),
        StubLLM({"answer": "", "citations": [GYM], "sufficient": True}),
    ]
    for llm in failures:
        verdict = build(decisive(), llm, settings).ask("Which gym?", record=False)
        assert verdict.answered is False, llm.calls
        assert verdict.model == "stub/answerer"


# ---- the refusal a person reads -------------------------------------------- #


def test_the_abstention_says_what_was_searched_and_what_was_missing(settings):
    warrant = warrant_of(evidence(GYM, "Northline."), quarantined_seen=2)
    llm = StubLLM({"answer": "no", "citations": [], "sufficient": False})
    verdict = build(warrant, llm, settings).ask("Which gym?", record=False)

    assert "searched" in verdict.answer
    assert "'gym'" in verdict.answer
    assert "1 related fact" in verdict.answer
    assert "2 retrieved items were refused as untrusted" in verdict.answer
    assert gate.CHECK_SUFFICIENT not in verdict.answer  # no check names leak to the reader


def test_every_failure_produces_the_same_shape_of_refusal(settings):
    from custodia.llm import LLMUnavailable

    failures = [
        StubLLM(LLMUnavailable("down")),
        StubLLM(TimeoutError("slow")),
        StubLLM({"answer": "x", "citations": [INVENTED], "sufficient": True}),
        StubLLM({"nonsense": True}),
    ]
    for llm in failures:
        verdict = build(ambiguous(), llm, settings).ask("Which gym?", record=False)
        assert verdict.answered is False
        assert verdict.citations == []
        assert verdict.answer.startswith("I don't have enough in memory")
        assert verdict.abstained_because


def test_the_reason_vocabulary_is_closed(settings):
    """Nothing leaves the gate that the surfaces cannot render."""
    known = {
        gate.NO_EVIDENCE,
        gate.INSUFFICIENT,
        gate.NO_CITATIONS,
        gate.INVENTED_CITATION,
        gate.UNVERIFIED_CITATION,
        gate.MODEL_UNAVAILABLE,
        gate.MALFORMED_RESPONSE,
    }
    assert set(gate._REASON_FOR.values()) <= known


# ---- explain and write-back ------------------------------------------------ #


def test_explain_returns_the_chain_and_interval_per_citation(settings):
    llm = StubLLM(good_reply(GYM))
    gate_ = build(ambiguous(), llm, settings)
    verdict = gate_.ask("Which gym?", record=False)
    explained = gate_.explain(verdict)

    assert [c["fact_id"] for c in explained["citations"]] == [GYM]
    cited = explained["citations"][0]
    assert cited["chain"] == ["Entity:gym", "MENTIONS", f"Fact:{GYM}"]
    assert cited["interval"]["open"] is True
    assert cited["provenance"]["turn_text"].startswith("I switched")
    assert [c["fact_id"] for c in explained["considered"]] == [MILK]


def test_explain_works_for_an_abstention(settings):
    gate_ = build(warrant_of(), None, settings)
    explained = gate_.explain(gate_.ask("Which gym?", record=False))

    assert explained["answered"] is False
    assert explained["citations"] == []
    assert explained["abstained_because"] == gate.NO_EVIDENCE


def test_the_auditor_sees_answers_and_abstentions(settings):
    auditor = RecordingAuditor()
    build(ambiguous(), StubLLM(good_reply(GYM)), settings, auditor=auditor).ask("Which gym?")
    build(ambiguous(), StubLLM(good_reply(INVENTED)), settings, auditor=auditor).ask("Which gym?")

    assert [v.answered for _, _, v in auditor.records] == [True, False]


def test_record_false_writes_nothing(settings):
    auditor = RecordingAuditor()
    build(decisive(), StubLLM(good_reply(GYM)), settings, auditor=auditor).ask(
        "Which gym?", record=False
    )
    assert auditor.records == []


def test_an_audit_failure_does_not_turn_an_abstention_into_an_answer(settings):
    auditor = RecordingAuditor(explode=True)
    verdict = build(
        decisive(), StubLLM(good_reply(GYM)), settings, auditor=auditor
    ).ask("Which gym?")
    assert verdict.answered is True  # the answer stands; only the write-back failed


def test_verdict_as_dict_is_serialisable(settings):
    import json

    verdict = build(decisive(), StubLLM(good_reply(GYM)), settings).ask(
        "Which gym?", record=False
    )
    payload = json.loads(json.dumps(verdict.as_dict()))

    assert set(payload) == {
        "answered", "answer", "citations", "abstained_because",
        "latency_ms", "model", "verified", "checks", "warrant",
    }
    assert payload["citations"] == [GYM]
    assert payload["warrant"]["evidence"][0]["fact_id"] == GYM
    assert set(payload["warrant"]["evidence"][0]) == {
        "fact_id", "text", "tier", "status", "valid_from", "valid_to", "session",
        "session_index", "turn_index", "turn_text", "turn_ts", "score", "hops",
        "path", "superseded_by",
    }
