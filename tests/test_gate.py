"""The gate. Every abstention branch, driven by a stub model.

The claim under test is narrow: no reply from a language model, however
confident, well-formed or insistent, can produce an answer that the warrant does
not justify. So each test here breaks exactly one thing and asserts the same
outcome - `answered is False`, with the failed check named.
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


FULL = None  # placeholder replaced per-test by warrant_of(...)


# ---- abstention branches --------------------------------------------------- #


def test_empty_warrant_never_reaches_the_model(settings):
    llm = StubLLM(good_reply(GYM))
    verdict = build(warrant_of(), llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.CHECK_WARRANT
    assert verdict.checks == [gate.CHECK_WARRANT]
    assert llm.calls == []


def test_no_model_configured_abstains(settings):
    verdict = build(warrant_of(evidence(GYM, "Northline.")), None, settings).ask("q", record=False)
    assert verdict.answered is False
    assert verdict.abstained_because == gate.CHECK_MODEL


def test_a_disabled_model_abstains(settings):
    llm = StubLLM(good_reply(GYM), enabled=False)
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.abstained_because == gate.CHECK_MODEL
    assert llm.calls == []


def test_a_non_mapping_reply_abstains(settings):
    llm = StubLLM(["not", "an", "object"])
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.answered is False
    assert verdict.abstained_because == gate.CHECK_JSON


def test_unparseable_json_from_the_provider_abstains(settings):
    from custodia.llm import LLMUnavailable

    llm = StubLLM(LLMUnavailable("model did not return parseable JSON"))
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.answered is False
    assert verdict.abstained_because == gate.REASON_UNAVAILABLE


def test_a_reply_missing_required_keys_abstains(settings):
    llm = StubLLM({"answer": "Northline.", "sufficient": True})  # no citations key
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.abstained_because == gate.CHECK_SCHEMA


def test_sufficient_false_short_circuits(settings):
    llm = StubLLM({"answer": "Probably Northline.", "citations": [GYM], "sufficient": False})
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.answered is False
    assert verdict.abstained_because == gate.CHECK_SUFFICIENT
    assert "Probably Northline" not in verdict.answer


def test_a_truthy_but_non_boolean_sufficient_is_not_accepted(settings):
    llm = StubLLM({"answer": "Northline.", "citations": [GYM], "sufficient": "yes"})
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.abstained_because == gate.CHECK_SUFFICIENT


def test_an_empty_citation_list_abstains(settings):
    llm = StubLLM(good_reply())
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.abstained_because == gate.CHECK_CITED


def test_a_hallucinated_citation_id_abstains(settings):
    llm = StubLLM(good_reply(INVENTED))
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.CHECK_IN_WARRANT
    assert verdict.citations == []
    assert "Northline" not in verdict.answer


def test_one_invented_id_among_real_ones_discards_the_whole_answer(settings):
    warrant = warrant_of(evidence(GYM, "Northline."), evidence(MILK, "Oat milk."))
    llm = StubLLM(good_reply(GYM, INVENTED))
    verdict = build(warrant, llm, settings).ask("q", record=False)
    assert verdict.abstained_because == gate.CHECK_IN_WARRANT


def test_citations_that_are_not_ids_at_all_abstain(settings):
    llm = StubLLM({"answer": "Northline.", "citations": ["fact one"], "sufficient": True})
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.abstained_because == gate.CHECK_CITED


def test_a_citation_id_sent_as_a_string_is_accepted(settings):
    llm = StubLLM({"answer": "Northline.", "citations": [str(GYM)], "sufficient": True})
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.answered is True
    assert verdict.citations == [GYM]


def test_an_empty_answer_abstains(settings):
    llm = StubLLM({"answer": "   ", "citations": [GYM], "sufficient": True})
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.abstained_because == gate.CHECK_TEXT


@pytest.mark.parametrize(
    "text",
    [
        "I don't know which gym the user goes to.",
        "I do not have information about that.",
        "There is no record of the user's gym.",
        "Insufficient evidence to answer.",
        "Unable to determine the user's gym.",
    ],
)
def test_an_answer_that_is_itself_a_refusal_abstains(settings, text):
    llm = StubLLM({"answer": text, "citations": [GYM], "sufficient": True})
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.answered is False
    assert verdict.abstained_because == gate.CHECK_NOT_REFUSAL


def test_a_hedged_but_real_answer_is_not_mistaken_for_a_refusal(settings):
    llm = StubLLM(
        {
            "answer": "I don't know the exact date, but the gym is Northline Fitness.",
            "citations": [GYM],
            "sufficient": True,
        }
    )
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.answered is False  # opening clause is a refusal, so it is caught
    assert verdict.abstained_because == gate.CHECK_NOT_REFUSAL

    llm = StubLLM(
        {
            "answer": "The gym is Northline Fitness, though I don't know when they joined.",
            "citations": [GYM],
            "sufficient": True,
        }
    )
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.answered is True


def test_provider_unavailable_abstains(settings):
    from custodia.llm import LLMUnavailable

    llm = StubLLM(LLMUnavailable("no credentials configured"))
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.answered is False
    assert verdict.abstained_because == gate.REASON_UNAVAILABLE


def test_a_timeout_abstains(settings):
    llm = StubLLM(TimeoutError("read timed out"))
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.answered is False
    assert verdict.abstained_because == gate.REASON_TIMEOUT


def test_an_unexpected_transport_error_abstains(settings):
    llm = StubLLM(ConnectionResetError("peer reset"))
    verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask("q", record=False)
    assert verdict.answered is False
    assert verdict.abstained_because == gate.REASON_ERROR


# ---- the happy path -------------------------------------------------------- #


def test_a_warranted_answer_is_served(settings):
    warrant = warrant_of(evidence(GYM, "The user's gym is Northline Fitness."))
    llm = StubLLM(good_reply(GYM))
    verdict = build(warrant, llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert verdict.answer == "The user goes to Northline Fitness."
    assert verdict.citations == [GYM]
    assert verdict.abstained_because == ""
    assert verdict.latency_ms >= 0
    assert verdict.model == "stub/answerer"
    assert gate.CHECK_IN_WARRANT in verdict.checks


def test_the_model_sees_the_warrant_and_not_the_conversation(settings):
    warrant = warrant_of(evidence(GYM, "The user's gym is Northline Fitness."))
    llm = StubLLM(good_reply(GYM))
    build(warrant, llm, settings).ask("Which gym?", record=False)

    prompt = llm.calls[0][-1]["content"]
    assert str(GYM) in prompt
    assert "The user's gym is Northline Fitness." in prompt
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
    warrant = warrant_of(evidence(GYM, "The user's gym is Northline Fitness."))
    llm = StubLLM(good_reply(GYM), {"supports": True})
    verdict = build(warrant, llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert verdict.verified == 1
    assert gate.CHECK_SUPPORTED in verdict.checks


def test_the_verifier_drops_a_citation_that_does_not_support_the_answer(settings):
    settings.verify_citations = True
    warrant = warrant_of(
        evidence(GYM, "The user's gym is Northline Fitness."),
        evidence(MILK, "The user drinks oat milk."),
    )
    llm = StubLLM(good_reply(GYM, MILK), {"supports": True}, {"supports": False})
    verdict = build(warrant, llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert verdict.citations == [GYM]
    assert verdict.verified == 1


def test_the_verifier_rejecting_every_citation_abstains(settings):
    settings.verify_citations = True
    warrant = warrant_of(evidence(GYM, "The user drinks oat milk."))
    llm = StubLLM(good_reply(GYM), {"supports": False})
    verdict = build(warrant, llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.CHECK_SUPPORTED
    assert verdict.verified == 0


def test_a_verifier_that_errors_drops_the_citation_rather_than_trusting_it(settings):
    from custodia.llm import LLMUnavailable

    settings.verify_citations = True
    warrant = warrant_of(evidence(GYM, "Northline."))
    llm = StubLLM(good_reply(GYM), LLMUnavailable("provider down"))
    verdict = build(warrant, llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is False
    assert verdict.abstained_because == gate.CHECK_SUPPORTED


def test_verification_is_skipped_when_disabled(settings):
    settings.verify_citations = False
    warrant = warrant_of(evidence(GYM, "Northline."))
    llm = StubLLM(good_reply(GYM))
    verdict = build(warrant, llm, settings).ask("Which gym?", record=False)

    assert verdict.answered is True
    assert gate.CHECK_SUPPORTED not in verdict.checks
    assert len(llm.calls) == 1


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
        verdict = build(warrant_of(evidence(GYM, "Northline.")), llm, settings).ask(
            "Which gym?", record=False
        )
        assert verdict.answered is False
        assert verdict.citations == []
        assert verdict.answer.startswith("I don't have enough in memory")
        assert verdict.abstained_because


# ---- explain and write-back ------------------------------------------------ #


def test_explain_returns_the_chain_and_interval_per_citation(settings):
    warrant = warrant_of(
        evidence(GYM, "The user's gym is Northline Fitness."),
        evidence(MILK, "The user drinks oat milk.", score=0.3),
    )
    llm = StubLLM(good_reply(GYM))
    gate_ = build(warrant, llm, settings)
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
    assert explained["abstained_because"] == gate.CHECK_WARRANT


def test_the_auditor_sees_answers_and_abstentions(settings):
    auditor = RecordingAuditor()
    warrant = warrant_of(evidence(GYM, "Northline."))
    build(warrant, StubLLM(good_reply(GYM)), settings, auditor=auditor).ask("Which gym?")
    build(warrant, StubLLM(good_reply(INVENTED)), settings, auditor=auditor).ask("Which gym?")

    assert [v.answered for _, _, v in auditor.records] == [True, False]


def test_record_false_writes_nothing(settings):
    auditor = RecordingAuditor()
    build(warrant_of(evidence(GYM, "N.")), StubLLM(good_reply(GYM)), settings, auditor=auditor).ask(
        "Which gym?", record=False
    )
    assert auditor.records == []


def test_an_audit_failure_does_not_turn_an_abstention_into_an_answer(settings):
    auditor = RecordingAuditor(explode=True)
    verdict = build(
        warrant_of(evidence(GYM, "N.")), StubLLM(good_reply(GYM)), settings, auditor=auditor
    ).ask("Which gym?")
    assert verdict.answered is True  # the answer stands; only the write-back failed


def test_verdict_as_dict_is_serialisable(settings):
    import json

    warrant = warrant_of(evidence(GYM, "Northline."))
    verdict = build(warrant, StubLLM(good_reply(GYM)), settings).ask("Which gym?", record=False)
    payload = json.loads(json.dumps(verdict.as_dict()))

    assert payload["answered"] is True
    assert payload["citations"] == [GYM]
    assert payload["warrant"]["evidence"][0]["fid"] == GYM
