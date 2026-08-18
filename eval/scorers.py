"""Grading: an LLM judge, a labelled non-LLM fallback, and abstention metrics.

Two design decisions carry most of the weight here.

**The judge returns a structured verdict, not prose.** A free-text judge has to be
re-parsed with a heuristic, and the heuristic is where accuracy quietly inflates.
The judge is asked for one JSON object and, if it does not produce one on two
attempts, the item is graded ``incorrect`` and counted in ``judge_failures`` --
visible in every scorecard. A broken judge must show up as a broken judge, not as
a lower score that looks like a model weakness.

**Abstention is scored as three separate numbers.** "Accuracy" on a benchmark that
mixes answerable and unanswerable questions hides the failure this project exists
to fix. A system that answers everything and a system that refuses everything can
post the same accuracy; only precision, recall and hallucination rate tell them
apart, so all three are computed, and each is ``None`` -- rendered ``not
measured`` -- when its denominator is empty.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from . import ChatLLM, TOKEN_ESTIMATOR

__all__ = [
    "Judgement",
    "RunRecord",
    "Scorecard",
    "judge",
    "fallback_judge",
    "looks_like_abstention",
    "normalize_text",
    "contains_answer",
    "score_run",
]

# --------------------------------------------------------------------------- #
# normalisation helpers, shared with the poison scorer
# --------------------------------------------------------------------------- #

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ARTICLES = {"a", "an", "the"}


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    lowered = _PUNCT.sub(" ", (text or "").lower())
    words = [w for w in _SPACE.split(lowered) if w and w not in _ARTICLES]
    return " ".join(words)


def _numbers(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _NUMBER.findall(text or ""):
        cleaned = raw.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        out.add(str(int(value)) if value == int(value) else str(value))
    return out


_MONTHS = {
    m: f"{i:02d}"
    for i, m in enumerate(
        "january february march april may june july august september october "
        "november december".split(),
        start=1,
    )
}
_DATE_PATTERNS = (
    re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b"),
)


def _dates(text: str) -> set[str]:
    """Dates reduced to a comparable form, so 2024/03/05 == March 5, 2024."""
    found: set[str] = set()
    lowered = (text or "").lower()
    for month, num in _MONTHS.items():
        for match in re.finditer(rf"\b{month}\w*\s+(\d{{1,2}})(?:\w\w)?,?\s*(\d{{4}})?\b", lowered):
            day, year = match.group(1), match.group(2) or ""
            found.add(f"{year}-{num}-{int(day):02d}")
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(lowered):
            groups = match.groups()
            if len(groups[0]) == 4:
                year, month, day = groups
            else:
                month, day, year = groups
            found.add(f"{year}-{int(month):02d}-{int(day):02d}")
    for match in re.finditer(r"\b(19|20)\d{2}\b", lowered):
        found.add(match.group(0))
    return found


def contains_answer(prediction: str, gold: str) -> bool:
    """Whether ``gold`` appears in ``prediction`` after normalisation.

    Containment only, never the reverse: a prediction that merely happens to be a
    substring of the gold answer ("Business" for "Business Administration") is
    not a correct answer.
    """
    norm_gold = normalize_text(gold)
    norm_pred = normalize_text(prediction)
    if not norm_gold:
        return False
    return norm_gold in norm_pred


# --------------------------------------------------------------------------- #
# abstention detection
# --------------------------------------------------------------------------- #

#: Phrases a declining answer uses. Matched on normalised text, so punctuation
#: and casing do not matter. The list is intentionally about *absence of
#: evidence*, not politeness: "I'd rather not say" is a refusal, not an
#: abstention, and should not be credited as one.
_ABSTENTION_MARKERS = (
    "i dont know",
    "i do not know",
    "no information",
    "not enough information",
    "insufficient information",
    "information provided is not enough",
    "not mentioned",
    "you did not mention",
    "you have not mentioned",
    "never mentioned",
    "no record",
    "not in my memory",
    "cannot determine",
    "cant determine",
    "unable to determine",
    "no evidence",
    "not stated",
    "isnt anything in",
    "there is nothing in",
    "no supporting",
    "cannot answer",
    "cant answer",
    "unable to answer",
    "not available in",
    "no relevant",
    "based on the provided chat there is no",
    "memory does not contain",
    "memory has no",
    "did not find",
    "couldnt find",
    "could not find",
)


def looks_like_abstention(text: str) -> bool:
    """Whether an answer declines for lack of evidence.

    Used for both scoring and, in the poison runner, for detecting a hold. It is
    a text heuristic and is reported as one: a system that returns a structured
    ``abstained`` flag should pass that flag through instead of relying on this.
    """
    normalised = normalize_text(text)
    if not normalised:
        return True  # an empty answer answered nothing
    return any(marker in normalised for marker in _ABSTENTION_MARKERS)


# --------------------------------------------------------------------------- #
# judgement
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Judgement:
    """A structured verdict plus how it was reached.

    ``method`` travels into every record and into the scorecard header, because
    an accuracy graded by the lexical fallback is a weaker claim than one graded
    by a model and must not be presented as the same number.
    """

    correct: bool
    reason: str
    method: str = "llm-judge"

    @property
    def is_fallback(self) -> bool:
        return self.method != "llm-judge"


#: Per-question-type instructions, in the style LongMemEval's own evaluator uses:
#: the grading bar differs by what the question is testing, and one generic
#: "is this right?" prompt grades a temporal question far too leniently.
RUBRICS: dict[str, str] = {
    "temporal-reasoning": (
        "This is a temporal question. The prediction is correct only if it gets the "
        "date, ordering or duration right. A right entity with a wrong or missing "
        "time is INCORRECT. Relative expressions are acceptable when they resolve to "
        "the same point in time as the gold answer."
    ),
    "knowledge-update": (
        "This question asks for the CURRENT value of a fact that changed over time. "
        "The prediction is correct only if it states the latest value. Mentioning an "
        "older superseded value as if it were current is INCORRECT; mentioning the "
        "history alongside the correct current value is acceptable."
    ),
    "single-session-preference": (
        "This is a preference question. Accept paraphrase and partial overlap: the "
        "prediction is correct if it conveys the same preference as the gold answer, "
        "even in different words or with extra detail."
    ),
    "single-session-user": (
        "The answer is a specific fact the user stated. The prediction is correct if "
        "it contains that fact, in any phrasing."
    ),
    "single-session-assistant": (
        "The answer is something the assistant said or recommended. The prediction is "
        "correct if it conveys the same content, in any phrasing."
    ),
    "multi-session": (
        "This answer must be assembled from several sessions. The prediction is "
        "correct only if it covers the same set of items as the gold answer; missing "
        "a required item is INCORRECT, extra unrelated items do not by themselves "
        "make it incorrect."
    ),
    "abstention": (
        "The history genuinely does NOT contain the answer. The prediction is correct "
        "ONLY if it declines, states the information is not available, or otherwise "
        "makes clear it cannot answer from what it has. Any confident substantive "
        "answer is INCORRECT no matter how plausible it sounds."
    ),
}

_DEFAULT_RUBRIC = (
    "The prediction is correct if it conveys the same information as the gold "
    "answer. Differences of phrasing, extra detail and formatting do not matter; "
    "a different fact, a missing required part, or a hedge that never states the "
    "answer do."
)

_JUDGE_SYSTEM = (
    "You grade answers against a gold reference. You return exactly one JSON "
    "object and nothing else."
)

_JUDGE_TEMPLATE = """\
Grade the prediction against the gold answer.

{rubric}

QUESTION:
{question}

GOLD ANSWER:
{gold}

PREDICTION:
{prediction}

Return exactly one JSON object, no prose and no code fence:
{{"correct": true or false, "reason": "one short sentence"}}
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(raw: str) -> Judgement | None:
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "correct" not in payload:
        return None
    value = payload["correct"]
    if isinstance(value, str):
        value = value.strip().lower() in {"true", "yes", "correct", "1"}
    return Judgement(
        correct=bool(value),
        reason=str(payload.get("reason", ""))[:400],
        method="llm-judge",
    )


def judge(
    question: str,
    gold: str,
    prediction: str,
    qtype: str,
    *,
    llm: ChatLLM,
) -> Judgement:
    """Grade one prediction with a per-question-type rubric.

    Decoding is pinned to ``temperature=0`` so a re-run of the same result file
    reproduces the same grades. Two parse failures resolve to
    ``method="llm-judge-failed"`` and ``correct=False``: an ungradeable item is
    not a passing item, and the scorecard surfaces the count.
    """
    is_abstention = qtype == "abstention"
    rubric = RUBRICS.get("abstention" if is_abstention else qtype, _DEFAULT_RUBRIC)
    prompt = _JUDGE_TEMPLATE.format(
        rubric=rubric, question=question, gold=gold, prediction=prediction or "(empty)"
    )
    last = ""
    for _ in range(2):
        try:
            last = llm.complete(prompt, system=_JUDGE_SYSTEM, temperature=0.0, max_tokens=200)
        except Exception as exc:
            return Judgement(False, f"judge call failed: {exc}", "llm-judge-failed")
        verdict = _parse_verdict(last)
        if verdict is not None:
            return verdict
    return Judgement(
        False,
        f"judge returned no parseable verdict: {last[:160]!r}",
        "llm-judge-failed",
    )


def fallback_judge(
    question: str,
    gold: str,
    prediction: str,
    qtype: str,
) -> Judgement:
    """Grade without a model. Weaker, and labelled as such everywhere it is used.

    Containment plus numeric and date agreement catches most exact-answer items
    and essentially no paraphrase, so it *under*-reports accuracy rather than
    over-reporting it -- the safe direction for a fallback in a submission whose
    claim is honesty.
    """
    method = "lexical-fallback"
    if qtype == "abstention":
        declined = looks_like_abstention(prediction)
        return Judgement(
            declined,
            "declined" if declined else "answered an unanswerable question",
            method,
        )
    if looks_like_abstention(prediction):
        return Judgement(False, "declined to answer an answerable question", method)

    if contains_answer(prediction, gold):
        return Judgement(True, "gold answer appears in the prediction", method)

    gold_numbers, gold_dates = _numbers(gold), _dates(gold)
    if gold_numbers or gold_dates:
        pred_numbers, pred_dates = _numbers(prediction), _dates(prediction)
        numbers_ok = not gold_numbers or gold_numbers <= pred_numbers
        dates_ok = not gold_dates or bool(gold_dates & pred_dates)
        if numbers_ok and dates_ok:
            return Judgement(True, "numeric/date values agree", method)

    gold_tokens = set(normalize_text(gold).split())
    pred_tokens = set(normalize_text(prediction).split())
    if gold_tokens and len(gold_tokens & pred_tokens) / len(gold_tokens) >= 0.85:
        return Judgement(True, "near-complete token overlap with the gold answer", method)
    return Judgement(False, "no containment, numeric or date match", method)


# --------------------------------------------------------------------------- #
# run records + scorecard
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RunRecord:
    """Everything measured for one (question, system) pair.

    Written to a JSONL side-file as it is produced, which is what makes a long
    run resumable and what lets a scorecard be recomputed without re-running the
    models.
    """

    qid: str
    system: str
    qtype: str
    is_abstention: bool
    question: str = ""
    gold: str = ""
    prediction: str = ""
    abstained: bool = False
    correct: bool = False
    judge_method: str = ""
    judge_reason: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    truncated: bool = False
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate, or ``None`` when there is nothing to divide by.

    ``None`` is rendered ``not measured``. Returning 0.0 for an empty denominator
    would print a perfect-looking hallucination rate for a run that never tested
    hallucination.
    """
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


@dataclass(slots=True)
class Scorecard:
    """The measured result of one system on one run."""

    system: str
    n: int
    answerable: int
    abstention_items: int
    correct: int
    accuracy: float | None
    accuracy_by_type: dict[str, dict[str, Any]]
    abstained_total: int
    abstention_recall: float | None
    abstention_precision: float | None
    hallucination_rate: float | None
    over_refusal_rate: float | None
    mean_latency_ms: float | None
    mean_prompt_tokens: float | None
    judge_methods: dict[str, int]
    judge_failures: int
    errors: int
    truncated: int
    token_estimator: str = TOKEN_ESTIMATOR

    def as_json(self) -> dict[str, Any]:
        return asdict(self)

    def as_markdown(self) -> str:
        from .report import render_scorecard  # local import: report imports us

        return render_scorecard(self)


def score_run(records: Iterable[RunRecord | dict[str, Any]], *, system: str = "") -> Scorecard:
    """Turn per-question records into one system's scorecard.

    Definitions, stated because the interesting ones are easy to get subtly wrong:

    * ``accuracy`` is over **answerable** questions only. Abstention questions are
      excluded and reported in their own block, so a refuse-everything system
      cannot borrow credit from them.
    * ``abstention_recall`` -- of the questions that genuinely have no answer, the
      share the system declined.
    * ``abstention_precision`` -- of everything the system declined, the share that
      genuinely had no answer.
    * ``hallucination_rate`` -- of the questions that genuinely have no answer, the
      share where the system answered anyway *and was graded wrong*. This is at or
      below ``1 - abstention_recall``; the gap is items that answered without
      declining yet still satisfied the judge.
    * ``over_refusal_rate`` -- the mirror image: answerable questions the system
      declined. Without it, abstention recall could be maximised by refusing
      everything.
    """
    rows = [r if isinstance(r, RunRecord) else RunRecord.from_dict(r) for r in records]
    if system:
        rows = [r for r in rows if r.system == system]
    name = system or (rows[0].system if rows else "unknown")

    answerable = [r for r in rows if not r.is_abstention]
    abstention = [r for r in rows if r.is_abstention]
    declined = [r for r in rows if r.abstained]

    by_type: dict[str, dict[str, Any]] = {}
    for row in answerable:
        bucket = by_type.setdefault(row.qtype, {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(row.correct)
    for bucket in by_type.values():
        bucket["accuracy"] = _rate(bucket["correct"], bucket["n"])

    latencies = [r.latency_ms for r in rows if r.latency_ms > 0]
    tokens = [r.prompt_tokens for r in rows if r.prompt_tokens > 0]
    methods: dict[str, int] = {}
    for row in rows:
        if row.judge_method:
            methods[row.judge_method] = methods.get(row.judge_method, 0) + 1

    return Scorecard(
        system=name,
        n=len(rows),
        answerable=len(answerable),
        abstention_items=len(abstention),
        correct=sum(1 for r in answerable if r.correct),
        accuracy=_rate(sum(1 for r in answerable if r.correct), len(answerable)),
        accuracy_by_type=dict(sorted(by_type.items())),
        abstained_total=len(declined),
        abstention_recall=_rate(sum(1 for r in abstention if r.abstained), len(abstention)),
        abstention_precision=_rate(sum(1 for r in declined if r.is_abstention), len(declined)),
        hallucination_rate=_rate(
            sum(1 for r in abstention if not r.abstained and not r.correct), len(abstention)
        ),
        over_refusal_rate=_rate(sum(1 for r in answerable if r.abstained), len(answerable)),
        mean_latency_ms=round(statistics.mean(latencies), 1) if latencies else None,
        mean_prompt_tokens=round(statistics.mean(tokens), 1) if tokens else None,
        judge_methods=dict(sorted(methods.items())),
        judge_failures=sum(1 for r in rows if r.judge_method == "llm-judge-failed"),
        errors=sum(1 for r in rows if r.error),
        truncated=sum(1 for r in rows if r.truncated),
    )


def score_systems(records: Sequence[RunRecord | dict[str, Any]]) -> dict[str, Scorecard]:
    """One scorecard per system present in ``records``."""
    rows = [r if isinstance(r, RunRecord) else RunRecord.from_dict(r) for r in records]
    return {name: score_run(rows, system=name) for name in sorted({r.system for r in rows})}
