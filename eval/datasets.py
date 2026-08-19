"""Benchmark loading, normalisation and deterministic sampling.

Three published benchmarks are supported behind one interface, because a memory
layer that only ever sees one question format is a memory layer tuned to that
format:

======================  ================================================
``s``, ``s_cleaned``,   LongMemEval (Wu et al.), chat histories of ~50
``m``, ``oracle``       sessions and ~120k estimated tokens per question
``v2``                  LongMemEval-V2, web/enterprise agent trajectories
``beam-100k`` ...       BEAM, long synthetic conversations with ten
``beam-1m``             probing-question families
======================  ================================================

Everything is normalised to the same :class:`Instance`: a question asked at a
point in time, against an ordered list of sessions of turns. That shape is what
Custodia ingests and what the baselines concatenate, so all three systems in a
run see byte-identical inputs.

Two properties matter more than convenience here:

*reproducibility* -- every raw file is downloaded to ``eval/data/raw/`` (which is
gitignored), hashed, and recorded in ``eval/data/manifest.json`` with its source
URL, size, sha256 and item count. That manifest is committed, so a result file
can be tied back to the exact bytes it was produced from;

*honesty about availability* -- a dataset that cannot be reached raises
:class:`DatasetUnavailable` naming what was missing. Nothing silently falls back
to a different or smaller dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import TOKEN_ESTIMATOR, estimate_tokens

__all__ = [
    "EvalTurn",
    "EvalSession",
    "Instance",
    "DatasetUnavailable",
    "Source",
    "SOURCES",
    "load_longmemeval",
    "load_dataset",
    "dataset_stats",
    "normalize_longmemeval",
    "normalize_beam",
    "normalize_longmemeval_v2",
    "sample_instances",
    "parse_lme_date",
    "manifest",
    "DATA_DIR",
    "RAW_DIR",
    "MANIFEST_PATH",
]

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"

#: A LongMemEval question id ending in this suffix is an abstention question:
#: the haystack deliberately does not contain the answer, and the only correct
#: behaviour is to decline. This is the benchmark's own convention.
ABSTENTION_SUFFIX = "_abs"

#: Guard against a >1 GB download happening as a surprise inside a test or a
#: quick CLI run. Set to "1" (or pass ``allow_large=True``) to permit it.
LARGE_DOWNLOAD_ENV = "CUSTODIA_EVAL_ALLOW_LARGE"


class DatasetUnavailable(RuntimeError):
    """A dataset could not be obtained, with the reason spelled out.

    Deliberately fatal. Substituting a different dataset would make a scorecard
    describe something other than what its header claims.
    """


# --------------------------------------------------------------------------- #
# normalised records
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class EvalTurn:
    """One message. ``has_answer`` marks a turn the benchmark labels as evidence."""

    role: str
    content: str
    has_answer: bool = False


@dataclass(slots=True)
class EvalSession:
    """One conversation. ``ts`` is epoch seconds; ``date`` is the source string."""

    sid: str
    ts: int
    date: str
    turns: list[EvalTurn] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(f"{t.role}: {t.content}" for t in self.turns)


@dataclass(slots=True)
class Instance:
    """One benchmark question and the whole history it must be answered from."""

    qid: str
    qtype: str
    question: str
    answer: str
    question_date: str
    asked_at: int
    sessions: list[EvalSession] = field(default_factory=list)
    answer_session_ids: list[str] = field(default_factory=list)
    dataset: str = "longmemeval-s"

    @property
    def is_abstention(self) -> bool:
        """True when the benchmark guarantees the answer is *not* in the history.

        LongMemEval marks these by suffixing the question id with ``_abs``. The
        other adapters mint ids the same way, so this one predicate is correct
        for every dataset the harness loads.
        """
        return self.qid.endswith(ABSTENTION_SUFFIX)

    @property
    def n_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)

    @property
    def haystack_chars(self) -> int:
        return sum(len(t.content) for s in self.sessions for t in s.turns)

    def estimated_tokens(self) -> int:
        return estimate_tokens("x" * self.haystack_chars)

    def evidence_turns(self) -> list[tuple[str, EvalTurn]]:
        """Turns the benchmark itself labelled as carrying the answer."""
        return [
            (s.sid, t) for s in self.sessions for t in s.turns if t.has_answer
        ]


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class Source:
    """Where a raw artefact comes from and what it should hash to.

    ``sha256`` is pinned only where it has been verified against a real download;
    where it is ``None`` the value is recorded into the manifest at download time
    instead of asserted. A pin we have not checked would be a lie in a file whose
    whole job is provenance.
    """

    key: str
    repo: str
    remote: str
    local: str
    kind: str  # "lme-chat" | "lme-v2" | "beam-parquet"
    size: int | None = None
    sha256: str | None = None
    note: str = ""

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/resolve/main/{self.remote}"


_LME_REPO = "xiaowu0162/longmemeval"
_LME_CLEAN_REPO = "xiaowu0162/longmemeval-cleaned"
_LME_V2_REPO = "xiaowu0162/longmemeval-v2"
_BEAM_REPO = "Mohammadta/BEAM"

#: Sizes and hashes below were read from the Hugging Face file listing and, for
#: ``s`` and ``oracle``, confirmed against a completed local download.
SOURCES: dict[str, Source] = {
    "s": Source(
        key="s",
        repo=_LME_REPO,
        remote="longmemeval_s",
        local="longmemeval_s.json",
        kind="lme-chat",
        size=278025796,
        sha256="08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894",
        note="LongMemEval-S: 500 questions, ~50 sessions each, ~122k estimated tokens per question",
    ),
    "oracle": Source(
        key="oracle",
        repo=_LME_REPO,
        remote="longmemeval_oracle",
        local="longmemeval_oracle.json",
        kind="lme-chat",
        size=15388478,
        sha256="821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
        note="evidence sessions only - a fast smoke-test haystack, not a difficulty measurement",
    ),
    "m": Source(
        key="m",
        repo=_LME_REPO,
        remote="longmemeval_m",
        local="longmemeval_m.json",
        kind="lme-chat",
        size=2745274681,
        sha256="fb5413e3b077c62927daab794836991a2fcfa61ceacab57dc679fb02daaff2d9",
        note="LongMemEval-M: ~500 sessions per question (2.7 GB download)",
    ),
    "s_cleaned": Source(
        key="s_cleaned",
        repo=_LME_CLEAN_REPO,
        remote="longmemeval_s_cleaned.json",
        local="longmemeval_s_cleaned.json",
        kind="lme-chat",
        size=277383467,
        sha256=None,
        note="author-published revision of S with noisy distractor sessions removed",
    ),
    "v2": Source(
        key="v2",
        repo=_LME_V2_REPO,
        remote="questions.jsonl",
        local="lme_v2_questions.jsonl",
        kind="lme-v2",
        size=286186,
        sha256=None,
        note="LongMemEval-V2: 451 agent-trajectory questions; needs the 1.2 GB trajectory file too",
    ),
    "beam-100k": Source(
        key="beam-100k",
        repo=_BEAM_REPO,
        remote="data/100K-00000-of-00001.parquet",
        local="beam_100K.parquet",
        kind="beam-parquet",
        size=5429768,
        sha256=None,
        note="BEAM 100K split: 20 conversations x 10 probing-question families",
    ),
    "beam-500k": Source(
        key="beam-500k",
        repo=_BEAM_REPO,
        remote="data/500K-00000-of-00001.parquet",
        local="beam_500K.parquet",
        kind="beam-parquet",
        size=33956263,
        sha256=None,
        note="BEAM 500K split: 35 conversations",
    ),
    "beam-1m": Source(
        key="beam-1m",
        repo=_BEAM_REPO,
        remote="data/1M-00000-of-00001.parquet",
        local="beam_1M.parquet",
        kind="beam-parquet",
        size=66156374,
        sha256=None,
        note="BEAM 1M split: 35 conversations",
    ),
}

#: Extra artefacts LongMemEval-V2 needs beyond its question file.
_V2_EXTRA = {
    "trajectories": Source(
        key="v2-trajectories",
        repo=_LME_V2_REPO,
        remote="trajectories.jsonl",
        local="lme_v2_trajectories.jsonl",
        kind="lme-v2",
        size=1195604539,
        sha256=None,
        note="1.2 GB - gated behind allow_large",
    ),
    "haystack": Source(
        key="v2-haystack-small",
        repo=_LME_V2_REPO,
        remote="haystacks/lme_v2_small.json",
        local="lme_v2_small.json",
        kind="lme-v2",
        size=None,
        sha256=None,
        note="question id -> 100 trajectory ids",
    ),
}


# --------------------------------------------------------------------------- #
# download + manifest
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(source: Source, *, retries: int = 12) -> Path:
    """Download to ``eval/data/raw`` if absent, resuming a partial transfer.

    Hugging Face closes long transfers often enough that a non-resuming download
    of a 278 MB file fails routinely; a partial ``.part`` file is continued with a
    ranged request rather than restarted, so a flaky link still converges.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / source.local
    if dest.exists() and (source.size is None or dest.stat().st_size == source.size):
        return dest

    import httpx

    part = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for _ in range(max(1, retries)):
        have = part.stat().st_size if part.exists() else 0
        if source.size is not None and have >= source.size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.stream(
                "GET", source.url, headers=headers, timeout=120, follow_redirects=True
            ) as response:
                if have and response.status_code == 200:
                    # server ignored the range: start clean rather than corrupt
                    part.unlink(missing_ok=True)
                    have = 0
                response.raise_for_status()
                with part.open("ab" if have else "wb") as handle:
                    for chunk in response.iter_bytes(1 << 20):
                        handle.write(chunk)
            if source.size is None or part.stat().st_size >= source.size:
                break
        except Exception as exc:  # network flakiness is expected; retry
            last = exc
    if not part.exists():
        raise DatasetUnavailable(f"download of {source.url} produced nothing: {last}")
    if source.size is not None and part.stat().st_size != source.size:
        raise DatasetUnavailable(
            f"download of {source.url} is incomplete: "
            f"{part.stat().st_size} of {source.size} bytes (last error: {last})"
        )
    digest = _sha256(part)
    if source.sha256 and digest != source.sha256:
        part.unlink(missing_ok=True)
        raise DatasetUnavailable(
            f"{source.url} hashed to {digest}, expected {source.sha256}; refusing to use it"
        )
    part.replace(dest)
    return dest


def _record_manifest(source: Source, path: Path, items: int) -> dict[str, Any]:
    """Merge one artefact's provenance into the committed manifest."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "key": source.key,
        "repo": source.repo,
        "url": source.url,
        "file": source.local,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "items": items,
        "note": source.note,
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    current: dict[str, Any] = {"schema": 1, "artifacts": {}}
    if MANIFEST_PATH.exists():
        try:
            current = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    current.setdefault("artifacts", {})[source.key] = entry
    MANIFEST_PATH.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return entry


def manifest() -> dict[str, Any]:
    """The recorded provenance of every artefact this machine has downloaded."""
    if not MANIFEST_PATH.exists():
        return {"schema": 1, "artifacts": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# LongMemEval (S / M / oracle / cleaned)
# --------------------------------------------------------------------------- #

#: "2023/05/30 (Tue) 23:40" - the weekday is decorative and locale-hostile, so it
#: is matched and discarded rather than parsed.
_LME_DATE = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})\s*(?:\([^)]*\))?\s*(\d{1,2}):(\d{2})")


def parse_lme_date(raw: str) -> int:
    """LongMemEval's date string to epoch seconds (UTC)."""
    match = _LME_DATE.match((raw or "").strip())
    if not match:
        raise ValueError(f"unparseable LongMemEval date: {raw!r}")
    year, month, day, hour, minute = (int(g) for g in match.groups())
    return int(datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp())


def normalize_longmemeval(raw: Sequence[dict[str, Any]], *, dataset: str = "longmemeval-s") -> list[Instance]:
    """Map LongMemEval's on-disk records onto :class:`Instance`.

    Field names below were read off the real file, not assumed: each record has
    ``question_id``, ``question_type``, ``question``, ``answer``,
    ``question_date``, ``answer_session_ids`` and the three parallel arrays
    ``haystack_dates`` / ``haystack_session_ids`` / ``haystack_sessions``. Turns
    carry ``role`` and ``content``; ``has_answer`` appears only on evidence turns
    in the S/M haystacks, so its absence means False rather than unknown.
    """
    instances: list[Instance] = []
    for record in raw:
        dates = list(record.get("haystack_dates") or [])
        sids = list(record.get("haystack_session_ids") or [])
        raw_sessions = list(record.get("haystack_sessions") or [])
        if not (len(dates) == len(sids) == len(raw_sessions)):
            raise ValueError(
                f"{record.get('question_id')}: haystack arrays disagree "
                f"({len(dates)}/{len(sids)}/{len(raw_sessions)})"
            )
        sessions: list[EvalSession] = []
        for date, sid, turns in zip(dates, sids, raw_sessions):
            sessions.append(
                EvalSession(
                    sid=str(sid),
                    ts=parse_lme_date(date),
                    date=str(date),
                    turns=[
                        EvalTurn(
                            role=str(turn.get("role", "user")),
                            content=str(turn.get("content", "")),
                            has_answer=bool(turn.get("has_answer", False)),
                        )
                        for turn in turns
                    ],
                )
            )
        sessions.sort(key=lambda s: s.ts)
        instances.append(
            Instance(
                qid=str(record["question_id"]),
                qtype=str(record.get("question_type", "unknown")),
                question=str(record.get("question", "")),
                answer=str(record.get("answer", "")),
                question_date=str(record.get("question_date", "")),
                asked_at=parse_lme_date(record["question_date"])
                if record.get("question_date")
                else 0,
                sessions=sessions,
                answer_session_ids=[str(x) for x in (record.get("answer_session_ids") or [])],
                dataset=dataset,
            )
        )
    return instances


# --------------------------------------------------------------------------- #
# LongMemEval-V2 (agent trajectories)
# --------------------------------------------------------------------------- #


def normalize_longmemeval_v2(
    questions: Sequence[dict[str, Any]],
    trajectories: dict[str, dict[str, Any]],
    haystack: dict[str, Sequence[str]],
) -> list[Instance]:
    """Flatten V2's trajectories into the same session/turn shape.

    A trajectory is a session; each state contributes up to three turns -- the
    agent's thought (assistant), the accessibility-tree observation (tool) and
    the action taken (assistant). Mapping observations onto the ``tool`` role
    matters downstream: Custodia derives a fact's trust tier from the role of the
    turn it came from, so an environment observation must not arrive dressed as
    the principal.

    V2 has no per-question timestamps, so session times are synthesised as an
    ordinal sequence and the question is stamped after the last one. Ordering is
    therefore real; wall-clock spacing is not, and no temporal metric should be
    read off it.
    """
    instances: list[Instance] = []
    base_ts = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    for question in questions:
        qid = str(question["id"])
        qtype = str(question.get("question_type", "unknown"))
        ids = list(haystack.get(qid) or [])
        sessions: list[EvalSession] = []
        for order, tid in enumerate(ids):
            traj = trajectories.get(str(tid))
            if traj is None:
                continue
            turns: list[EvalTurn] = [
                EvalTurn(role="user", content=f"Task: {traj.get('goal', '')}")
            ]
            for state in traj.get("states") or []:
                if state.get("thought"):
                    turns.append(EvalTurn(role="assistant", content=str(state["thought"])))
                if state.get("accessibility_tree"):
                    turns.append(
                        EvalTurn(
                            role="tool",
                            content=f"[{state.get('url', '')}]\n{state['accessibility_tree']}",
                        )
                    )
                if state.get("action"):
                    turns.append(EvalTurn(role="assistant", content=f"action: {state['action']}"))
            turns.append(
                EvalTurn(
                    role="tool",
                    content=f"outcome: {traj.get('outcome', 'unknown')}",
                )
            )
            ts = base_ts + order * 3600
            sessions.append(
                EvalSession(
                    sid=str(tid),
                    ts=ts,
                    date=datetime.fromtimestamp(ts, timezone.utc).strftime("%Y/%m/%d (%a) %H:%M"),
                    turns=turns,
                )
            )
        asked_at = (sessions[-1].ts + 3600) if sessions else base_ts
        # V2 marks unanswerable questions with an "-abs" question_type; mint the
        # id with the same "_abs" suffix LongMemEval-S uses so one predicate works.
        suffix = ABSTENTION_SUFFIX if qtype.endswith("-abs") else ""
        instances.append(
            Instance(
                qid=qid + suffix,
                qtype=qtype,
                question=str(question.get("question", "")),
                answer=str(question.get("answer", "")),
                question_date=datetime.fromtimestamp(asked_at, timezone.utc).strftime(
                    "%Y/%m/%d (%a) %H:%M"
                ),
                asked_at=asked_at,
                sessions=sessions,
                answer_session_ids=[],
                dataset="longmemeval-v2",
            )
        )
    return instances


def _load_v2(*, allow_large: bool) -> list[Instance]:
    questions_path = _fetch(SOURCES["v2"])
    questions = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _record_manifest(SOURCES["v2"], questions_path, len(questions))

    haystack_source = _V2_EXTRA["haystack"]
    haystack_path = RAW_DIR / haystack_source.local
    if not haystack_path.exists():
        haystack_path = _fetch(haystack_source)
    haystack = json.loads(haystack_path.read_text(encoding="utf-8"))
    _record_manifest(haystack_source, haystack_path, len(haystack))

    traj_source = _V2_EXTRA["trajectories"]
    traj_path = RAW_DIR / traj_source.local
    if not traj_path.exists():
        permitted = allow_large or os.environ.get(LARGE_DOWNLOAD_ENV, "") == "1"
        if not permitted:
            raise DatasetUnavailable(
                "LongMemEval-V2 needs trajectories.jsonl (1.2 GB) which is not cached. "
                f"Re-run with allow_large=True or {LARGE_DOWNLOAD_ENV}=1, or download it to "
                f"{traj_path}. Skipping rather than substituting a smaller dataset."
            )
        traj_path = _fetch(traj_source)
    trajectories: dict[str, dict[str, Any]] = {}
    with traj_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                trajectories[str(record["id"])] = record
    _record_manifest(traj_source, traj_path, len(trajectories))
    return normalize_longmemeval_v2(questions, trajectories, haystack)


# --------------------------------------------------------------------------- #
# BEAM
# --------------------------------------------------------------------------- #

#: BEAM's ten probing families. ``abstention`` is the one this project is built
#: around, which is precisely why the other nine are kept: a system tuned to
#: refuse would look excellent on one family and terrible on the rest.
BEAM_FAMILIES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)


def normalize_beam(rows: Sequence[dict[str, Any]], *, split: str) -> list[Instance]:
    """One :class:`Instance` per BEAM probing question.

    BEAM stores ``probing_questions`` as a Python literal string rather than
    JSON, and its chat is a list of sessions of turns carrying ``role``,
    ``content`` and a ``time_anchor`` such as ``"March-15-2024"``. Sessions
    inherit an ordinal timestamp because the anchors are coarse and repeat across
    sessions; ordering is faithful, spacing is not.
    """
    import ast

    instances: list[Instance] = []
    base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    for row in rows:
        conv = str(row.get("conversation_id", ""))
        sessions: list[EvalSession] = []
        for order, chat_session in enumerate(row.get("chat") or []):
            ts = base_ts + order * 86400
            sessions.append(
                EvalSession(
                    sid=f"beam-{conv}-s{order}",
                    ts=ts,
                    date=datetime.fromtimestamp(ts, timezone.utc).strftime("%Y/%m/%d (%a) %H:%M"),
                    turns=[
                        EvalTurn(
                            role=str(turn.get("role", "user")),
                            content=str(turn.get("content", "")),
                        )
                        for turn in chat_session
                    ],
                )
            )
        raw_probes = row.get("probing_questions")
        if isinstance(raw_probes, str):
            try:
                probes = ast.literal_eval(raw_probes)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"BEAM {conv}: probing_questions is not a literal: {exc}") from exc
        else:
            probes = raw_probes or {}
        asked_at = (sessions[-1].ts + 86400) if sessions else base_ts
        for family, items in sorted(probes.items()):
            for index, probe in enumerate(items or []):
                suffix = ABSTENTION_SUFFIX if family == "abstention" else ""
                instances.append(
                    Instance(
                        qid=f"beam-{split}-{conv}-{family}-{index}{suffix}",
                        qtype=family,
                        question=str(probe.get("question", "")),
                        answer=str(probe.get("ideal_response", "")),
                        question_date=datetime.fromtimestamp(asked_at, timezone.utc).strftime(
                            "%Y/%m/%d (%a) %H:%M"
                        ),
                        asked_at=asked_at,
                        sessions=sessions,
                        answer_session_ids=[],
                        dataset=f"beam-{split}",
                    )
                )
    return instances


def _load_beam(source: Source) -> list[Instance]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DatasetUnavailable(
            "BEAM ships as Parquet and pyarrow is not installed "
            "(pip install pyarrow). Skipping rather than substituting."
        ) from exc
    path = _fetch(source)
    table = pq.read_table(path)
    rows = table.to_pylist()
    _record_manifest(source, path, len(rows))
    split = source.key.split("-", 1)[1]
    return normalize_beam(rows, split=split)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #

#: Never let a sample carry fewer abstention questions than this (or fewer than
#: the dataset has, if it has fewer). Abstention is the headline claim, and a
#: strictly proportional sample of 50 questions from LongMemEval-S would contain
#: three of them -- a denominator too small for the resulting rate to mean
#: anything.
DEFAULT_MIN_ABSTENTION = 8

#: ...but the floor is itself capped at this share of the sample, so a small
#: ``--limit`` cannot turn into a mostly-abstention run. Without the cap a
#: 12-question smoke test would be two-thirds unanswerable, and its overall
#: accuracy would describe a benchmark nobody published.
MAX_ABSTENTION_SHARE = 0.25


def _largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    """Allocate ``total`` slots across ``weights`` proportionally, exactly.

    Plain rounding loses or invents items; the largest-remainder method keeps the
    allocation summing to ``total`` and is deterministic given a stable key order.
    """
    pool = sum(weights.values())
    if pool <= 0 or total <= 0:
        return {k: 0 for k in weights}
    exact = {k: total * v / pool for k, v in weights.items()}
    base = {k: int(math.floor(v)) for k, v in exact.items()}
    remaining = total - sum(base.values())
    order = sorted(weights, key=lambda k: (-(exact[k] - base[k]), k))
    for key in order[:remaining]:
        base[key] += 1
    for key in base:
        base[key] = min(base[key], weights[key])
    # proportional caps can leave slots unused; hand them out deterministically
    leftover = total - sum(base.values())
    while leftover > 0:
        progressed = False
        for key in sorted(weights):
            if leftover <= 0:
                break
            if base[key] < weights[key]:
                base[key] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return base


def sample_instances(
    instances: Sequence[Instance],
    *,
    limit: int | None = None,
    seed: int = 0,
    stratify: bool = True,
    types: Sequence[str] | None = None,
    min_abstention: int = DEFAULT_MIN_ABSTENTION,
) -> list[Instance]:
    """Deterministically pick a subset that keeps the shape of the benchmark.

    Two guarantees:

    * given the same ``seed``, ``limit`` and inputs, the returned list is
      identical -- candidates are sorted by qid before any random draw, so the
      order the file happened to be written in cannot leak into the sample;
    * abstention questions are never sampled away. They are allocated first, at
      their true share of the dataset but never below ``min_abstention`` and
      never above :data:`MAX_ABSTENTION_SHARE` of the sample, and the remaining
      slots are distributed proportionally across the other question types by
      largest remainder.
    """
    pool = [i for i in instances if not types or i.qtype in set(types)]
    pool.sort(key=lambda i: i.qid)
    if limit is None or limit >= len(pool):
        return pool

    rng = random.Random(seed)
    abstention = [i for i in pool if i.is_abstention]
    answerable = [i for i in pool if not i.is_abstention]

    if not stratify:
        return sorted(rng.sample(pool, limit), key=lambda i: i.qid)

    share = len(abstention) / len(pool) if pool else 0.0
    floor = min(min_abstention, max(1, int(limit * MAX_ABSTENTION_SHARE)))
    want_abs = max(floor, int(round(limit * share)))
    n_abs = min(len(abstention), want_abs, limit)
    picked: list[Instance] = rng.sample(abstention, n_abs) if n_abs else []

    remaining = limit - n_abs
    if remaining > 0 and answerable:
        buckets: dict[str, list[Instance]] = defaultdict(list)
        for item in answerable:
            buckets[item.qtype].append(item)
        quota = _largest_remainder({k: len(v) for k, v in buckets.items()}, remaining)
        for qtype in sorted(buckets):
            take = quota.get(qtype, 0)
            if take:
                picked.extend(rng.sample(buckets[qtype], take))
    return sorted(picked, key=lambda i: i.qid)


# --------------------------------------------------------------------------- #
# public loaders
# --------------------------------------------------------------------------- #


def load_dataset(
    variant: str = "s",
    limit: int | None = None,
    seed: int = 0,
    stratify: bool = True,
    types: list[str] | None = None,
    *,
    allow_large: bool = False,
    min_abstention: int = DEFAULT_MIN_ABSTENTION,
) -> list[Instance]:
    """Load, normalise and sample any supported benchmark.

    Raises :class:`DatasetUnavailable` -- never returns a stand-in -- when the
    requested variant cannot be obtained.
    """
    key = variant.strip().lower()
    if key in {"lme-s", "longmemeval-s"}:
        key = "s"
    if key not in SOURCES:
        raise DatasetUnavailable(
            f"unknown dataset variant {variant!r}; known: {', '.join(sorted(SOURCES))}"
        )
    source = SOURCES[key]

    if source.kind == "lme-chat":
        path = _fetch(source)
        raw = json.loads(path.read_text(encoding="utf-8"))
        _record_manifest(source, path, len(raw))
        instances = normalize_longmemeval(raw, dataset=f"longmemeval-{key}")
    elif source.kind == "lme-v2":
        instances = _load_v2(allow_large=allow_large)
    elif source.kind == "beam-parquet":
        instances = _load_beam(source)
    else:  # pragma: no cover - guarded by the registry
        raise DatasetUnavailable(f"no loader for source kind {source.kind!r}")

    return sample_instances(
        instances,
        limit=limit,
        seed=seed,
        stratify=stratify,
        types=types,
        min_abstention=min_abstention,
    )


def load_longmemeval(
    variant: str = "s",
    limit: int | None = None,
    seed: int = 0,
    stratify: bool = True,
    types: list[str] | None = None,
) -> list[Instance]:
    """Load LongMemEval (or any sibling benchmark) behind one interface."""
    return load_dataset(variant, limit=limit, seed=seed, stratify=stratify, types=types)


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #


def dataset_stats(instances: Iterable[Instance]) -> dict[str, Any]:
    """Describe a loaded sample precisely enough to reproduce and audit it.

    Token figures are estimates (see :data:`TOKEN_ESTIMATOR`) and are labelled as
    such in the returned dict, so a report cannot present them as measured counts.
    """
    items = list(instances)
    if not items:
        return {
            "instances": 0,
            "by_type": {},
            "abstention": {"items": 0, "share": None},
            "sessions": {},
            "tokens_estimated": {"estimator": TOKEN_ESTIMATOR},
        }

    by_type = Counter(i.qtype for i in items)
    abstention = [i for i in items if i.is_abstention]
    sessions = [len(i.sessions) for i in items]
    turns = [i.n_turns for i in items]
    tokens = [i.estimated_tokens() for i in items]
    datasets = sorted({i.dataset for i in items})

    return {
        "datasets": datasets,
        "instances": len(items),
        "by_type": dict(sorted(by_type.items())),
        "abstention": {
            "items": len(abstention),
            "share": round(len(abstention) / len(items), 4),
            "by_type": dict(sorted(Counter(i.qtype for i in abstention).items())),
            "note": (
                "in a sampled run this share may exceed the source dataset's share: "
                f"sampling enforces a floor of {DEFAULT_MIN_ABSTENTION} abstention questions "
                "so the abstention rates have a usable denominator, itself capped at "
                f"{MAX_ABSTENTION_SHARE:.0%} of the sample so a small run is not dominated by them"
            ),
        },
        "sessions": {
            "total": sum(sessions),
            "min": min(sessions),
            "max": max(sessions),
            "mean": round(statistics.mean(sessions), 2),
        },
        "turns": {
            "total": sum(turns),
            "min": min(turns),
            "max": max(turns),
            "mean": round(statistics.mean(turns), 2),
        },
        "tokens_estimated": {
            "estimator": TOKEN_ESTIMATOR,
            "per_instance_min": min(tokens),
            "per_instance_max": max(tokens),
            "per_instance_mean": round(statistics.mean(tokens), 1),
            "total": sum(tokens),
        },
        "evidence_turns_labelled": sum(len(i.evidence_turns()) for i in items),
    }
