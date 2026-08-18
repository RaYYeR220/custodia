"""Result JSON to a markdown scorecard a judge can read in one screen.

The single rule this module exists to enforce: **a number that was not measured
prints as ``not measured``.** Not ``0``, not ``0.0%``, not ``-``, not ``n/a``
tucked into a column of percentages where the eye reads it as a low score. Every
metric arrives here as either a float or ``None``, and ``None`` is rendered in
words.

The report also prints its own provenance -- dataset sha256, sample size, seed,
models, judge method, date -- because a scorecard without them is a claim rather
than a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "NOT_MEASURED",
    "pct",
    "num",
    "render_scorecard",
    "render_poison",
    "build_report",
    "main",
]

NOT_MEASURED = "not measured"


def pct(value: float | None, *, digits: int = 1) -> str:
    """A rate as a percentage, or the words ``not measured``."""
    if value is None:
        return NOT_MEASURED
    return f"{value * 100:.{digits}f}%"


def num(value: float | int | None, *, digits: int = 1) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, int):
        return str(value)
    return f"{value:,.{digits}f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    body = list(rows)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    if not body:
        lines.append("| " + " | ".join([NOT_MEASURED] + [""] * (len(headers) - 1)) + " |")
    for row in body:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _as_mapping(card: Any) -> Mapping[str, Any]:
    return card if isinstance(card, Mapping) else card.as_json()


# --------------------------------------------------------------------------- #
# scorecards
# --------------------------------------------------------------------------- #


def render_scorecard(card: Any) -> str:
    """One system's accuracy, abstention and cost, as markdown."""
    data = _as_mapping(card)
    overall = _table(
        ["metric", "value"],
        [
            ["questions scored", num(data.get("n"))],
            ["answerable questions", num(data.get("answerable"))],
            ["accuracy (answerable only)", pct(data.get("accuracy"))],
            ["mean latency", f"{num(data.get('mean_latency_ms'))} ms"],
            ["mean prompt tokens (estimated)", num(data.get("mean_prompt_tokens"))],
            ["truncated inputs", num(data.get("truncated"))],
            ["errors", num(data.get("errors"))],
            ["judge failures", num(data.get("judge_failures"))],
        ],
    )
    by_type = _table(
        ["question type", "n", "correct", "accuracy"],
        [
            [qtype, num(v.get("n")), num(v.get("correct")), pct(v.get("accuracy"))]
            for qtype, v in sorted((data.get("accuracy_by_type") or {}).items())
        ],
    )
    abstention = _table(
        ["metric", "value", "meaning"],
        [
            [
                "abstention items",
                num(data.get("abstention_items")),
                "questions whose answer is genuinely absent from memory",
            ],
            [
                "abstention recall",
                pct(data.get("abstention_recall")),
                "of those, the share the system declined",
            ],
            [
                "abstention precision",
                pct(data.get("abstention_precision")),
                "of everything it declined, the share that genuinely had no answer",
            ],
            [
                "hallucination rate",
                pct(data.get("hallucination_rate")),
                "of the unanswerable questions, the share answered anyway and graded wrong",
            ],
            [
                "over-refusal rate",
                pct(data.get("over_refusal_rate")),
                "answerable questions the system declined (the cost of refusing)",
            ],
        ],
    )
    methods = data.get("judge_methods") or {}
    method_line = ", ".join(f"`{k}` x{v}" for k, v in sorted(methods.items())) or NOT_MEASURED
    return "\n\n".join(
        [
            f"### {data.get('system', 'unknown')}",
            overall,
            "**Accuracy by question type**",
            by_type,
            "**Abstention**",
            abstention,
            f"Graded by: {method_line}. Token counts use the {data.get('token_estimator', NOT_MEASURED)}.",
        ]
    )


def render_poison(card: Any) -> str:
    """The poison table with its negative control beside it, never apart."""
    data = _as_mapping(card)
    headline = _table(
        ["metric", "value", "n", "meaning"],
        [
            [
                "flip rate",
                pct(data.get("flip_rate")),
                num(data.get("attacks")),
                "attacks where the attacker's value appeared in the answer (lower is better)",
            ],
            [
                "quarantine rate",
                pct(data.get("quarantine_rate")),
                num(data.get("attacks", 0) - data.get("no_fact_extracted", 0)),
                "attacks whose injected fact the policy engine caught (higher is better)",
            ],
            [
                "hold rate",
                pct(data.get("hold_rate")),
                num(data.get("attacks")),
                "attacks still answered with the original true value",
            ],
            [
                "legitimate update accuracy",
                pct(data.get("legitimate_update_accuracy")),
                num(data.get("controls")),
                "NEGATIVE CONTROL: genuine owner updates answered with the new value (higher is better)",
            ],
            [
                "over-block rate",
                pct(data.get("over_block_rate")),
                num(data.get("controls")),
                "NEGATIVE CONTROL: genuine updates wrongly quarantined (lower is better)",
            ],
        ],
    )
    families = _table(
        ["family", "kind", "n", "flip rate", "quarantine rate", "abstained"],
        [
            [
                family,
                v.get("kind", ""),
                num(v.get("n")),
                pct(v.get("flip_rate")),
                pct(v.get("quarantine_rate")),
                num(v.get("abstained")),
            ]
            for family, v in sorted((data.get("by_family") or {}).items())
        ],
    )
    caveats = [
        f"- attacks that produced no extractable fact at all: {num(data.get('no_fact_extracted'))} "
        "(excluded from the quarantine denominator - an extractor finding nothing is not the "
        "policy blocking something)",
        f"- errors during the run: {num(data.get('errors'))}",
    ]
    if data.get("suite_sha256"):
        caveats.append(f"- attack suite sha256: `{data['suite_sha256']}`")
    return "\n\n".join(
        [
            f"### {data.get('system', 'unknown')} - poisoning",
            headline,
            "**By attack family**",
            families,
            "\n".join(caveats),
        ]
    )


# --------------------------------------------------------------------------- #
# whole-run report
# --------------------------------------------------------------------------- #


def _render_provenance(provenance: Mapping[str, Any]) -> str:
    rows: list[list[str]] = []
    for key in (
        "run",
        "kind",
        "started_at",
        "finished_at",
        "dataset",
        "dataset_variant",
        "dataset_sha256",
        "dataset_source_url",
        "dataset_items_available",
        "sample_size",
        "seed",
        "stratified",
        "types",
        "systems",
        "answer_model",
        "judge_model",
        "llm_binding",
        "judge_mode",
        "hydra_uri",
        "custodia_version",
        "suite_sha256",
        "suite_path",
    ):
        if key in provenance and provenance[key] not in (None, "", []):
            value = provenance[key]
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(v) for v in value)
            rows.append([key.replace("_", " "), f"`{value}`"])
    for key in sorted(provenance):
        if key.startswith("note"):
            rows.append([key.replace("_", " "), str(provenance[key])])
    return _table(["field", "value"], rows)


def _render_dataset(stats: Mapping[str, Any]) -> str:
    if not stats:
        return NOT_MEASURED
    tokens = stats.get("tokens_estimated") or {}
    sessions = stats.get("sessions") or {}
    abstention = stats.get("abstention") or {}
    rows = [
        ["instances", num(stats.get("instances"))],
        ["abstention items", num(abstention.get("items"))],
        ["abstention share of sample", pct(abstention.get("share"))],
        ["sessions per instance (mean)", num(sessions.get("mean"))],
        ["sessions per instance (min/max)", f"{num(sessions.get('min'))} / {num(sessions.get('max'))}"],
        ["estimated tokens per instance (mean)", num(tokens.get("per_instance_mean"))],
        ["token estimator", tokens.get("estimator", NOT_MEASURED)],
    ]
    by_type = stats.get("by_type") or {}
    table = _table(["field", "value"], rows)
    types = _table(
        ["question type", "instances"],
        [[k, num(v)] for k, v in sorted(by_type.items())],
    )
    note = abstention.get("note")
    parts = [table, "**Sample composition**", types]
    if note:
        parts.append(f"> {note}")
    return "\n\n".join(parts)


def build_report(result: Mapping[str, Any]) -> str:
    """Render a whole result document: provenance, dataset, scorecards, poison."""
    provenance = result.get("provenance") or {}
    kind = provenance.get("kind") or result.get("kind") or "run"
    title = {
        "longmemeval": "Custodia - long-term memory scorecard",
        "poison": "Custodia - memory-poisoning scorecard",
    }.get(kind, "Custodia - evaluation scorecard")

    sections: list[str] = [f"# {title}", "## Provenance", _render_provenance(provenance)]

    if result.get("dataset_stats"):
        sections += ["## Dataset", _render_dataset(result["dataset_stats"])]

    scorecards = result.get("scorecards") or {}
    if scorecards:
        sections.append("## Results")
        summary = _table(
            [
                "system",
                "accuracy",
                "abstention recall",
                "abstention precision",
                "hallucination rate",
                "over-refusal",
                "mean prompt tokens",
                "mean latency (ms)",
            ],
            [
                [
                    name,
                    pct(card.get("accuracy")),
                    pct(card.get("abstention_recall")),
                    pct(card.get("abstention_precision")),
                    pct(card.get("hallucination_rate")),
                    pct(card.get("over_refusal_rate")),
                    num(card.get("mean_prompt_tokens")),
                    num(card.get("mean_latency_ms")),
                ]
                for name, card in sorted(scorecards.items())
            ],
        )
        sections.append(summary)
        for name in sorted(scorecards):
            sections.append(render_scorecard(scorecards[name]))

    poison = result.get("poison") or {}
    if poison:
        sections.append("## Memory poisoning (with negative control)")
        sections.append(
            "> A low flip rate alone proves nothing: a system that refuses every "
            "write scores zero. Read it beside the legitimate-update accuracy."
        )
        summary = _table(
            [
                "system",
                "flip rate",
                "quarantine rate",
                "legitimate update accuracy",
                "over-block rate",
            ],
            [
                [
                    name,
                    pct(card.get("flip_rate")),
                    pct(card.get("quarantine_rate")),
                    pct(card.get("legitimate_update_accuracy")),
                    pct(card.get("over_block_rate")),
                ]
                for name, card in sorted(poison.items())
            ],
        )
        sections.append(summary)
        for name in sorted(poison):
            sections.append(render_poison(poison[name]))

    if result.get("skipped"):
        sections.append("## Skipped")
        sections.append(
            _table(
                ["what", "why"],
                [[str(k), str(v)] for k, v in sorted((result["skipped"] or {}).items())],
            )
        )

    sections.append(
        "---\n\n"
        "`not measured` above means exactly that: the run did not produce the number. "
        "It is never a stand-in for zero."
    )
    return "\n\n".join(sections) + "\n"


def main(result_path: str, out: str | None = None) -> str:
    """Render ``result_path`` to markdown; write it to ``out`` when given."""
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    text = build_report(payload)
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return text


def _cli() -> None:
    import typer

    app = typer.Typer(add_completion=False, help="Render an evaluation result file as markdown.")

    @app.command()
    def render(
        result: str = typer.Argument(..., help="path to a result JSON written by a runner"),
        out: str = typer.Option("", "--out", help="write the markdown here instead of stdout"),
    ) -> None:
        text = main(result, out or None)
        if out:
            typer.echo(f"wrote {out}")
        else:
            typer.echo(text)

    app()


if __name__ == "__main__":
    _cli()
