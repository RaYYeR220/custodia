"""Tests for entity resolution and temporal reconciliation.

The reconciliation tests are written around the failure they exist to prevent:
memory silently deleting something it should have kept. A multi-valued predicate
that supersedes, or an external page allowed to overwrite the principal, both
look like tidy graphs and both are wrong.
"""

from __future__ import annotations

from itertools import permutations

import pytest

from custodia.policy import Policy
from custodia.resolve import (
    MULTI_VALUED,
    Reconciliation,
    canonical_key,
    is_single_valued,
    normalize_entity,
    normalize_predicate,
    reconcile,
    resolve_entities,
    same_object,
)
from custodia.schema import ACTIVE, OPEN_INTERVAL, QUARANTINED, SUPERSEDED, Fact, Tier

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def make(
    subject: str,
    predicate: str,
    obj: str,
    *,
    vfrom: int,
    sidx: int,
    sid: str | None = None,
    tidx: int = 0,
    tier: Tier = Tier.OWNER,
    status: str = ACTIVE,
    conf: float = 1.0,
) -> Fact:
    return Fact(
        corpus="t",
        key=canonical_key(subject, predicate, obj),
        text=f"{subject} {predicate} {obj}",
        subject=subject,
        predicate=predicate,
        object=obj,
        tier=tier,
        status=status,
        valid_from=vfrom,
        sid=sid if sid is not None else f"s{sidx}",
        sidx=sidx,
        tidx=tidx,
        conf=conf,
    )


def pairs(items: list[tuple[Fact, Fact]]) -> list[tuple[str, str]]:
    return [(a.key, b.key) for a, b in items]


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Ironworks' Gym", "ironworks gym"),
        ("  IRONWORKS   gym  ", "ironworks gym"),
        ("Café Zoë", "cafe zoe"),
        ("an_apple-pie", "apple pie"),
        ("Nora's bike", "nora bike"),
        ("the Beatles", "beatles"),
        ("A&M Records, Inc.", "m records inc"),
        ("", ""),
        ("   ", ""),
        ("!!!", ""),
        ("the", "the"),
    ],
)
def test_normalize_entity(raw: str, expected: str) -> None:
    assert normalize_entity(raw) == expected


def test_normalize_predicate() -> None:
    assert normalize_predicate("Works At") == "works_at"
    assert normalize_predicate("works-at") == "works_at"
    assert normalize_predicate("  is a  ") == "is_a"


def test_resolve_entities_dedupes_and_keeps_order() -> None:
    assert resolve_entities(["The Gym", "gym", "Ironworks", ""]) == ["gym", "ironworks"]


def test_resolve_entities_applies_aliases_in_either_spelling() -> None:
    aliases = {"Ironworks Gym": "ironworks"}
    assert resolve_entities(["Ironworks Gym", "ironworks"], aliases=aliases) == ["ironworks"]


def test_canonical_key_is_stable_across_spellings() -> None:
    assert canonical_key("The User", "Works At", "Acme Corp.") == canonical_key(
        "user", "works_at", "acme corp"
    )
    assert canonical_key("a", "b", "c").count("|") == 2


def test_same_object_treats_a_token_subset_as_the_same_value() -> None:
    assert same_object("Ironworks", "Ironworks Gym")
    assert same_object("ironworks gym", "The Ironworks' Gym")
    assert not same_object("Ironworks", "Fitwell")
    assert not same_object("North Gym", "South Gym")


# --------------------------------------------------------------------------- #
# predicate arity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "predicate",
    ["is", "works_at", "lives_in", "prefers", "email", "preferred_airline", "current_gym"],
)
def test_single_valued_predicates(predicate: str) -> None:
    assert is_single_valued(predicate)


@pytest.mark.parametrize("predicate", ["owns", "visited", "knows", "likes", "attended", ""])
def test_multi_valued_predicates(predicate: str) -> None:
    assert not is_single_valued(predicate)


def test_unknown_predicates_default_to_multi_valued() -> None:
    """Defaulting the other way would make memory forget things it was never told to."""
    assert not is_single_valued("collaborated_on")
    assert not is_single_valued("mentioned_during")


def test_multi_valued_list_wins_over_the_single_valued_patterns() -> None:
    assert "uses" in MULTI_VALUED
    assert not is_single_valued("uses")


def test_multi_valued_facts_coexist() -> None:
    bike = make("nora", "owns", "a bike", vfrom=10, sidx=0)
    car = make("nora", "owns", "a car", vfrom=20, sidx=1)
    result = reconcile([bike, car], policy=Policy())

    assert result.supersedes == []
    assert result.contradicts == []
    assert bike.status == car.status == ACTIVE
    assert bike.valid_to == car.valid_to == OPEN_INTERVAL


# --------------------------------------------------------------------------- #
# supersession
# --------------------------------------------------------------------------- #


def test_supersession_chain_closes_each_interval_in_turn() -> None:
    acme = make("nora", "works_at", "Acme", vfrom=10, sidx=0)
    globex = make("nora", "works_at", "Globex", vfrom=20, sidx=1)
    initech = make("nora", "works_at", "Initech", vfrom=30, sidx=2)

    result = reconcile([acme, globex, initech], policy=Policy())

    assert pairs(result.supersedes) == [(globex.key, acme.key), (initech.key, globex.key)]
    assert (acme.status, acme.valid_to) == (SUPERSEDED, 20)
    assert (globex.status, globex.valid_to) == (SUPERSEDED, 30)
    assert (initech.status, initech.valid_to) == (ACTIVE, OPEN_INTERVAL)
    assert [f.key for f in result.updated] == [acme.key, globex.key]


def test_existing_facts_take_part_as_the_older_side() -> None:
    held = make("nora", "lives_in", "Vienna", vfrom=10, sidx=0)
    incoming = make("nora", "lives_in", "Berlin", vfrom=40, sidx=3)

    result = reconcile([incoming], policy=Policy(), existing=[held])

    assert pairs(result.supersedes) == [(incoming.key, held.key)]
    assert result.updated == [held]
    assert held.status == SUPERSEDED


def test_ties_are_broken_by_session_then_turn_index() -> None:
    first = make("nora", "works_at", "Acme", vfrom=10, sidx=0, tidx=1)
    second = make("nora", "works_at", "Globex", vfrom=10, sidx=0, tidx=7)

    result = reconcile([second, first], policy=Policy())

    assert pairs(result.supersedes) == [(second.key, first.key)]
    assert first.status == SUPERSEDED


# --------------------------------------------------------------------------- #
# refusal
# --------------------------------------------------------------------------- #


def test_refused_supersession_becomes_a_contradiction() -> None:
    """An external page may not rewrite what the principal said, so both are kept."""
    owner = make("nora", "lives_in", "Vienna", vfrom=10, sidx=0)
    attacker = make("nora", "lives_in", "Berlin", vfrom=20, sidx=1, tier=Tier.EXTERNAL)

    result = reconcile([owner, attacker], policy=Policy())

    assert result.supersedes == []
    assert pairs(result.contradicts) == [(attacker.key, owner.key)]
    assert (owner.status, owner.valid_to) == (ACTIVE, OPEN_INTERVAL)
    assert attacker.status == QUARANTINED
    assert attacker.quarantine_reason
    assert [r[2].rule for r in result.refusals] == ["tier-floor"]


def test_identity_forgery_refusal_is_named_ahead_of_the_tier_floor() -> None:
    """A specific diagnosis beats the general backstop when both would fire."""
    owner = make("user", "lives_in", "Vienna", vfrom=10, sidx=0)
    tool = make("user", "lives_in", "Berlin", vfrom=20, sidx=1, tier=Tier.TOOL)

    result = reconcile([owner, tool], policy=Policy())

    assert pairs(result.contradicts) == [(tool.key, owner.key)]
    assert [r[2].rule for r in result.refusals] == ["identity-forgery"]
    assert owner.status == ACTIVE


def test_a_higher_tier_write_still_supersedes_a_lower_tier_fact() -> None:
    tool = make("acme corp", "located_in", "Vienna", vfrom=10, sidx=0, tier=Tier.TOOL)
    owner = make("acme corp", "located_in", "Berlin", vfrom=20, sidx=1)

    result = reconcile([tool, owner], policy=Policy())

    assert pairs(result.supersedes) == [(owner.key, tool.key)]
    assert tool.status == SUPERSEDED


def test_a_quarantined_fact_never_supersedes() -> None:
    owner = make("acme corp", "located_in", "Vienna", vfrom=10, sidx=0)
    poisoned = make(
        "acme corp", "located_in", "Berlin", vfrom=20, sidx=1, status=QUARANTINED
    )

    result = reconcile([owner, poisoned], policy=Policy())

    assert result.supersedes == []
    assert pairs(result.contradicts) == [(poisoned.key, owner.key)]
    assert result.refusals == []          # ingest already raised the rejection
    assert owner.status == ACTIVE


def test_research_mode_lets_the_supersession_through() -> None:
    owner = make("acme corp", "located_in", "Vienna", vfrom=10, sidx=0)
    tool = make("acme corp", "located_in", "Berlin", vfrom=20, sidx=1, tier=Tier.TOOL)

    result = reconcile([owner, tool], policy=Policy(strict=False))

    assert pairs(result.supersedes) == [(tool.key, owner.key)]
    assert owner.status == SUPERSEDED


# --------------------------------------------------------------------------- #
# corroboration
# --------------------------------------------------------------------------- #


def test_corroboration_across_sessions_raises_confidence() -> None:
    first = make("nora", "goes_to", "Ironworks", vfrom=10, sidx=0, sid="s0", conf=0.8)
    later = make("nora", "goes_to", "Ironworks Gym", vfrom=40, sidx=2, sid="s2", conf=0.8)

    result = reconcile([first, later], policy=Policy())

    assert pairs(result.corroborates) == [(later.key, first.key)]
    assert result.supersedes == []
    assert first.conf > 0.8 and later.conf > 0.8
    assert first.status == later.status == ACTIVE
    assert result.updated == [first]


def test_confidence_is_capped_at_one() -> None:
    first = make("nora", "goes_to", "Ironworks", vfrom=10, sidx=0, sid="s0")
    later = make("nora", "goes_to", "Ironworks Gym", vfrom=40, sidx=2, sid="s2")

    reconcile([first, later], policy=Policy())

    assert first.conf == 1.0 and later.conf == 1.0


def test_repetition_inside_one_session_is_not_corroboration() -> None:
    first = make("nora", "goes_to", "Ironworks", vfrom=10, sidx=0, sid="s0", conf=0.8)
    again = make("nora", "goes_to", "Ironworks Gym", vfrom=20, sidx=0, sid="s0", conf=0.8)

    result = reconcile([first, again], policy=Policy())

    assert result.corroborates == []
    assert first.conf == 0.8


def test_identical_triples_are_one_fact_and_never_self_corroborate() -> None:
    """Same key, same vertex id -- a repeat is more provenance, not a second claim."""
    first = make("nora", "goes_to", "Ironworks", vfrom=10, sidx=0, sid="s0", conf=0.8)
    repeat = make("nora", "goes_to", "Ironworks", vfrom=40, sidx=2, sid="s2", conf=0.8)
    assert first.key == repeat.key

    result = reconcile([first, repeat], policy=Policy())

    assert result.corroborates == []
    assert first.conf > 0.8                       # the repeat still counts
    assert first.valid_from == 10                 # but the claim started earlier


def test_corroboration_survives_a_later_supersession() -> None:
    first = make("nora", "works_at", "Acme", vfrom=10, sidx=0, sid="s0", conf=0.5)
    again = make("nora", "works_at", "Acme Corp", vfrom=20, sidx=1, sid="s1", conf=0.5)
    moved = make("nora", "works_at", "Globex", vfrom=30, sidx=2, sid="s2")

    result = reconcile([first, again, moved], policy=Policy())

    assert pairs(result.corroborates) == [(again.key, first.key)]
    assert set(pairs(result.supersedes)) == {(moved.key, first.key), (moved.key, again.key)}
    assert first.status == again.status == SUPERSEDED
    assert moved.status == ACTIVE


# --------------------------------------------------------------------------- #
# order independence
# --------------------------------------------------------------------------- #


def _corpus() -> list[Fact]:
    return [
        make("nora", "works_at", "Acme", vfrom=10, sidx=0, sid="s0"),
        make("nora", "works_at", "Globex", vfrom=20, sidx=1, sid="s1"),
        make("nora", "works_at", "Initech", vfrom=30, sidx=2, sid="s2"),
        make("nora", "owns", "a bike", vfrom=5, sidx=0, sid="s0"),
        make("nora", "owns", "a car", vfrom=25, sidx=1, sid="s1"),
        make("nora", "lives_in", "Berlin", vfrom=40, sidx=3, sid="s3", tier=Tier.EXTERNAL),
        make("nora", "lives_in", "Vienna", vfrom=15, sidx=0, sid="s0"),
    ]


def _fingerprint(result: Reconciliation, facts: list[Fact]) -> tuple:
    return (
        tuple(sorted(pairs(result.supersedes))),
        tuple(sorted(pairs(result.contradicts))),
        tuple(sorted(pairs(result.corroborates))),
        tuple(sorted((f.key, f.status, f.valid_to, f.conf) for f in facts)),
    )


def test_reconciliation_is_order_independent() -> None:
    baseline = None
    for order in permutations(range(7)):
        facts = _corpus()
        shuffled = [facts[i] for i in order]
        result = reconcile(shuffled, policy=Policy())
        fingerprint = _fingerprint(result, facts)
        if baseline is None:
            baseline = fingerprint
        assert fingerprint == baseline


def test_reconciliation_is_idempotent_when_run_twice() -> None:
    facts = _corpus()
    first = _fingerprint(reconcile(facts, policy=Policy()), facts)
    second = _fingerprint(reconcile(facts, policy=Policy()), facts)
    assert first == second


def test_counts_summarise_the_reconciliation() -> None:
    facts = _corpus()
    result = reconcile(facts, policy=Policy())
    counts = result.counts()
    assert counts["supersedes"] == len(result.supersedes)
    assert counts["contradicts"] == len(result.contradicts)
    assert counts["corroborates"] == len(result.corroborates)
    assert counts["refusals"] == len(result.refusals)
