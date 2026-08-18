"""The BM25 index: tokenisation, ranking sanity, persistence, and empty."""

from __future__ import annotations

import json

import pytest

from custodia.lexical import LexicalIndex, index_path, tokenize, words

DOCS = [
    (1, "The user's gym is Northline Fitness in Zurich."),
    (2, "The user drinks oat milk in their coffee."),
    (3, "The user's squat rack access code is 4417."),
    (4, "The user is training for a marathon in Zurich this spring."),
    (5, "The user prefers morning workouts at the gym."),
]


@pytest.fixture()
def index() -> LexicalIndex:
    return LexicalIndex.from_documents("test", DOCS)


# ---- tokenizer ------------------------------------------------------------- #


def test_words_are_lowercased_and_split_on_punctuation():
    assert words("The user's gym -- Northline/Fitness!") == [
        "the",
        "user",
        "s",
        "gym",
        "northline",
        "fitness",
    ]


def test_words_keep_digits_and_accented_letters():
    assert words("café 4417 zürich_west") == ["café", "4417", "zürich", "west"]


def test_tokenize_drops_stopwords_and_question_words():
    assert "the" not in tokenize("the user")
    assert "where" not in tokenize("where does the user live")
    assert tokenize("the and or of") == []


def test_tokenize_collapses_plural_and_verb_forms():
    for surface in ("like", "likes", "liked", "liking"):
        assert tokenize(surface) == tokenize("like"), surface
    for surface in ("gym", "gyms"):
        assert tokenize(surface) == tokenize("gym"), surface
    assert tokenize("cities") == tokenize("city")


def test_tokenize_leaves_short_words_and_numbers_alone():
    assert tokenize("4417") == ["4417"]
    assert tokenize("gas") == ["gas"]


# ---- ranking --------------------------------------------------------------- #


def test_search_finds_the_document_that_contains_the_term(index: LexicalIndex):
    top = index.search("squat rack access code", k=3)
    assert top
    assert top[0][0] == 3


def test_scores_are_normalised_to_the_unit_interval(index: LexicalIndex):
    hits = index.search("gym zurich", k=5)
    assert hits[0][1] == pytest.approx(1.0)
    assert all(0.0 <= score <= 1.0 for _, score in hits)


def test_scores_are_monotonically_ordered(index: LexicalIndex):
    hits = index.search("gym", k=5)
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_a_rare_term_outranks_a_common_one(index: LexicalIndex):
    # "user" appears in every document and carries almost no information;
    # "marathon" appears once. A query with both must be led by the rare term.
    top = index.search("user marathon", k=5)
    assert top[0][0] == 4


def test_a_document_matching_two_query_terms_beats_one_matching_one(index: LexicalIndex):
    hits = dict(index.search("gym zurich", k=5))
    assert hits[1] > hits[5]  # doc 1 has both, doc 5 only "gym"


def test_search_ignores_a_query_of_only_stopwords(index: LexicalIndex):
    assert index.search("what is the", k=5) == []


def test_search_returns_nothing_for_an_unseen_term(index: LexicalIndex):
    assert index.search("helicopter", k=5) == []


def test_search_respects_k(index: LexicalIndex):
    assert len(index.search("user", k=2)) == 2


def test_search_is_stable_across_equal_scores():
    twins = LexicalIndex.from_documents("test", [(9, "alpha beta"), (7, "alpha beta")])
    assert [fid for fid, _ in twins.search("alpha beta", k=2)] == [7, 9]


# ---- persistence ----------------------------------------------------------- #


def test_save_and_load_round_trip(index: LexicalIndex, tmp_path):
    path = tmp_path / "test.json"
    index.save(path)
    restored = LexicalIndex.load(path)

    assert len(restored) == len(index)
    assert restored.corpus == index.corpus
    assert restored.postings == index.postings
    assert restored.search("gym zurich", k=5) == index.search("gym zurich", k=5)


def test_saved_file_is_plain_json(index: LexicalIndex, tmp_path):
    path = tmp_path / "test.json"
    index.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == 1
    assert payload["ids"] == [1, 2, 3, 4, 5]


def test_save_creates_missing_directories(index: LexicalIndex, tmp_path):
    path = tmp_path / "deep" / "deeper" / "test.json"
    index.save(path)
    assert path.exists()


def test_load_rejects_an_unknown_format(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format": 99}), encoding="utf-8")
    with pytest.raises(ValueError):
        LexicalIndex.load(path)


def test_open_returns_none_when_no_index_was_written():
    assert LexicalIndex.open("a-corpus-that-was-never-indexed") is None


def test_index_path_sanitises_the_corpus_name():
    assert index_path("acme/prod:1").name == "acme_prod_1.json"


# ---- empty ----------------------------------------------------------------- #


def test_empty_index_searches_without_error():
    empty = LexicalIndex(corpus="test")
    assert len(empty) == 0
    assert empty.search("anything", k=5) == []


def test_empty_index_round_trips(tmp_path):
    path = tmp_path / "empty.json"
    LexicalIndex(corpus="test").save(path)
    assert len(LexicalIndex.load(path)) == 0
