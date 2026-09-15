"""Tokenisation, sentence splitting, stemming and overlap measures."""

from __future__ import annotations

import pytest

from rag_platform.utils.text import (
    containment,
    content_tokens,
    jaccard_overlap,
    normalise_whitespace,
    split_sentences,
    stable_hash,
    stem,
    tokenize,
    truncate,
)
from rag_platform.utils.timing import Stopwatch, latency_summary, percentile


def test_normalise_whitespace_collapses_runs() -> None:
    assert normalise_whitespace("a  \t b\n\n\n\nc ") == "a b\n\nc"


def test_tokenize_keeps_numbers_and_decimals() -> None:
    assert tokenize("Revenue rose 4.2 percent") == ["revenue", "rose", "4.2", "percent"]


def test_tokenize_drops_stopwords_when_asked() -> None:
    assert "the" not in tokenize("the revenue", drop_stopwords=True)


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("turbines", "turbine"),
        ("suppliers", "supplier"),
        ("editing", "edit"),
        ("reported", "report"),
        ("increased", "increase"),
        ("analyses", "analysis"),
        ("crises", "crisis"),
        ("losses", "loss"),
        # Words whose ending is part of the stem must survive untouched.
        ("basis", "basis"),
        ("business", "business"),
        ("calibration", "calibration"),
    ],
)
def test_stem_is_conservative(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_stemming_makes_inflected_forms_match() -> None:
    assert (
        containment(content_tokens("turbine supplier"), content_tokens("the suppliers of turbines"))
        == 1.0
    )


def test_split_sentences_preserves_abbreviations() -> None:
    text = "Apple Inc. reported revenue of $383.3 billion. Net income rose. Dr. Smith agreed."
    assert len(split_sentences(text)) == 3


def test_split_sentences_on_empty_input() -> None:
    assert split_sentences("   ") == []


def test_containment_is_asymmetric() -> None:
    claim = content_tokens("revenue rose")
    evidence = content_tokens("total revenue rose sharply during the period")
    assert containment(claim, evidence) == 1.0
    assert containment(evidence, claim) < 1.0


def test_jaccard_overlap_bounds() -> None:
    assert jaccard_overlap(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard_overlap(["a"], ["b"]) == 0.0
    assert jaccard_overlap([], ["b"]) == 0.0


def test_stable_hash_is_deterministic_and_collision_resistant() -> None:
    assert stable_hash("a", "b") == stable_hash("a", "b")
    assert stable_hash("a", "b") != stable_hash("ab", "")


def test_truncate_breaks_on_word_boundary() -> None:
    assert truncate("one two three four", 12).endswith("...")
    assert truncate("short", 20) == "short"


def test_percentile_matches_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0


def test_percentile_rejects_out_of_range_q() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_latency_summary_handles_empty_input() -> None:
    assert latency_summary([])["count"] == 0


def test_stopwatch_measures_elapsed_time() -> None:
    with Stopwatch() as sw:
        sum(range(10000))
    assert sw.elapsed_ms >= 0.0
