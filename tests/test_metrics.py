"""Ranking metrics, judgement resolution and answer-quality measures."""

from __future__ import annotations

import json

import pytest

from rag_platform.data.models import Answer, Chunk, Citation, ScoredChunk, SourceType
from rag_platform.errors import EvaluationError
from rag_platform.evaluation.groundedness import (
    answer_completeness,
    citation_correctness,
    evaluate_answer,
    groundedness_score,
)
from rag_platform.evaluation.harness import bootstrap_ci
from rag_platform.evaluation.qrels import QrelSet, build_span_index, load_qrels
from rag_platform.evaluation.retrieval_metrics import (
    average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarise_run,
    truncate_to_budget,
)

RANKING = ["a", "b", "c", "d", "e"]
RELEVANT = {"c", "e", "z"}


# ---------------------------------------------------------------- ranking


def test_recall_is_not_capped_at_k() -> None:
    # Three items are relevant but only two are reachable in the top five, so
    # recall must reflect the real ceiling rather than being renormalised.
    assert recall_at_k(RANKING, RELEVANT, 5) == pytest.approx(2 / 3)


def test_precision_divides_by_k_not_by_results_returned() -> None:
    assert precision_at_k(["c"], RELEVANT, 5) == pytest.approx(0.2)


def test_reciprocal_rank_uses_the_first_hit() -> None:
    assert reciprocal_rank(RANKING, RELEVANT) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0


def test_ndcg_is_one_for_the_ideal_ranking() -> None:
    assert ndcg_at_k(["c", "e", "z", "a", "b"], RELEVANT, 5) == pytest.approx(1.0)


def test_ndcg_rewards_earlier_hits() -> None:
    assert ndcg_at_k(["c", "a", "b"], {"c"}, 3) > ndcg_at_k(["a", "b", "c"], {"c"}, 3)


def test_average_precision_matches_worked_example() -> None:
    # Hits at ranks 3 and 5 give (1/3 + 2/5) / 3.
    assert average_precision(RANKING, RELEVANT, 5) == pytest.approx((1 / 3 + 2 / 5) / 3)


def test_metrics_are_zero_without_judgements() -> None:
    assert recall_at_k(RANKING, set(), 5) == 0.0
    assert ndcg_at_k(RANKING, set(), 5) == 0.0
    assert average_precision(RANKING, set(), 5) == 0.0


def test_mean_reciprocal_rank_averages_over_queries() -> None:
    assert mean_reciprocal_rank([(["c"], {"c"}), (["x"], {"c"})]) == pytest.approx(0.5)
    assert mean_reciprocal_rank([]) == 0.0


def test_summarise_run_covers_every_requested_k() -> None:
    metrics = summarise_run(RANKING, RELEVANT, [1, 3, 5])
    assert {"recall@1", "recall@3", "recall@5", "ndcg@5", "map@3", "reciprocal_rank"} <= set(
        metrics
    )


def test_truncate_to_budget_respects_the_token_ceiling() -> None:
    assert truncate_to_budget(["a", "b", "c"], [100, 100, 100], 250) == ["a", "b"]


def test_truncate_to_budget_always_returns_one_item() -> None:
    assert truncate_to_budget(["a", "b"], [500, 10], 100) == ["a"]


# -------------------------------------------------------------- bootstrap


def test_bootstrap_interval_brackets_the_mean() -> None:
    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    lower, upper = bootstrap_ci(values, resamples=500)
    assert lower <= sum(values) / len(values) <= upper


def test_bootstrap_on_a_single_observation_is_degenerate() -> None:
    assert bootstrap_ci([0.7]) == (0.7, 0.7)


def test_bootstrap_is_deterministic_given_a_seed() -> None:
    values = [0.1, 0.9, 0.5, 0.3]
    assert bootstrap_ci(values, seed=7, resamples=200) == bootstrap_ci(
        values, seed=7, resamples=200
    )


# ------------------------------------------------------------ groundedness


def _chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        source_type=SourceType.SEC_FILING,
        text=text,
        position=0,
        char_start=0,
        char_end=len(text),
    )


def test_groundedness_is_one_for_a_quoted_claim() -> None:
    evidence = ["Two manufacturers supplied 84 percent of the turbines in our operating fleet."]
    assert (
        groundedness_score("Two manufacturers supplied 84 percent of the turbines.", evidence)
        == 1.0
    )


def test_groundedness_is_zero_for_an_invented_claim() -> None:
    evidence = ["Two manufacturers supplied 84 percent of the turbines."]
    assert (
        groundedness_score("Revenue grew twelve percent in Brazil last quarter.", evidence) == 0.0
    )


def test_groundedness_scores_a_mixed_answer_partially() -> None:
    evidence = ["Two manufacturers supplied 84 percent of the turbines."]
    mixed = "Two manufacturers supplied 84 percent of the turbines. Brazilian revenue tripled overnight."
    assert 0.0 < groundedness_score(mixed, evidence) < 1.0


def test_groundedness_of_empty_text_is_zero() -> None:
    assert groundedness_score("", ["anything"]) == 0.0


def test_citation_correctness_separates_resolvable_from_supported() -> None:
    retrieved = [
        ScoredChunk(chunk=_chunk("Revenue rose 4.2 percent during the year.", "c1"), score=1.0)
    ]
    citations = [
        Citation(marker=1, chunk_id="c1", doc_id="d1", label="x", quote="Revenue rose 4.2 percent"),
        Citation(
            marker=2, chunk_id="c1", doc_id="d1", label="x", quote="Entirely unrelated wording here"
        ),
        Citation(marker=3, chunk_id="missing", doc_id="d1", label="x", quote="Revenue rose"),
    ]
    metrics = citation_correctness(citations, retrieved)
    assert metrics["count"] == 3
    assert metrics["resolvable"] == pytest.approx(2 / 3)
    assert metrics["supported"] == pytest.approx(1 / 3)
    assert metrics["precision"] == pytest.approx(0.5)


def test_citation_correctness_on_no_citations() -> None:
    assert citation_correctness([], [])["count"] == 0


def test_answer_completeness_counts_expected_elements() -> None:
    answer = "Two manufacturers supplied 84 percent of the turbines."
    assert answer_completeness(
        answer, ["two manufacturers", "84 percent", "alternative supplier"]
    ) == pytest.approx(2 / 3)


def test_answer_completeness_does_not_match_a_longer_number() -> None:
    assert answer_completeness("the figure was 0.718", ["0.71"]) == 0.0


def test_answer_completeness_without_expected_terms_is_zero() -> None:
    assert answer_completeness("anything", []) == 0.0


def test_evaluate_answer_reports_every_measure() -> None:
    scored = ScoredChunk(chunk=_chunk("Revenue rose 4.2 percent during the year."), score=1.0)
    answer = Answer(
        query="how much did revenue rise",
        text="Revenue rose 4.2 percent during the year. [1]",
        citations=[
            Citation(
                marker=1, chunk_id="c1", doc_id="d1", label="x", quote="Revenue rose 4.2 percent"
            )
        ],
        supporting_chunks=[scored],
    )
    metrics = evaluate_answer(answer, ["4.2 percent"])
    assert metrics["groundedness"] == 1.0
    assert metrics["citation_supported"] == 1.0
    assert metrics["answer_completeness"] == 1.0


# ------------------------------------------------------------------ qrels


def test_bundled_judgements_resolve_against_the_index(index) -> None:
    qrels = load_qrels(index.config.corpus.qrels_path)
    resolved = qrels.resolve(index.chunks, build_span_index(index.documents))
    assert len(resolved) == len(qrels)
    assert all(ids for ids in resolved.values())


def test_judgements_resolve_under_both_chunking_strategies(config, corpus) -> None:
    from rag_platform.retrieval.index import RetrievalIndex

    qrels = load_qrels(config.corpus.qrels_path)
    for strategy in ("section_aware", "fixed_window"):
        variant = config.with_overrides({"chunking.strategy": strategy})
        built = RetrievalIndex(variant)
        built.build_from_documents(corpus)
        resolved = qrels.resolve(built.chunks, build_span_index(built.documents))
        assert all(ids for ids in resolved.values()), strategy


def test_training_split_is_deterministic_and_disjoint(config) -> None:
    qrels = load_qrels(config.corpus.qrels_path)
    train, test = qrels.training_split(0.5, seed=11)
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(qrels)
    assert (train, test) == qrels.training_split(0.5, seed=11)


def test_training_split_rejects_degenerate_fractions(config) -> None:
    qrels = load_qrels(config.corpus.qrels_path)
    with pytest.raises(EvaluationError):
        qrels.training_split(1.0)


def test_unresolvable_judgement_is_an_error(index) -> None:
    qrels = QrelSet.model_validate(
        {
            "queries": [
                {
                    "query_id": "bad",
                    "query": "anything",
                    "relevant": [{"document": "no_such_file.md", "sections": []}],
                }
            ]
        }
    )
    with pytest.raises(EvaluationError, match="out of sync"):
        qrels.resolve(index.chunks, build_span_index(index.documents))


def test_missing_judgement_file_raises() -> None:
    with pytest.raises(EvaluationError, match="not found"):
        load_qrels("data/fixtures/no-such-file.json")


def test_malformed_judgement_file_raises(tmp_path) -> None:
    path = tmp_path / "qrels.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(EvaluationError, match="could not parse"):
        load_qrels(path)


def test_empty_judgement_file_raises(tmp_path) -> None:
    path = tmp_path / "qrels.json"
    path.write_text(json.dumps({"queries": []}), encoding="utf-8")
    with pytest.raises(EvaluationError, match="no queries"):
        load_qrels(path)
