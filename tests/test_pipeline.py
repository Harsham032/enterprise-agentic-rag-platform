"""End-to-end behaviour: build, retrieve, answer, evaluate.

These tests assert the properties the whole system is supposed to guarantee,
rather than the behaviour of any one component. They are the ones that fail when
two correct components are wired together wrongly.
"""

from __future__ import annotations

import pytest

from rag_platform.agents.orchestrator import Orchestrator
from rag_platform.config import PipelineConfig
from rag_platform.errors import IndexNotBuiltError
from rag_platform.evaluation.harness import evaluate_answers, evaluate_retrieval, run_configuration
from rag_platform.evaluation.qrels import build_span_index, load_qrels
from rag_platform.retrieval.index import RetrievalIndex


def test_index_reports_its_contents(index: RetrievalIndex) -> None:
    described = index.describe()
    assert described["built"] is True
    assert described["documents"] == 20
    assert described["chunks"] > described["documents"]
    assert set(described["sources"]) == {"sec_filing", "biomedical_abstract", "user_document"}


def test_searching_an_unbuilt_index_raises(config: PipelineConfig) -> None:
    with pytest.raises(IndexNotBuiltError):
        RetrievalIndex(config).search("anything")


def test_building_from_no_documents_raises(config: PipelineConfig) -> None:
    with pytest.raises(IndexNotBuiltError):
        RetrievalIndex(config).build_from_documents([])


def test_index_build_is_deterministic(config: PipelineConfig, corpus) -> None:
    """Two builds of the same corpus must rank identically, or no result is reproducible."""
    first, second = RetrievalIndex(config), RetrievalIndex(config)
    first.build_from_documents(corpus)
    second.build_from_documents(corpus)
    query = "turbine supplier concentration"
    assert [c.chunk_id for c in first.search(query)] == [c.chunk_id for c in second.search(query)]


@pytest.mark.parametrize(
    ("query", "expected_source_file"),
    [
        (
            "Which turbine supplier concentration risk does Northwind Energy disclose?",
            "northwind_energy_10k_fy2024.md",
        ),
        (
            "What was the objective response rate for claudin-18.2 CAR T cells?",
            "car_t_solid_tumour.md",
        ),
        (
            "What is the most common production failure of the retrieval service?",
            "retrieval_service_runbook.md",
        ),
        (
            "Which carbapenemase gene became more prevalent?",
            "antimicrobial_resistance_surveillance.md",
        ),
    ],
)
def test_known_queries_retrieve_their_source_document(
    index: RetrievalIndex, query: str, expected_source_file: str
) -> None:
    files = {result.chunk.metadata.source_file for result in index.search(query, top_k=5)}
    assert expected_source_file in files


def test_every_answer_citation_resolves_to_returned_evidence(index: RetrievalIndex, config) -> None:
    """The property that makes a citation worth printing."""
    orchestrator = Orchestrator(index, config)
    qrels = load_qrels(config.corpus.qrels_path)
    for judgement in qrels.queries[:12]:
        answer = orchestrator.answer(judgement.query)
        evidence_ids = {scored.chunk_id for scored in answer.supporting_chunks}
        assert all(
            citation.chunk_id in evidence_ids for citation in answer.citations
        ), judgement.query_id
        assert all(citation.verified for citation in answer.citations), judgement.query_id


def test_extractive_answers_are_fully_grounded(index: RetrievalIndex, config) -> None:
    """Every sentence is lifted from a retrieved chunk, so groundedness is 1.0 by construction."""
    orchestrator = Orchestrator(index, config)
    qrels = load_qrels(config.corpus.qrels_path)
    for judgement in qrels.queries[:12]:
        assert orchestrator.answer(judgement.query).groundedness == pytest.approx(1.0)


def test_an_out_of_domain_question_never_invents_content(index: RetrievalIndex, config) -> None:
    """The guarantee is not that the system refuses, but that it never fabricates.

    Nothing in the corpus answers this question. The extractive path will still
    return the closest passages it found, and that is acceptable - what is not
    acceptable is a sentence that appears nowhere in the evidence. Groundedness
    is exactly that check.
    """
    from rag_platform.evaluation.groundedness import groundedness_score

    answer = Orchestrator(index, config).answer(
        "What is the boiling point of liquid helium at sea level?"
    )
    evidence = [scored.chunk.text for scored in answer.supporting_chunks]
    assert groundedness_score(answer.text, evidence) == pytest.approx(1.0)
    assert all(citation.verified for citation in answer.citations)


def test_comparison_queries_gather_evidence_from_several_documents(
    index: RetrievalIndex, config
) -> None:
    answer = Orchestrator(index, config).answer(
        "Compare Northwind Energy revenue growth in 2024 against 2023"
    )
    assert answer.diagnostics["strategy"] == "temporal_comparison"
    assert len({scored.chunk.doc_id for scored in answer.supporting_chunks}) > 1


def test_ingesting_a_document_makes_it_retrievable(config: PipelineConfig, corpus) -> None:
    from rag_platform.data.models import Document, DocumentMetadata, SourceType

    built = RetrievalIndex(config)
    built.build_from_documents(corpus)
    before = len(built.documents)

    built.add_documents(
        [
            Document.create(
                source_type=SourceType.USER_DOCUMENT,
                text=(
                    "The escalation rota assigns a secondary on-call engineer for every "
                    "weekend shift. The secondary is paged only when the primary does not "
                    "acknowledge within ten minutes."
                ),
                metadata=DocumentMetadata(title="Escalation Rota"),
                natural_key="escalation-rota",
            )
        ]
    )
    assert len(built.documents) == before + 1
    results = built.search("When is the secondary on-call engineer paged?", top_k=3)
    assert any("secondary" in result.chunk.text for result in results)


def test_reranker_training_uses_only_the_training_queries(config: PipelineConfig, corpus) -> None:
    built = RetrievalIndex(config)
    built.build_from_documents(corpus)
    qrels = load_qrels(config.corpus.qrels_path)
    train_ids, _ = qrels.training_split(0.5, seed=config.run.seed)
    spans = build_span_index(built.documents)
    training = {
        judgement.query: judgement.relevant_chunk_ids(built.chunks, spans)
        for judgement in qrels.queries
        if judgement.query_id in set(train_ids)
    }
    coefficients = built.fit_reranker(training)
    assert coefficients
    assert len(training) == len(train_ids)


def test_hybrid_retrieval_is_at_least_as_good_as_either_half(
    config: PipelineConfig, corpus
) -> None:
    """The headline claim of the architecture, checked on the held-out split."""
    qrels = load_qrels(config.corpus.qrels_path)
    train_ids, test_ids = qrels.training_split(0.5, seed=config.run.seed)

    scores = {}
    for name, overrides in (
        ("lexical", {"retrieval.lexical_weight": 1.0, "retrieval.dense_weight": 0.0}),
        ("dense", {"retrieval.lexical_weight": 0.0, "retrieval.dense_weight": 1.0}),
        ("hybrid", {}),
    ):
        variant = config.with_overrides({**overrides, "reranking.enabled": False})
        built = RetrievalIndex(variant)
        built.build_from_documents(corpus)
        scores[name] = evaluate_retrieval(built, qrels, query_ids=test_ids)["metrics"][
            "recall@budget"
        ]

    assert scores["hybrid"] >= max(scores["lexical"], scores["dense"])


def test_run_configuration_produces_a_complete_report(config: PipelineConfig, corpus) -> None:
    qrels = load_qrels(config.corpus.qrels_path)
    train_ids, test_ids = qrels.training_split(0.5, seed=config.run.seed)
    result = run_configuration("smoke", config, corpus, qrels, train_ids, test_ids[:6])

    assert result["retrieval"]["queries"] == 6
    assert "recall@budget" in result["retrieval"]["metrics"]
    assert "recall@budget" in result["retrieval"]["confidence_intervals"]
    assert result["answers"]["metrics"]["groundedness"] > 0.0
    assert result["index"]["chunks"] > 0
    assert set(result["config"]) == {
        "chunking",
        "embedding",
        "retrieval",
        "reranking",
        "generation",
    }


def test_answer_evaluation_reports_latency(index: RetrievalIndex, config) -> None:
    qrels = load_qrels(config.corpus.qrels_path)
    _, test_ids = qrels.training_split(0.5, seed=config.run.seed)
    result = evaluate_answers(Orchestrator(index, config), qrels, query_ids=test_ids[:5], warmup=1)
    assert result["queries"] == 5
    assert result["latency"]["p95_ms"] > 0.0
    assert 0.0 <= result["metrics"]["answer_completeness"] <= 1.0
