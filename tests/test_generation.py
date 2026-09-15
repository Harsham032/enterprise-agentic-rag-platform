"""Query planning, tool routing, answer synthesis and citation verification."""

from __future__ import annotations

import pytest

from rag_platform.agents.orchestrator import merge_results
from rag_platform.agents.planner import PlanningConfig, QueryPlanner, route
from rag_platform.agents.tools import ToolRegistry, ToolResult
from rag_platform.config import GenerationConfig
from rag_platform.data.models import Chunk, Citation, ScoredChunk, SourceType
from rag_platform.generation.citations import (
    build_citations,
    extract_markers,
    strip_unresolvable_markers,
    verify_citations,
)
from rag_platform.generation.synthesizer import (
    INSUFFICIENT_EVIDENCE,
    ExtractiveSynthesizer,
    LLMSynthesizer,
    build_synthesizer,
)


def _scored(text: str, chunk_id: str, rank: int = 1, heading: str | None = None) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_type=SourceType.SEC_FILING,
        text=text,
        position=0,
        section_heading=heading,
        char_start=0,
        char_end=len(text),
    )
    return ScoredChunk(chunk=chunk, score=1.0 / rank, rank=rank)


# ---------------------------------------------------------------- planning


def test_temporal_comparison_splits_by_period() -> None:
    plan = QueryPlanner(PlanningConfig()).plan(
        "Compare Northwind Energy revenue growth in 2024 against 2023"
    )
    assert plan.strategy == "temporal_comparison"
    assert len(plan.subqueries) == 2
    assert {"2024", "2023"} <= {word for sq in plan.subqueries for word in sq.text.split()}


def test_entity_comparison_splits_by_company() -> None:
    plan = QueryPlanner(PlanningConfig()).plan(
        "Compare Meridian Semiconductor against Atlas Payments on supplier concentration"
    )
    assert plan.strategy == "entity_comparison"
    texts = [sq.text for sq in plan.subqueries]
    assert any(t.startswith("Meridian Semiconductor") for t in texts)
    assert any(t.startswith("Atlas Payments") for t in texts)
    # The leading comparison verb must not be absorbed into the entity name.
    assert not any(t.startswith("Compare") for t in texts)


def test_conjunction_splits_two_questions() -> None:
    plan = QueryPlanner(PlanningConfig()).plan(
        "What was the vaccine efficacy and how many patients were enrolled?"
    )
    assert plan.strategy == "conjunction"
    assert len(plan.subqueries) == 2


def test_single_hop_query_is_not_decomposed() -> None:
    plan = QueryPlanner(PlanningConfig()).plan(
        "Which turbine supplier risk does Northwind Energy disclose?"
    )
    assert plan.strategy == "single_hop"
    assert len(plan.subqueries) == 1


def test_planning_can_be_disabled() -> None:
    plan = QueryPlanner(PlanningConfig(enabled=False)).plan("Compare 2023 against 2024 revenue")
    assert plan.is_decomposed is False
    assert len(plan.subqueries) == 1


def test_max_subqueries_is_respected() -> None:
    plan = QueryPlanner(PlanningConfig(max_subqueries=1)).plan(
        "Compare revenue in 2022, 2023 and 2024"
    )
    assert len(plan.subqueries) == 1


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What was EBITDA margin in the fiscal year", "financial_lookup"),
        ("How many patients were randomised in the clinical trial", "biomedical_search"),
        ("What does the runbook say about the rebuild window", "document_search"),
    ],
)
def test_routing_picks_the_specialised_tool(query: str, expected: str) -> None:
    assert route(query) == expected


def test_routing_handles_inflected_vocabulary() -> None:
    # The routing vocabularies are stemmed, so plurals must still route.
    assert route("how many patients were enrolled") == "biomedical_search"


# ------------------------------------------------------------------- tools


def test_tool_registry_exposes_every_tool(index) -> None:
    assert ToolRegistry(index).names == [
        "biomedical_search",
        "document_search",
        "financial_lookup",
        "sql_lookup",
    ]


def test_financial_lookup_returns_only_filings(index) -> None:
    result = ToolRegistry(index).call("financial_lookup", "revenue growth", 5)
    assert result.chunks
    assert all(c.chunk.source_type is SourceType.SEC_FILING for c in result.chunks)


def test_biomedical_search_returns_only_abstracts(index) -> None:
    result = ToolRegistry(index).call("biomedical_search", "clinical trial endpoint", 5)
    assert result.chunks
    assert all(c.chunk.source_type is SourceType.BIOMEDICAL_ABSTRACT for c in result.chunks)


def test_unknown_tool_degrades_to_document_search(index) -> None:
    assert ToolRegistry(index).call("no_such_tool", "revenue", 3).tool == "document_search"


def test_a_failing_tool_does_not_raise(index) -> None:
    registry = ToolRegistry(index)

    def explode(query: str, top_k: int, entity: str | None = None) -> ToolResult:
        raise RuntimeError("backend down")

    registry._tools["document_search"] = explode  # noqa: SLF001
    result = registry.call("document_search", "anything", 3)
    assert result.error == "backend down"
    assert result.chunks == []


def test_structured_facts_are_attached_when_a_lookup_is_wired(index) -> None:
    def lookup(cik: str | None, year: int | None) -> list[dict[str, object]]:
        return [{"cik": cik, "fiscal_year": year, "concept": "Revenues", "value": 1.0}]

    result = ToolRegistry(index, fact_lookup=lookup).call("financial_lookup", "revenue in 2024", 5)
    assert result.structured
    assert result.structured[0]["fiscal_year"] == 2024


# ------------------------------------------------------------------- merge


def test_merge_deduplicates_and_keeps_the_best_score() -> None:
    merged = merge_results(
        [
            ToolResult(tool="a", query="q", chunks=[_scored("text one", "c1", 1)]),
            ToolResult(
                tool="b",
                query="q",
                chunks=[_scored("text one", "c1", 4), _scored("text two", "c2", 2)],
            ),
        ],
        top_k=5,
    )
    assert [c.chunk_id for c in merged] == ["c1", "c2"]
    assert merged[0].score == pytest.approx(1.0)
    assert [c.rank for c in merged] == [1, 2]


def test_merge_respects_top_k() -> None:
    chunks = [_scored(f"text {i}", f"c{i}", i + 1) for i in range(10)]
    assert len(merge_results([ToolResult(tool="a", query="q", chunks=chunks)], top_k=3)) == 3


# --------------------------------------------------------------- citations


def test_build_citations_numbers_chunks_in_order_of_first_use() -> None:
    one, two = _scored("alpha text here", "c1"), _scored("beta text here", "c2")
    citations, markers = build_citations(
        [("alpha text here", one), ("beta text here", two), ("alpha again", one)]
    )
    assert markers == {"c1": 1, "c2": 2}
    assert len(citations) == 2


def test_verify_citations_flags_an_unsupported_quote() -> None:
    retrieved = [_scored("Revenue rose 4.2 percent during the year.", "c1")]
    citations = [
        Citation(marker=1, chunk_id="c1", doc_id="d", label="x", quote="Revenue rose 4.2 percent"),
        Citation(
            marker=2, chunk_id="c1", doc_id="d", label="x", quote="Completely different wording"
        ),
        Citation(marker=3, chunk_id="absent", doc_id="d", label="x", quote="Revenue rose"),
    ]
    verified = verify_citations(citations, retrieved)
    assert [c.verified for c in verified] == [True, False, False]
    assert verified[0].support_score == pytest.approx(1.0)


def test_extract_markers_reads_them_in_order() -> None:
    assert extract_markers("claim [2] and claim [1] again [2]") == [2, 1, 2]


def test_strip_unresolvable_markers_removes_invented_references() -> None:
    cleaned = strip_unresolvable_markers("real [1] invented [9]", {1})
    assert "[1]" in cleaned
    assert "[9]" not in cleaned


# -------------------------------------------------------------- synthesis


def test_extractive_answer_quotes_evidence_and_cites_it() -> None:
    evidence = [
        _scored(
            "We depend on a concentrated set of turbine suppliers. Two manufacturers "
            "supplied 84 percent of the turbines in our operating fleet.",
            "c1",
            1,
            "Item 1A. Risk Factors",
        )
    ]
    answer = ExtractiveSynthesizer(GenerationConfig()).synthesise(
        "turbine supplier concentration", evidence
    )
    assert "84 percent" in answer.text
    assert answer.citations and all(c.verified for c in answer.citations)
    assert answer.groundedness == pytest.approx(1.0)


def test_extractive_answer_without_evidence_declines_to_answer() -> None:
    answer = ExtractiveSynthesizer(GenerationConfig()).synthesise("anything", [])
    assert answer.text == INSUFFICIENT_EVIDENCE
    assert answer.groundedness == 0.0
    assert answer.citations == []


def test_spans_pick_up_detail_following_a_lead_in_sentence() -> None:
    text = (
        "Three alerts page the on-call engineer. First, groundedness below 0.80 "
        "averaged over a rolling one-hour window. Second, latency above 1,500 milliseconds."
    )
    evidence = [_scored(text, "c1", 1, "Alerting")]
    answer = ExtractiveSynthesizer(GenerationConfig(span_sentences=2)).synthesise(
        "What alert thresholds page the on-call engineer?", evidence
    )
    # A single-sentence unit would stop at the lead-in and never reach a threshold.
    assert "0.80" in answer.text


def test_span_budget_is_respected() -> None:
    sentences = " ".join(f"Sentence {i} mentions revenue and margin explicitly." for i in range(20))
    evidence = [_scored(sentences, "c1", 1)]
    answer = ExtractiveSynthesizer(GenerationConfig(max_spans=2)).synthesise(
        "revenue margin", evidence
    )
    assert answer.diagnostics["selected_spans"] <= 2


def test_selection_prefers_the_on_topic_chunk() -> None:
    evidence = [
        _scored("Warehouse throughput rose after the new conveyor was installed.", "c1", 1),
        _scored("Two manufacturers supplied 84 percent of the turbines in the fleet.", "c2", 2),
    ]
    answer = ExtractiveSynthesizer(GenerationConfig(max_spans=1)).synthesise(
        "turbine supplier share", evidence
    )
    assert "turbines" in answer.text


def test_llm_synthesizer_strips_invented_markers() -> None:
    class FakeClient:
        def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
            return "Revenue rose 4.2 percent [1] and margins doubled [7]."

    evidence = [_scored("Revenue rose 4.2 percent during the year.", "c1", 1)]
    answer = LLMSynthesizer(GenerationConfig(backend="anthropic"), FakeClient()).synthesise(
        "revenue", evidence
    )
    assert "[1]" in answer.text
    assert "[7]" not in answer.text
    assert answer.diagnostics["mode"] == "abstractive"


def test_build_synthesizer_defaults_to_extractive() -> None:
    assert isinstance(build_synthesizer(GenerationConfig()), ExtractiveSynthesizer)
    assert isinstance(
        build_synthesizer(GenerationConfig(backend="anthropic"), None), ExtractiveSynthesizer
    )


# ------------------------------------------------------- entity scoping


def test_planner_records_the_entity_a_subquery_is_about() -> None:
    planner = QueryPlanner(PlanningConfig())

    temporal = planner.plan("Compare Northwind Energy revenue growth in 2024 against 2023")
    # A temporal comparison is about one entity across periods.
    assert {sq.entity for sq in temporal.subqueries} == {"Northwind Energy"}

    entity = planner.plan("Compare Meridian Semiconductor against Atlas Payments on supplier risk")
    assert {sq.entity for sq in entity.subqueries} == {"Meridian Semiconductor", "Atlas Payments"}


def test_primary_entity_ignores_sentence_initial_capitals() -> None:
    from rag_platform.agents.planner import primary_entity

    assert primary_entity("Which risk does Northwind Energy disclose?") == "Northwind Energy"
    assert primary_entity("what was the revenue") is None


def test_entity_filter_restricts_results_to_the_named_company(index) -> None:
    scoped = index.search("revenue growth", top_k=8, entity="Northwind Energy")
    assert scoped
    assert all("Northwind" in (c.chunk.metadata.company_name or "") for c in scoped)


def test_unmatched_entity_degrades_to_an_unscoped_search(index) -> None:
    """An unrecognised name must not turn a good query into an empty result."""
    assert index.search("revenue growth", top_k=5, entity="Nonexistent Holdings Plc")


def test_chunk_matches_entity_uses_document_metadata_not_body_text(index) -> None:
    from rag_platform.retrieval.index import chunk_matches_entity

    # Every chunk of the Northwind filing matches, including ones whose body
    # text never repeats the company name.
    northwind = [c for c in index.chunks if (c.metadata.company_name or "").startswith("Northwind")]
    assert northwind
    assert all(chunk_matches_entity(chunk, "Northwind Energy") for chunk in northwind)
    assert not any(chunk_matches_entity(chunk, "Atlas Payments") for chunk in northwind)


def test_comparison_evidence_stays_within_the_named_entities(index, config) -> None:
    """The bug this guards: quoting one company's figures in answer to another's question."""
    from rag_platform.agents.orchestrator import Orchestrator

    answer = Orchestrator(index, config).answer(
        "Compare Northwind Energy revenue growth in 2024 against 2023", top_k=6
    )
    companies = {
        scored.chunk.metadata.company_name
        for scored in answer.supporting_chunks
        if scored.chunk.metadata.company_name
    }
    assert companies == {"Northwind Energy Corporation"}


def test_tools_forward_the_entity_scope(index) -> None:
    result = ToolRegistry(index).call("financial_lookup", "revenue growth", 6, "Cascade Logistics")
    assert result.chunks
    assert all("Cascade" in (c.chunk.metadata.company_name or "") for c in result.chunks)


def test_sql_lookup_resolves_the_entity_to_its_identifier(index) -> None:
    seen: list[str | None] = []

    def lookup(cik: str | None, year: int | None) -> list[dict[str, object]]:
        seen.append(cik)
        return [{"cik": cik, "fiscal_year": year, "concept": "Revenues", "value": 1.0}]

    registry = ToolRegistry(index, fact_lookup=lookup)
    result = registry.call("sql_lookup", "revenue in 2024", 5, "Northwind Energy")
    assert result.structured
    # The CIK comes from the matched document's metadata, not from parsing the query.
    assert seen and seen[0] == "0001000101"
