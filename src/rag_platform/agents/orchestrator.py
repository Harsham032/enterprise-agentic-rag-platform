"""Orchestration: plan, execute tools, merge evidence, synthesise, verify.

This is the request path. It owns the latency budget and the diagnostics that
make a bad answer debuggable: which sub-queries ran, which tools served them,
how much evidence each contributed, and where the time went.
"""

from __future__ import annotations

from typing import Any

from ..config import PipelineConfig
from ..data.models import Answer, ScoredChunk
from ..generation.synthesizer import Synthesizer, build_synthesizer
from ..logging_utils import get_logger
from ..models.llm import LLMClient
from ..retrieval.index import RetrievalIndex
from ..utils.timing import Stopwatch
from .planner import QueryPlan, QueryPlanner
from .tools import ToolRegistry, ToolResult

logger = get_logger(__name__)


def merge_results(results: list[ToolResult], top_k: int) -> list[ScoredChunk]:
    """Merge per-sub-query evidence into one ranked list.

    A chunk retrieved for several sub-queries keeps its best score and is
    emitted once. Deduplication matters here: a comparison query asks about two
    periods that frequently share a chunk, and returning it twice would both
    waste the answer budget and double-count it in the evidence.
    """
    best: dict[str, ScoredChunk] = {}
    for result in results:
        for chunk in result.chunks:
            existing = best.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                best[chunk.chunk_id] = chunk
    ordered = sorted(best.values(), key=lambda c: c.score, reverse=True)[:top_k]
    return [chunk.model_copy(update={"rank": rank}) for rank, chunk in enumerate(ordered, start=1)]


class Orchestrator:
    """Runs the full query path and returns a cited answer."""

    def __init__(
        self,
        index: RetrievalIndex,
        config: PipelineConfig,
        *,
        llm_client: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self.index = index
        self.config = config
        self.planner = QueryPlanner(config.planning, llm_client)
        self.tools = tools or ToolRegistry(index)
        self.synthesizer = synthesizer or build_synthesizer(config.generation, llm_client)

    def answer(self, query: str, *, top_k: int | None = None) -> Answer:
        """Plan, retrieve, synthesise and verify a response to ``query``."""
        final_k = top_k or self.config.retrieval.top_k

        with Stopwatch() as total:
            with Stopwatch() as planning:
                plan: QueryPlan = self.planner.plan(query)

            # Each sub-query gets its own share of the evidence budget, with a
            # floor so a three-way decomposition still returns usable context.
            per_subquery_k = max(final_k // max(len(plan.subqueries), 1), 3)

            with Stopwatch() as retrieval:
                results = [
                    self.tools.call(subquery.tool, subquery.text, per_subquery_k, subquery.entity)
                    for subquery in plan.subqueries
                ]
                evidence = merge_results(results, final_k)

            with Stopwatch() as generation:
                answer = self.synthesizer.synthesise(query, evidence)

        diagnostics: dict[str, Any] = dict(answer.diagnostics)
        diagnostics.update(
            {
                "strategy": plan.strategy,
                "subquery_count": len(plan.subqueries),
                "evidence_chunks": len(evidence),
                "structured_rows": sum(len(result.structured) for result in results),
                "tool_errors": [result.error for result in results if result.error],
                "planning_ms": round(planning.elapsed_ms, 2),
                "retrieval_ms": round(retrieval.elapsed_ms, 2),
                "generation_ms": round(generation.elapsed_ms, 2),
            }
        )

        enriched = answer.model_copy(
            update={
                "subqueries": [subquery.text for subquery in plan.subqueries],
                "tools_used": sorted({result.tool for result in results}),
                "latency_ms": round(total.elapsed_ms, 2),
                "diagnostics": diagnostics,
            }
        )
        logger.info(
            "query_answered",
            strategy=plan.strategy,
            subqueries=len(plan.subqueries),
            evidence=len(evidence),
            groundedness=round(enriched.groundedness, 3),
            latency_ms=enriched.latency_ms,
        )
        return enriched
