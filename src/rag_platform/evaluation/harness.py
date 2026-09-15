"""Evaluation harness.

Runs a configuration end to end and reports retrieval quality, answer quality
and latency over a labelled query set.

Two methodological points the numbers depend on:

*Held-out evaluation.* The reranker is trained on labelled queries. Every
configuration - reranked or not - is therefore scored on the same held-out
query split, so the comparison is not contaminated by the reranker having seen
its evaluation queries. The split is by query id, deterministic given the seed.

*Uncertainty.* The labelled set is small. Point estimates on a few dozen queries
move considerably under resampling, so every headline metric is reported with a
bootstrap confidence interval. A configuration whose interval overlaps another's
has not been shown to beat it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from ..agents.orchestrator import Orchestrator
from ..config import PipelineConfig
from ..data.models import Document
from ..logging_utils import get_logger
from ..retrieval.index import RetrievalIndex
from ..utils.timing import Stopwatch, latency_summary
from .groundedness import evaluate_answer
from .qrels import QrelSet, RelevanceJudgement, build_span_index
from .retrieval_metrics import (
    aggregate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    summarise_run,
    truncate_to_budget,
)

logger = get_logger(__name__)

BOOTSTRAP_RESAMPLES = 2000


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 20260101,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``."""
    if len(values) < 2:
        single = float(values[0]) if values else 0.0
        return single, single
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    lower = means[int(tail * resamples)]
    upper = means[min(int((1.0 - tail) * resamples), resamples - 1)]
    return lower, upper


def evaluate_retrieval(
    index: RetrievalIndex,
    qrels: QrelSet,
    *,
    query_ids: Sequence[str] | None = None,
    k_values: Sequence[int] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Score retrieval over the judged queries and report per-query detail."""
    k_values = list(k_values or index.config.evaluation.k_values)
    # Retrieve deep enough that the token-budget truncation below is never
    # limited by the size of the candidate list.
    retrieval_k = top_k or max(max(k_values), index.config.evaluation.max_budget_k)
    resolved = qrels.resolve(index.chunks, build_span_index(index.documents))
    selected = set(query_ids) if query_ids else {j.query_id for j in qrels.queries}

    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []

    for judgement in qrels.queries:
        if judgement.query_id not in selected:
            continue
        relevant = resolved[judgement.query_id]
        with Stopwatch() as sw:
            results = index.search(judgement.query, top_k=retrieval_k)
        latencies.append(sw.elapsed_ms)
        retrieved_ids = [scored.chunk_id for scored in results]
        metrics = summarise_run(retrieved_ids, relevant, k_values)

        # Budgeted metrics: the same ranking truncated at a fixed number of
        # evidence tokens. These are the numbers to compare across chunk sizes.
        budget = index.config.evaluation.token_budget
        token_counts = [scored.chunk.token_count for scored in results]
        budgeted = truncate_to_budget(retrieved_ids, token_counts, budget)
        budget_k = len(budgeted)
        metrics.update(
            {
                "recall@budget": recall_at_k(budgeted, relevant, budget_k),
                "precision@budget": precision_at_k(budgeted, relevant, budget_k),
                "ndcg@budget": ndcg_at_k(budgeted, relevant, budget_k),
                "budget_chunks": float(budget_k),
                "budget_tokens_used": float(sum(token_counts[:budget_k])),
            }
        )

        per_query.append(
            {
                "query_id": judgement.query_id,
                "query": judgement.query,
                "type": judgement.type,
                "relevant_chunks": len(relevant),
                "retrieved": len(retrieved_ids),
                **metrics,
            }
        )

    if not per_query:
        return {"queries": 0, "metrics": {}, "per_query": []}

    metric_keys = [key for key in per_query[0] if key not in {"query_id", "query", "type"}]
    aggregated = aggregate([{key: row[key] for key in metric_keys} for row in per_query])

    intervals = {}
    for key in (
        f"recall@{index.config.evaluation.primary_k}",
        f"ndcg@{index.config.evaluation.primary_k}",
        "recall@budget",
        "ndcg@budget",
        "reciprocal_rank",
    ):
        if key in aggregated:
            lower, upper = bootstrap_ci([row[key] for row in per_query])
            intervals[key] = {"lower": lower, "upper": upper}

    by_type: dict[str, dict[str, float]] = {}
    for query_type in sorted({row["type"] for row in per_query}):
        rows = [row for row in per_query if row["type"] == query_type]
        by_type[query_type] = {
            "queries": float(len(rows)),
            **aggregate([{key: row[key] for key in metric_keys} for row in rows]),
        }

    return {
        "queries": len(per_query),
        "metrics": aggregated,
        "confidence_intervals": intervals,
        "by_query_type": by_type,
        "latency": latency_summary(latencies),
        "per_query": per_query,
    }


def evaluate_answers(
    orchestrator: Orchestrator,
    qrels: QrelSet,
    *,
    query_ids: Sequence[str] | None = None,
    warmup: int = 2,
) -> dict[str, Any]:
    """Score generated answers for groundedness, citations and completeness."""
    selected = set(query_ids) if query_ids else {j.query_id for j in qrels.queries}
    judgements: list[RelevanceJudgement] = [
        judgement for judgement in qrels.queries if judgement.query_id in selected
    ]
    if not judgements:
        return {"queries": 0, "metrics": {}, "per_query": []}

    # Warm the caches the first request would otherwise pay for, so the reported
    # latency describes steady state rather than first-call overhead.
    for judgement in judgements[:warmup]:
        orchestrator.answer(judgement.query)

    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    for judgement in judgements:
        answer = orchestrator.answer(judgement.query)
        latencies.append(answer.latency_ms)
        metrics = evaluate_answer(
            answer,
            judgement.answer_terms,
            threshold=orchestrator.config.generation.min_support_overlap,
        )
        per_query.append(
            {
                "query_id": judgement.query_id,
                "query": judgement.query,
                "type": judgement.type,
                "strategy": answer.diagnostics.get("strategy", "single_hop"),
                "subqueries": len(answer.subqueries),
                "latency_ms": answer.latency_ms,
                **metrics,
            }
        )

    metric_keys = [
        key
        for key in per_query[0]
        if key not in {"query_id", "query", "type", "strategy", "subqueries"}
    ]
    aggregated = aggregate([{key: row[key] for key in metric_keys} for row in per_query])

    intervals = {}
    for key in ("groundedness", "answer_completeness", "citation_supported"):
        lower, upper = bootstrap_ci([row[key] for row in per_query])
        intervals[key] = {"lower": lower, "upper": upper}

    return {
        "queries": len(per_query),
        "metrics": aggregated,
        "confidence_intervals": intervals,
        "latency": latency_summary(latencies),
        "per_query": per_query,
    }


def build_index(config: PipelineConfig, documents: Sequence[Document]) -> RetrievalIndex:
    """Build an index for ``config`` over ``documents``."""
    index = RetrievalIndex(config)
    index.build_from_documents(documents)
    return index


def prepare_index(
    config: PipelineConfig,
    documents: Sequence[Document],
    qrels: QrelSet,
    train_ids: Sequence[str],
) -> tuple[RetrievalIndex, dict[str, float]]:
    """Build an index and fit its reranker on the training queries only."""
    index = build_index(config, documents)
    if not config.reranking.enabled:
        return index, {}
    spans = build_span_index(index.documents)
    training = {
        judgement.query: judgement.relevant_chunk_ids(index.chunks, spans)
        for judgement in qrels.queries
        if judgement.query_id in set(train_ids)
    }
    coefficients = index.fit_reranker(training)
    return index, coefficients


def run_configuration(
    name: str,
    config: PipelineConfig,
    documents: Sequence[Document],
    qrels: QrelSet,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    *,
    include_answers: bool = True,
) -> dict[str, Any]:
    """Evaluate one named configuration on the held-out split."""
    logger.info("evaluating_configuration", name=name)
    index, coefficients = prepare_index(config, documents, qrels, train_ids)
    retrieval = evaluate_retrieval(index, qrels, query_ids=test_ids)

    result: dict[str, Any] = {
        "name": name,
        "index": index.describe(),
        "retrieval": retrieval,
        "reranker_coefficients": coefficients,
        "config": {
            "chunking": config.chunking.model_dump(mode="json"),
            "embedding": config.embedding.model_dump(mode="json"),
            "retrieval": config.retrieval.model_dump(mode="json"),
            "reranking": config.reranking.model_dump(mode="json"),
            "generation": config.generation.model_dump(mode="json"),
        },
    }
    if include_answers:
        orchestrator = Orchestrator(index, config)
        result["answers"] = evaluate_answers(
            orchestrator, qrels, query_ids=test_ids, warmup=config.evaluation.latency_warmup
        )
    return result
