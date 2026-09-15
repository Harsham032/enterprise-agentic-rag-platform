"""Ranking metrics over binary relevance judgements.

Definitions are spelled out because the same metric names are used
inconsistently across the retrieval literature:

* ``recall_at_k``    - judged-relevant items retrieved in the top *k*, divided
  by the total number of judged-relevant items for the query. Not capped at *k*:
  when a query has more relevant items than *k*, recall cannot reach 1.0, and
  that ceiling is real rather than something to normalise away.
* ``precision_at_k``  - relevant items in the top *k* divided by *k*. Dividing by
  *k* rather than by the number retrieved keeps short result lists honest.
* ``reciprocal_rank`` - one over the rank of the first relevant item.
* ``ndcg_at_k``       - binary-gain NDCG with the standard ``1/log2(rank+1)``
  discount, normalised by the ideal ranking for that query.
* ``average_precision`` - mean of precision measured at each relevant hit,
  divided by the number of relevant items.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _relevant_flags(retrieved: Sequence[str], relevant: set[str], k: int) -> list[bool]:
    return [item in relevant for item in retrieved[:k]]


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of judged-relevant items appearing in the top ``k``."""
    if not relevant:
        return 0.0
    hits = sum(_relevant_flags(retrieved, relevant, k))
    return hits / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top ``k`` positions holding a relevant item."""
    if k <= 0:
        return 0.0
    return sum(_relevant_flags(retrieved, relevant, k)) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """One over the rank of the first relevant item; zero when none is found."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(runs: Sequence[tuple[Sequence[str], set[str]]]) -> float:
    """Mean of :func:`reciprocal_rank` across queries."""
    if not runs:
        return 0.0
    return sum(reciprocal_rank(retrieved, relevant) for retrieved, relevant in runs) / len(runs)


def dcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Discounted cumulative gain with binary gains."""
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(_relevant_flags(retrieved, relevant, k), start=1)
        if is_relevant
    )


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised DCG against the ideal ranking for this query."""
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal


def average_precision(retrieved: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    """Average of the precision measured at each relevant hit."""
    if not relevant:
        return 0.0
    limit = k if k is not None else len(retrieved)
    hits = 0
    total = 0.0
    for rank, item in enumerate(retrieved[:limit], start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), limit)


def truncate_to_budget(
    items: Sequence[str],
    token_counts: Sequence[int],
    budget: int,
) -> list[str]:
    """Take ranked items until their cumulative token count exceeds ``budget``.

    At least one item is always returned, so a single chunk larger than the
    whole budget is still evaluated rather than producing an empty run.
    """
    kept: list[str] = []
    used = 0
    for item, tokens in zip(items, token_counts, strict=True):
        if kept and used + tokens > budget:
            break
        kept.append(item)
        used += tokens
    return kept


def summarise_run(
    retrieved: Sequence[str],
    relevant: set[str],
    k_values: Sequence[int],
) -> dict[str, float]:
    """All ranking metrics for a single query."""
    metrics: dict[str, float] = {"reciprocal_rank": reciprocal_rank(retrieved, relevant)}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
        metrics[f"map@{k}"] = average_precision(retrieved, relevant, k)
    return metrics


def aggregate(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    """Macro-average per-query metric dictionaries."""
    if not per_query:
        return {}
    keys = sorted({key for row in per_query for key in row})
    return {key: sum(row.get(key, 0.0) for row in per_query) / len(per_query) for key in keys}
