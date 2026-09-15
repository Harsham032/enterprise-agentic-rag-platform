"""Retrieval, citation and answer-quality evaluation."""

from .groundedness import answer_completeness, citation_correctness, groundedness_score
from .qrels import QrelSet, RelevanceJudgement, load_qrels
from .retrieval_metrics import (
    average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "QrelSet",
    "RelevanceJudgement",
    "answer_completeness",
    "average_precision",
    "citation_correctness",
    "groundedness_score",
    "load_qrels",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
