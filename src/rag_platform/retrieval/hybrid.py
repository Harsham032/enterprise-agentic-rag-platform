"""Score fusion for hybrid retrieval.

Two fusion strategies are implemented because they fail differently:

``weighted``
    Min-max normalise each retriever's scores over the candidate pool, then take
    a weighted sum. Keeps score magnitude, so a chunk that is a strong match on
    both signals outranks one that is merely present in both lists. Sensitive to
    outliers in the BM25 distribution.

``rrf``
    Reciprocal rank fusion: ``sum(1 / (k + rank))``. Ignores magnitude entirely,
    which makes it robust to badly scaled or incomparable scorers, at the cost of
    discarding confidence information.

The default is ``weighted``; ``docs/results.md`` reports both measured on the
same corpus.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import RetrievalConfig
from ..data.models import Chunk, ScoredChunk
from ..errors import IndexNotBuiltError
from ..logging_utils import get_logger
from .dense import DenseRetriever
from .lexical import BM25Retriever

logger = get_logger(__name__)


def min_max_normalise(scores: np.ndarray) -> np.ndarray:
    """Scale scores into ``[0, 1]``; a constant vector maps to all zeros."""
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def weighted_fusion(
    lexical: dict[int, float],
    dense: dict[int, float],
    *,
    lexical_weight: float,
    dense_weight: float,
) -> dict[int, float]:
    """Weighted sum of separately normalised score maps."""
    indices = sorted(set(lexical) | set(dense))
    if not indices:
        return {}
    lexical_values = min_max_normalise(
        np.array([lexical.get(i, 0.0) for i in indices], dtype=np.float64)
    )
    dense_values = min_max_normalise(
        np.array([dense.get(i, 0.0) for i in indices], dtype=np.float64)
    )
    total = lexical_weight + dense_weight
    fused = (lexical_weight * lexical_values + dense_weight * dense_values) / (total or 1.0)
    return dict(zip(indices, fused.tolist(), strict=True))


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    *,
    k: int = 60,
) -> dict[int, float]:
    """Combine ranked id lists with reciprocal rank fusion."""
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, index in enumerate(ranked, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    """Runs both retrievers over a shared chunk list and fuses their scores."""

    def __init__(
        self,
        lexical: BM25Retriever,
        dense: DenseRetriever,
        config: RetrievalConfig,
    ) -> None:
        self.lexical = lexical
        self.dense = dense
        self.config = config
        self._chunks: list[Chunk] = []

    @property
    def is_built(self) -> bool:
        return bool(self._chunks) and self.lexical.is_built and self.dense.is_built

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def build(self, chunks: Sequence[Chunk], *, fit: bool = True) -> None:
        """Build both underlying indexes over the same chunk ordering."""
        self._chunks = list(chunks)
        self.lexical.build(self._chunks)
        self.dense.build(self._chunks, fit=fit)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        source_types: set[str] | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve and fuse, optionally restricted to given source types."""
        if not self.is_built:
            raise IndexNotBuiltError("HybridRetriever.build must be called before searching")

        top_k = top_k or self.config.top_k
        candidate_k = candidate_k or self.config.candidate_k
        pool = max(candidate_k, top_k)

        lexical_hits = dict(self.lexical.search(query, pool))
        dense_hits = dict(self.dense.search(query, pool))

        if self.config.fusion == "rrf":
            lexical_ranked = [
                index for index, _ in sorted(lexical_hits.items(), key=lambda kv: -kv[1])
            ]
            dense_ranked = [index for index, _ in sorted(dense_hits.items(), key=lambda kv: -kv[1])]
            fused = reciprocal_rank_fusion([lexical_ranked, dense_ranked], k=self.config.rrf_k)
        else:
            fused = weighted_fusion(
                lexical_hits,
                dense_hits,
                lexical_weight=self.config.lexical_weight,
                dense_weight=self.config.dense_weight,
            )

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        results: list[ScoredChunk] = []
        for index, score in ordered:
            chunk = self._chunks[index]
            if source_types and chunk.source_type.value not in source_types:
                continue
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(score),
                    lexical_score=float(lexical_hits.get(index, 0.0)),
                    dense_score=float(dense_hits.get(index, 0.0)),
                    rank=len(results) + 1,
                )
            )
            if len(results) >= top_k:
                break
        return results
