"""Reranking stage.

First-stage retrieval optimises for recall over a large candidate pool; the
reranker re-scores a short list with features the fusion step cannot see. The
default is a logistic-regression ranker over cheap query-document features,
trained on labelled query/chunk pairs. It is a real learned model rather than a
hand-tuned score, it trains in milliseconds, and - unlike a cross-encoder - it
adds no model download or GPU requirement to the serving path.

A cross-encoder implementation is provided for deployments that can afford it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import RerankingConfig
from ..data.models import ScoredChunk
from ..logging_utils import get_logger
from ..utils.text import containment, content_tokens, jaccard_overlap

logger = get_logger(__name__)

FEATURE_NAMES = (
    "lexical_score",
    "dense_score",
    "fused_score",
    "query_containment",
    "token_jaccard",
    "bigram_containment",
    "heading_match",
    "length_ratio",
    "first_match_position",
    "exact_phrase",
)


def _bigrams(tokens: Sequence[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]


def extract_features(query: str, candidate: ScoredChunk) -> np.ndarray:
    """Compute the feature vector for one query/candidate pair.

    Features are intentionally cheap and backend-agnostic so the reranker keeps
    working when the embedding backend is swapped.
    """
    query_tokens = content_tokens(query)
    chunk_tokens = content_tokens(candidate.chunk.text)
    heading_tokens = content_tokens(candidate.chunk.section_heading or "")

    lowered_chunk = candidate.chunk.text.lower()
    lowered_query = query.lower().strip("? ")

    first_position = 1.0
    if query_tokens and chunk_tokens:
        positions = [i for i, token in enumerate(chunk_tokens) if token in set(query_tokens)]
        if positions:
            first_position = positions[0] / max(len(chunk_tokens), 1)

    return np.array(
        [
            candidate.lexical_score,
            candidate.dense_score,
            candidate.score,
            containment(query_tokens, chunk_tokens),
            jaccard_overlap(query_tokens, chunk_tokens),
            containment(_bigrams(query_tokens), _bigrams(chunk_tokens)),
            containment(query_tokens, heading_tokens) if heading_tokens else 0.0,
            min(len(chunk_tokens) / 200.0, 2.0),
            first_position,
            1.0 if lowered_query and lowered_query in lowered_chunk else 0.0,
        ],
        dtype=np.float64,
    )


@runtime_checkable
class Reranker(Protocol):
    """Interface implemented by every reranking strategy."""

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class IdentityReranker:
    """Pass-through reranker used when reranking is disabled."""

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return candidates[:top_k]


class LogisticReranker:
    """Logistic-regression ranker over :data:`FEATURE_NAMES`.

    Until :meth:`fit` is called it falls back to the fused first-stage score, so
    an unfitted ranker never degrades results.
    """

    def __init__(self, *, seed: int = 20260101, C: float = 1.0) -> None:
        self._seed = seed
        self._C = C
        self._model = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, examples: list[tuple[str, ScoredChunk, int]]) -> dict[str, float]:
        """Train on ``(query, candidate, relevant)`` triples.

        Returns the learned coefficients keyed by feature name, which the
        experiment report records so the model stays inspectable.
        """
        from sklearn.linear_model import LogisticRegression

        if not examples:
            logger.warning("reranker_no_training_data")
            return {}
        labels = np.array([label for _, _, label in examples], dtype=np.int64)
        if len(np.unique(labels)) < 2:
            logger.warning("reranker_single_class_training_data")
            return {}

        features = np.vstack([extract_features(query, cand) for query, cand, _ in examples])
        model = LogisticRegression(
            C=self._C,
            max_iter=1000,
            random_state=self._seed,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(features, labels)
        self._model = model
        self._fitted = True
        coefficients = dict(zip(FEATURE_NAMES, model.coef_[0].tolist(), strict=True))
        logger.info("reranker_fitted", examples=len(examples), positives=int(labels.sum()))
        return coefficients

    def save(self, path: Path) -> None:
        """Persist the fitted model so the service can rerank without labels.

        Without this the serving process runs an unfitted ranker and silently
        falls back to the first-stage score, which is the difference the
        benchmark attributes to reranking.
        """
        import joblib

        if not self._fitted:
            raise ValueError("cannot save an unfitted reranker")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "features": FEATURE_NAMES}, path)

    def load(self, path: Path) -> None:
        """Restore a previously saved reranker."""
        import joblib

        state = joblib.load(path)
        if tuple(state.get("features", ())) != FEATURE_NAMES:
            raise ValueError(
                "the saved reranker was trained on a different feature set; retrain it"
            )
        self._model = state["model"]
        self._fitted = True

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        if not self._fitted or self._model is None:
            return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]

        features = np.vstack([extract_features(query, candidate) for candidate in candidates])
        scores = self._model.predict_proba(features)[:, 1]
        ordered = sorted(
            (
                candidate.model_copy(update={"rerank_score": float(score)})
                for candidate, score in zip(candidates, scores, strict=True)
            ),
            key=lambda c: c.rerank_score or 0.0,
            reverse=True,
        )
        return [
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(ordered[:top_k], 1)
        ]


class CrossEncoderReranker:
    """Pretrained cross-encoder reranker.

    Requires the ``transformers`` extra and a model download; not exercised by
    the offline test suite or the published benchmark.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self._model.predict(pairs)
        ordered = sorted(
            (
                candidate.model_copy(update={"rerank_score": float(score)})
                for candidate, score in zip(candidates, scores, strict=True)
            ),
            key=lambda c: c.rerank_score or 0.0,
            reverse=True,
        )
        return [
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(ordered[:top_k], 1)
        ]


def build_reranker(config: RerankingConfig, *, seed: int = 20260101) -> Reranker:
    """Instantiate the reranker named by ``config``."""
    if not config.enabled or config.model == "none":
        return IdentityReranker()
    return LogisticReranker(seed=seed)
