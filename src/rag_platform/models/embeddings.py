"""Embedding backends.

Three interchangeable implementations sit behind one protocol:

``tfidf_svd``
    TF-IDF followed by truncated SVD, i.e. latent semantic indexing. It is fit
    on the corpus being indexed, needs no model download, no accelerator and no
    API credentials, and is fully deterministic given a seed. This is the
    default and the backend the published benchmark uses, so anyone cloning the
    repository reproduces the same numbers offline.

``sentence_transformers``
    A pretrained bi-encoder. Stronger on paraphrase and vocabulary mismatch than
    latent semantic indexing, at the cost of a model download and materially
    higher encode latency.

``openai``
    A hosted embedding endpoint, for deployments that prefer not to run model
    inference themselves.

The contract every backend honours: ``encode`` returns L2-normalised float32
rows, so cosine similarity is a plain dot product downstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..config import EmbeddingConfig
from ..errors import EmbeddingError
from ..logging_utils import get_logger

logger = get_logger(__name__)


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows untouched."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32, copy=False)


@runtime_checkable
class Embedder(Protocol):
    """Interface implemented by every embedding backend."""

    dim: int

    def fit(self, texts: Sequence[str]) -> None:
        """Prepare the backend for a corpus. A no-op for pretrained models."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` array of unit-norm float32 vectors."""

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`encode` can be called."""


class TfidfSvdEmbedder:
    """Latent semantic indexing over a TF-IDF term-document matrix."""

    def __init__(self, config: EmbeddingConfig, *, seed: int = 20260101) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.config = config
        self.dim = config.dim
        self._seed = seed
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            min_df=config.min_df,
            max_df=config.max_df,
            ngram_range=(1, config.ngram_max),
            strip_accents="unicode",
            dtype=np.float32,
        )
        self._svd: TruncatedSVD | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, texts: Sequence[str]) -> None:
        """Fit the vectoriser and the SVD projection on ``texts``."""
        from sklearn.decomposition import TruncatedSVD

        if not texts:
            raise EmbeddingError("cannot fit an embedder on an empty corpus")

        term_doc = self._vectorizer.fit_transform(texts)
        n_features = term_doc.shape[1]
        # TruncatedSVD requires n_components < n_features. Small corpora and
        # narrow vocabularies legitimately hit this bound, so the effective
        # dimensionality is reduced and reported rather than raising.
        components = min(self.config.dim, max(1, min(n_features - 1, len(texts) - 1)))
        if components < self.config.dim:
            logger.info(
                "embedding_dim_reduced",
                requested=self.config.dim,
                effective=components,
                vocabulary=n_features,
                documents=len(texts),
            )
        self._svd = TruncatedSVD(
            n_components=components, random_state=self._seed, algorithm="randomized"
        )
        self._svd.fit(term_doc)
        self.dim = components
        self._fitted = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted or self._svd is None:
            raise EmbeddingError("TfidfSvdEmbedder.fit must be called before encode")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        projected = self._svd.transform(self._vectorizer.transform(texts))
        return l2_normalise(np.asarray(projected, dtype=np.float32))

    @property
    def explained_variance(self) -> float:
        """Fraction of TF-IDF variance retained by the projection."""
        if self._svd is None:
            return 0.0
        return float(self._svd.explained_variance_ratio_.sum())

    def save(self, path: Path) -> None:
        """Persist the fitted vectoriser and projection."""
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self._vectorizer, "svd": self._svd, "dim": self.dim}, path)

    def load(self, path: Path) -> None:
        """Restore a previously saved backend."""
        import joblib

        state: dict[str, Any] = joblib.load(path)
        self._vectorizer = state["vectorizer"]
        self._svd = state["svd"]
        self.dim = state["dim"]
        self._fitted = True


class SentenceTransformerEmbedder:
    """Pretrained bi-encoder backend.

    Requires the ``transformers`` extra and a one-off model download; it is not
    exercised by the offline test suite or the published benchmark.
    """

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence_transformers is not installed; install the 'transformers' extra"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def is_fitted(self) -> bool:
        return True

    def fit(self, texts: Sequence[str]) -> None:
        """No-op: the encoder is already trained."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts), batch_size=self._batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return l2_normalise(np.asarray(vectors, dtype=np.float32))


class OpenAIEmbedder:
    """Hosted embedding backend."""

    def __init__(
        self, api_key: str, *, model: str = "text-embedding-3-small", dim: int = 1536
    ) -> None:
        if not api_key:
            raise EmbeddingError("an API key is required for the openai embedding backend")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError("the openai package is not installed") from exc
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.dim = dim

    @property
    def is_fitted(self) -> bool:
        return True

    def fit(self, texts: Sequence[str]) -> None:
        """No-op: the encoder is already trained."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        return l2_normalise(vectors)


def build_embedder(
    config: EmbeddingConfig,
    *,
    seed: int = 20260101,
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    openai_api_key: str = "",
) -> Embedder:
    """Instantiate the backend named by ``config.backend``."""
    if config.backend == "tfidf_svd":
        return TfidfSvdEmbedder(config, seed=seed)
    if config.backend == "sentence_transformers":
        return SentenceTransformerEmbedder(sentence_transformer_model)
    if config.backend == "openai":
        return OpenAIEmbedder(openai_api_key, dim=config.dim)
    raise EmbeddingError(f"unknown embedding backend: {config.backend}")
