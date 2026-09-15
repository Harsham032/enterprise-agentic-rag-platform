"""Dense retrieval over an in-process embedding matrix.

Because every embedding backend emits L2-normalised rows, cosine similarity is a
single matrix-vector product. For corpora up to a few hundred thousand chunks
that is faster than an approximate index and has no recall loss; the pgvector
and Qdrant backends in :mod:`rag_platform.data.store` cover larger deployments.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data.models import Chunk
from ..errors import IndexNotBuiltError
from ..logging_utils import get_logger
from ..models.embeddings import Embedder

logger = get_logger(__name__)


class DenseRetriever:
    """Exact nearest-neighbour search over chunk embeddings."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._matrix: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    @property
    def is_built(self) -> bool:
        return self._matrix is not None

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            raise IndexNotBuiltError("DenseRetriever.build must be called first")
        return self._matrix

    def build(self, chunks: Sequence[Chunk], *, fit: bool = True) -> None:
        """Embed and index ``chunks``.

        ``fit`` controls whether corpus-fitted backends (latent semantic
        indexing) are refit. Set it to ``False`` when adding documents to an
        index whose vector space must stay comparable with what is already
        stored.
        """
        if not chunks:
            raise IndexNotBuiltError("cannot build a dense index over zero chunks")
        self._chunks = list(chunks)
        texts = [
            f"{chunk.section_heading}. {chunk.text}" if chunk.section_heading else chunk.text
            for chunk in self._chunks
        ]
        if fit and not self.embedder.is_fitted or fit and hasattr(self.embedder, "_vectorizer"):
            self.embedder.fit(texts)
        self._matrix = self.embedder.encode(texts)
        logger.info("dense_index_built", chunks=len(self._chunks), dim=self.embedder.dim)

    def scores(self, query: str) -> np.ndarray:
        """Return cosine similarities aligned with the indexed chunk order."""
        if self._matrix is None:
            raise IndexNotBuiltError("DenseRetriever.build must be called before scoring")
        query_vector = self.embedder.encode([query])
        if query_vector.shape[0] == 0:
            return np.zeros(len(self._chunks), dtype=np.float32)
        return (self._matrix @ query_vector[0]).astype(np.float32)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return ``(chunk_index, cosine)`` for the ``top_k`` nearest chunks."""
        scores = self.scores(query)
        if scores.size == 0:
            return []
        limit = min(top_k, scores.size)
        top = np.argpartition(-scores, limit - 1)[:limit]
        ordered = top[np.argsort(-scores[top])]
        return [(int(index), float(scores[index])) for index in ordered]
