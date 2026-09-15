"""Lexical retrieval with Okapi BM25.

Lexical matching remains the stronger half of the system for the query types
this platform targets. Filing and abstract queries are dense with exact
identifiers - ticker symbols, fiscal years, gene names, us-gaap concept names -
and an exact term match on those carries more signal than embedding proximity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from ..data.models import Chunk
from ..errors import IndexNotBuiltError
from ..logging_utils import get_logger
from ..utils.text import tokenize

logger = get_logger(__name__)


class BM25Retriever:
    """BM25 over chunk text, with the section heading prepended.

    Repeating the heading inside the indexed text lets a query naming a section
    ("risk factors", "methods") match without a separate field-weighted index.
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        drop_stopwords: bool = True,
        stemming: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.drop_stopwords = drop_stopwords
        self.stemming = stemming
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def _tokenise(self, text: str) -> list[str]:
        return tokenize(text, drop_stopwords=self.drop_stopwords, stemming=self.stemming)

    def build(self, chunks: Sequence[Chunk]) -> None:
        """Index ``chunks``. Rebuilding replaces the previous index."""
        if not chunks:
            raise IndexNotBuiltError("cannot build a lexical index over zero chunks")
        self._chunks = list(chunks)
        corpus = [
            self._tokenise(f"{chunk.section_heading or ''} {chunk.text}") for chunk in self._chunks
        ]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        logger.info("lexical_index_built", chunks=len(self._chunks))

    def scores(self, query: str) -> np.ndarray:
        """Return raw BM25 scores aligned with the indexed chunk order."""
        if self._bm25 is None:
            raise IndexNotBuiltError("BM25Retriever.build must be called before scoring")
        tokens = self._tokenise(query)
        if not tokens:
            return np.zeros(len(self._chunks), dtype=np.float32)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return ``(chunk_index, score)`` for the ``top_k`` best matches."""
        scores = self.scores(query)
        if scores.size == 0:
            return []
        limit = min(top_k, scores.size)
        top = np.argpartition(-scores, limit - 1)[:limit]
        ordered = top[np.argsort(-scores[top])]
        return [(int(index), float(scores[index])) for index in ordered if scores[index] > 0.0]
