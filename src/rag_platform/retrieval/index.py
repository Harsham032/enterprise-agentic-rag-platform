"""The retrieval index: the object the API and the evaluation harness share.

It owns the chunk collection, the hybrid retriever and the reranker, and it is
the only place that knows how those pieces fit together. Building it from a
document list is a single call, which keeps the ingestion script, the service
startup path and the benchmark using the exact same code.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import PipelineConfig
from ..data.models import Chunk, Document, ScoredChunk
from ..errors import IndexNotBuiltError
from ..features.chunking import chunk_documents
from ..features.metadata import enrich_metadata
from ..logging_utils import get_logger
from ..models.embeddings import Embedder, build_embedder
from ..models.reranker import LogisticReranker, Reranker, build_reranker
from ..utils.text import content_tokens
from ..utils.timing import Stopwatch
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .lexical import BM25Retriever

logger = get_logger(__name__)


def chunk_matches_entity(chunk: Chunk, entity: str) -> bool:
    """Whether ``chunk`` belongs to a document about ``entity``.

    Matched on the document's company name, title and ticker rather than on the
    chunk body: a filing routinely names a competitor in its own risk factors,
    so a body-text match would not scope anything.
    """
    entity_tokens = set(content_tokens(entity))
    if not entity_tokens:
        return False
    meta = chunk.metadata
    haystack = " ".join(
        part for part in (meta.company_name, meta.title, meta.ticker, meta.journal) if part
    )
    return entity_tokens.issubset(set(content_tokens(haystack)))


class RetrievalIndex:
    """Chunk store plus the retrieval and reranking stack over it."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder or build_embedder(config.embedding, seed=config.run.seed)
        self.reranker = reranker or build_reranker(config.reranking, seed=config.run.seed)
        self._hybrid = HybridRetriever(
            BM25Retriever(stemming=config.retrieval.stemming),
            DenseRetriever(self.embedder),
            config.retrieval,
        )
        self._documents: dict[str, Document] = {}
        self._chunks_by_id: dict[str, Chunk] = {}
        self.build_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------ build

    def build_from_documents(self, documents: Sequence[Document]) -> None:
        """Enrich, chunk and index ``documents``."""
        if not documents:
            raise IndexNotBuiltError("cannot build an index over zero documents")

        with Stopwatch() as sw:
            enriched = [enrich_metadata(document) for document in documents]
            chunks = chunk_documents(enriched, self.config.chunking)
            if not chunks:
                raise IndexNotBuiltError("chunking produced no chunks")
            self._hybrid.build(chunks)
            self._documents = {document.doc_id: document for document in enriched}
            self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

        self.build_stats = {
            "documents": len(enriched),
            "chunks": len(chunks),
            "mean_chunk_tokens": sum(chunk.token_count for chunk in chunks) / len(chunks),
            "embedding_dim": self.embedder.dim,
            "build_ms": round(sw.elapsed_ms, 2),
        }
        logger.info("index_built", **self.build_stats)

    def add_documents(self, documents: Sequence[Document]) -> int:
        """Add documents to an existing index and rebuild it.

        Rebuilding rather than appending is deliberate: the default embedding
        backend is fit on the corpus, so appending without a refit would place
        new chunks in a stale vector space. Incremental append is the right
        design for a pretrained backend and is where the pgvector and Qdrant
        stores fit.
        """
        enriched = [enrich_metadata(document) for document in documents]
        merged = {**self._documents, **{d.doc_id: d for d in enriched}}
        self.build_from_documents(list(merged.values()))
        return len(enriched)

    # ----------------------------------------------------------------- access

    @property
    def is_built(self) -> bool:
        return self._hybrid.is_built

    @property
    def chunks(self) -> list[Chunk]:
        return self._hybrid.chunks

    @property
    def documents(self) -> list[Document]:
        return list(self._documents.values())

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks_by_id.get(chunk_id)

    def get_document(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    # ---------------------------------------------------------------- queries

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source_types: set[str] | None = None,
        entity: str | None = None,
        rerank: bool | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve, fuse and optionally rerank.

        ``entity`` scopes the results to documents about that entity. Two
        sub-queries of a comparison share nearly all their content words, so
        term and vector scores alone do not reliably keep each one's evidence in
        the right document; a metadata filter does. The filter is dropped when
        it matches nothing, so an unrecognised name degrades to an unscoped
        search rather than to an empty result.
        """
        if not self.is_built:
            raise IndexNotBuiltError("the index has not been built")

        use_rerank = self.config.reranking.enabled if rerank is None else rerank
        final_k = top_k or self.config.retrieval.top_k
        pool_k = max(self.config.reranking.candidate_k, final_k) if use_rerank else final_k

        # An entity filter narrows the pool, so retrieve deeper before applying
        # it: otherwise the filter removes most of a fixed-size candidate list.
        search_k = pool_k * 4 if entity else pool_k
        candidates = self._hybrid.search(
            query,
            top_k=search_k,
            candidate_k=self.config.retrieval.candidate_k,
            source_types=source_types,
        )
        if entity:
            scoped = [c for c in candidates if chunk_matches_entity(c.chunk, entity)]
            if scoped:
                candidates = scoped
        candidates = [
            c.model_copy(update={"rank": i}) for i, c in enumerate(candidates[:pool_k], 1)
        ]

        if not use_rerank:
            return candidates[:final_k]
        return self.reranker.rerank(query, candidates, final_k)

    def save_reranker(self, path: Path) -> None:
        """Persist the fitted reranker."""
        if isinstance(self.reranker, LogisticReranker):
            self.reranker.save(path)

    def load_reranker(self, path: Path) -> bool:
        """Load a saved reranker if one exists. Returns whether it was loaded."""
        if not path.is_file() or not isinstance(self.reranker, LogisticReranker):
            return False
        try:
            self.reranker.load(path)
        except (ValueError, OSError) as exc:
            logger.warning("reranker_load_failed", path=str(path), error=str(exc))
            return False
        logger.info("reranker_loaded", path=str(path))
        return True

    def fit_reranker(self, training_queries: dict[str, set[str]]) -> dict[str, float]:
        """Train the reranker from ``query -> relevant chunk ids`` judgements.

        Negatives are mined from the first-stage results, which is what the
        ranker sees at inference time. Sampling random chunks instead would make
        the task trivially easy and the learned weights useless.
        """
        if not isinstance(self.reranker, LogisticReranker):
            return {}

        examples: list[tuple[str, ScoredChunk, int]] = []
        for query, relevant_ids in training_queries.items():
            candidates = self._hybrid.search(
                query,
                top_k=self.config.reranking.candidate_k,
                candidate_k=self.config.retrieval.candidate_k,
            )
            for candidate in candidates:
                label = 1 if candidate.chunk_id in relevant_ids else 0
                examples.append((query, candidate, label))
        return self.reranker.fit(examples)

    # -------------------------------------------------------------- reporting

    def describe(self) -> dict[str, Any]:
        """Index statistics surfaced by the health endpoint."""
        source_counts: dict[str, int] = {}
        for chunk in self.chunks:
            source_counts[chunk.source_type.value] = (
                source_counts.get(chunk.source_type.value, 0) + 1
            )
        return {
            "built": self.is_built,
            "documents": len(self._documents),
            "chunks": len(self._chunks_by_id),
            "sources": source_counts,
            "embedding_backend": self.config.embedding.backend,
            "embedding_dim": self.embedder.dim,
            "fusion": self.config.retrieval.fusion,
            "reranking": self.config.reranking.enabled,
            **{k: v for k, v in self.build_stats.items() if k not in {"documents", "chunks"}},
        }


def load_corpus(source_dir: Path) -> list[Document]:
    """Read every supported document under ``source_dir``.

    Imported lazily by callers that need it so the retrieval package does not
    depend on the filesystem loader.
    """
    from ..data.local_documents import load_directory

    return load_directory(source_dir)
