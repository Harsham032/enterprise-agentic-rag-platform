"""Chunking strategies.

Chunk size is the single most consequential retrieval knob in this pipeline.
Long chunks dilute the query signal and make citations imprecise; short chunks
lose the context a claim depends on. Two strategies are provided:

``fixed_window``
    A sliding window over whitespace tokens with a configurable overlap. Simple,
    strategy-agnostic, and the right default for unstructured uploads.

``section_aware``
    Windows are confined to a document's sections and cut on sentence
    boundaries. Every chunk carries its section heading, which the citation
    layer surfaces and the reranker uses as a feature.

Token counts are whitespace-word counts, not model subword tokens. The
distinction is documented rather than hidden: it keeps chunking independent of
any tokenizer, at the cost of chunks being roughly 25-35 percent longer in
subword terms for English prose.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Protocol

from ..config import ChunkingConfig
from ..data.models import Chunk, Document, Section
from ..utils.text import split_sentences, stable_hash

_TOKEN_SPAN_RE = re.compile(r"\S+")


class Chunker(Protocol):
    """Strategy interface implemented by every chunking approach."""

    def chunk(self, document: Document) -> list[Chunk]:  # pragma: no cover - protocol
        ...


def _token_spans(text: str) -> list[tuple[int, int]]:
    """Character span of every whitespace-delimited token in ``text``."""
    return [(match.start(), match.end()) for match in _TOKEN_SPAN_RE.finditer(text)]


def _window_bounds(total: int, size: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` token offsets for a sliding window."""
    if total <= size:
        yield 0, total
        return
    step = max(size - overlap, 1)
    start = 0
    while start < total:
        end = min(start + size, total)
        yield start, end
        if end == total:
            return
        start += step


class _OffsetTracker:
    """Locates each chunk's character span inside the parent document text.

    Chunk spans are what makes relevance judgements independent of the chunking
    strategy: a judgement names a region of a document, and any chunk overlapping
    that region counts, whatever produced it. A forward-only cursor keeps the
    search linear and stops a repeated sentence from resolving to an earlier
    position than the chunk it belongs to.
    """

    PROBE_CHARS = 64

    def __init__(self, document_text: str) -> None:
        self._text = document_text
        self._cursor = 0

    def locate(self, chunk_text: str) -> tuple[int, int]:
        probe = chunk_text[: self.PROBE_CHARS]
        start = self._text.find(probe, self._cursor)
        if start < 0:
            start = self._text.find(probe)
        if start < 0:
            # A chunk that joins text across a paragraph break does not appear
            # verbatim in the document, so fall back to a whitespace-flexible
            # match before giving up.
            start = self._search_flexible(probe)
        if start < 0:
            return 0, len(chunk_text)
        self._cursor = start + max(len(probe) // 2, 1)
        return start, start + len(chunk_text)

    def _search_flexible(self, probe: str) -> int:
        words = probe.split()
        if len(words) < 2:
            return -1
        pattern = re.compile(r"\s+".join(re.escape(word) for word in words[:-1]))
        match = pattern.search(self._text, self._cursor) or pattern.search(self._text)
        return match.start() if match else -1


def _make_chunk(
    document: Document,
    text: str,
    position: int,
    section_heading: str | None,
    char_start: int,
    char_end: int,
) -> Chunk:
    return Chunk(
        chunk_id=stable_hash(document.doc_id, str(position), text[:64]),
        doc_id=document.doc_id,
        source_type=document.source_type,
        text=text,
        position=position,
        section_heading=section_heading,
        char_start=char_start,
        char_end=char_end,
        metadata=document.metadata,
    )


class FixedWindowChunker:
    """Sliding window over whitespace tokens."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = document.text.split()
        if len(tokens) < self.config.min_tokens:
            return (
                [_make_chunk(document, document.text, 0, None, 0, len(document.text))]
                if document.text.strip()
                else []
            )

        # Token spans give exact offsets without a string search: a window that
        # joins text across a paragraph break never appears verbatim in the
        # document, so searching for it would misplace the chunk.
        spans = _token_spans(document.text)
        chunks: list[Chunk] = []
        for position, (start, end) in enumerate(
            _window_bounds(len(tokens), self.config.target_tokens, self.config.overlap_tokens)
        ):
            window = tokens[start:end]
            if len(window) < self.config.min_tokens and chunks:
                break
            text = " ".join(window)
            char_start = spans[start][0] if start < len(spans) else 0
            char_end = spans[min(end, len(spans)) - 1][1] if spans else len(text)
            chunks.append(_make_chunk(document, text, position, None, char_start, char_end))
        return chunks


class SectionAwareChunker:
    """Windows confined to sections and cut on sentence boundaries."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk(self, document: Document) -> list[Chunk]:
        sections = document.sections or [Section(heading="", text=document.text, order=0)]
        offsets = _OffsetTracker(document.text)
        chunks: list[Chunk] = []
        position = 0
        for section in sections:
            for text in self._split_section(section.text):
                char_start, char_end = offsets.locate(text)
                chunks.append(
                    _make_chunk(
                        document, text, position, section.heading or None, char_start, char_end
                    )
                )
                position += 1
        return chunks

    def _split_section(self, text: str) -> list[str]:
        """Pack sentences into windows of roughly ``target_tokens`` words."""
        sentences = split_sentences(text)
        if not sentences:
            return []

        target = self.config.target_tokens
        overlap = self.config.overlap_tokens
        windows: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            # A single sentence longer than the target becomes its own chunk
            # rather than being cut mid-clause.
            if sentence_tokens >= target and not current:
                windows.append(sentence)
                continue
            if current_tokens + sentence_tokens > target and current:
                windows.append(" ".join(current))
                current, current_tokens = self._carry_overlap(current, overlap)
            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            tail = " ".join(current)
            if len(tail.split()) >= self.config.min_tokens or not windows:
                windows.append(tail)
            else:
                # Too short to stand alone: fold it into the previous window.
                windows[-1] = f"{windows[-1]} {tail}"
        return windows

    @staticmethod
    def _carry_overlap(sentences: list[str], overlap_tokens: int) -> tuple[list[str], int]:
        """Return the trailing sentences to repeat at the start of the next window."""
        if overlap_tokens <= 0:
            return [], 0
        carried: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            sentence_tokens = len(sentence.split())
            if total + sentence_tokens > overlap_tokens and carried:
                break
            carried.insert(0, sentence)
            total += sentence_tokens
        return carried, total


def build_chunker(config: ChunkingConfig) -> Chunker:
    """Instantiate the chunker named by ``config.strategy``."""
    if config.strategy == "fixed_window":
        return FixedWindowChunker(config)
    return SectionAwareChunker(config)


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Chunk a single document with the configured strategy."""
    return build_chunker(config).chunk(document)


def chunk_documents(documents: list[Document], config: ChunkingConfig) -> list[Chunk]:
    """Chunk a corpus, reusing one chunker instance."""
    chunker = build_chunker(config)
    return [chunk for document in documents for chunk in chunker.chunk(document)]
