"""Domain model shared by ingestion, retrieval, generation and evaluation.

Identifiers are content-derived (:func:`rag_platform.utils.text.stable_hash`) so
re-ingesting the same document produces the same chunk ids. That property is
what lets relevance judgements in ``data/fixtures/qrels.json`` stay valid across
index rebuilds.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..utils.text import stable_hash


class SourceType(StrEnum):
    """Origin of a document, used for routing and for source-scoped retrieval."""

    SEC_FILING = "sec_filing"
    BIOMEDICAL_ABSTRACT = "biomedical_abstract"
    USER_DOCUMENT = "user_document"


class DocumentMetadata(BaseModel):
    """Source-specific attributes extracted during ingestion.

    The union of SEC and biomedical fields is kept in one model rather than a
    polymorphic hierarchy: the fields are sparse, queries filter across sources,
    and a flat record maps directly onto one relational table.
    """

    model_config = ConfigDict(extra="allow")

    title: str = ""
    # SEC EDGAR
    company_name: str | None = None
    ticker: str | None = None
    cik: str | None = None
    form_type: str | None = None
    fiscal_year: int | None = None
    filing_date: date | None = None
    # PubMed
    pmid: str | None = None
    journal: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    # Generic
    url: str | None = None
    language: str = "en"
    # Filename the document was loaded from. Relevance judgements address
    # documents by this rather than by the content-derived doc_id, so a label
    # survives an edit to the document text.
    source_file: str | None = None

    @field_validator("pmid", "cik", "ticker", mode="before")
    @classmethod
    def _coerce_identifier(cls, value: object) -> object:
        """Accept numeric identifiers from JSON payloads and YAML front matter."""
        return str(value) if isinstance(value, int) else value

    @field_validator("cik")
    @classmethod
    def _zero_pad_cik(cls, value: str | None) -> str | None:
        """EDGAR uses 10-digit zero-padded CIKs in its JSON APIs."""
        return value.strip().zfill(10) if value else value


class Section(BaseModel):
    """A titled span of a document, e.g. ``Item 1A. Risk Factors``."""

    heading: str
    text: str
    order: int = 0


class Document(BaseModel):
    """A parsed source document prior to chunking."""

    doc_id: str
    source_type: SourceType
    text: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    sections: list[Section] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        source_type: SourceType,
        text: str,
        metadata: DocumentMetadata | None = None,
        sections: list[Section] | None = None,
        natural_key: str | None = None,
    ) -> Document:
        """Build a document with a deterministic id.

        ``natural_key`` should identify the document at its source (accession
        number, PMID, file path). When absent the document text is hashed, which
        still deduplicates identical uploads.
        """
        key = natural_key or text
        return cls(
            doc_id=stable_hash(source_type.value, key),
            source_type=source_type,
            text=text,
            metadata=metadata or DocumentMetadata(),
            sections=sections or [],
        )


class Chunk(BaseModel):
    """A retrievable unit of text with a pointer back to its document."""

    chunk_id: str
    doc_id: str
    source_type: SourceType
    text: str
    position: int
    section_heading: str | None = None
    char_start: int = 0
    char_end: int = 0
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    @property
    def token_count(self) -> int:
        return len(self.text.split())

    def citation_label(self) -> str:
        """Human-readable provenance string rendered next to generated answers."""
        meta = self.metadata
        if self.source_type is SourceType.SEC_FILING:
            parts = [p for p in (meta.company_name, meta.form_type) if p]
            if meta.fiscal_year:
                parts.append(f"FY{meta.fiscal_year}")
            if self.section_heading:
                parts.append(self.section_heading)
            return " - ".join(parts) or meta.title or self.doc_id
        if self.source_type is SourceType.BIOMEDICAL_ABSTRACT:
            label = meta.title or self.doc_id
            if meta.journal and meta.publication_year:
                return f"{label} ({meta.journal}, {meta.publication_year})"
            return label
        return meta.title or self.doc_id


class ScoredChunk(BaseModel):
    """A chunk with the scores that produced its ranking."""

    chunk: Chunk
    score: float
    lexical_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float | None = None
    rank: int = 0

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class Citation(BaseModel):
    """A claim in a generated answer linked to the evidence supporting it."""

    marker: int
    chunk_id: str
    doc_id: str
    label: str
    quote: str
    support_score: float = 0.0
    verified: bool = False


class Answer(BaseModel):
    """A synthesised response together with its evidence and diagnostics."""

    query: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    supporting_chunks: list[ScoredChunk] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    groundedness: float = 0.0
    latency_ms: float = 0.0
    diagnostics: dict[str, Any] = Field(default_factory=dict)
