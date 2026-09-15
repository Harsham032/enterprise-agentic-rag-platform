"""Filesystem corpus loader.

Documents are plain files with optional YAML front matter carrying metadata:

.. code-block:: text

    ---
    source_type: sec_filing
    company_name: Northwind Energy Corporation
    form_type: 10-K
    fiscal_year: 2024
    ---

    ## Item 1A. Risk Factors
    ...

Front matter is how the bundled evaluation corpus and user uploads both attach
provenance without a separate sidecar index. Files without it still load; their
metadata is inferred by :mod:`rag_platform.features.metadata`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..errors import ParsingError
from ..features.parsing import extract_text, supported_suffixes
from ..logging_utils import get_logger
from .models import Document, DocumentMetadata, Section, SourceType

logger = get_logger(__name__)

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

_METADATA_FIELDS = frozenset(DocumentMetadata.model_fields)


def split_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    """Return ``(metadata, body)`` for a document that may carry front matter."""
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    try:
        parsed = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ParsingError(f"malformed front matter: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ParsingError("front matter must be a YAML mapping")
    return parsed, raw[match.end() :]


def _build_metadata(front_matter: dict[str, Any]) -> tuple[SourceType, DocumentMetadata]:
    raw_source = str(front_matter.get("source_type", SourceType.USER_DOCUMENT.value))
    try:
        source_type = SourceType(raw_source)
    except ValueError as exc:
        valid = ", ".join(s.value for s in SourceType)
        raise ParsingError(f"unknown source_type {raw_source!r}; expected one of: {valid}") from exc

    fields = {key: value for key, value in front_matter.items() if key in _METADATA_FIELDS}
    extras = {
        key: value
        for key, value in front_matter.items()
        if key not in _METADATA_FIELDS and key != "source_type"
    }
    return source_type, DocumentMetadata(**fields, **extras)


def load_document(path: Path) -> Document:
    """Load a single file into a :class:`Document`."""
    suffix = path.suffix.lower()
    if suffix not in supported_suffixes():
        raise ParsingError(f"unsupported document type {suffix!r}: {path}")

    front_matter: dict[str, Any] = {}
    sections: list[Section] = []

    if suffix in {".md", ".markdown", ".txt"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        front_matter, body = split_front_matter(raw)
        temporary = path.with_suffix(suffix)  # keep the suffix for the parser's dispatch
        text, sections = _parse_body(temporary, body)
    else:
        text, sections = extract_text(path)

    source_type, metadata = _build_metadata(front_matter)
    if not metadata.title:
        metadata.title = _title_from_path(path)
    # Recorded so relevance judgements can address documents by filename
    # rather than by a content-derived id that changes when text is edited.
    metadata.source_file = path.name

    return Document.create(
        source_type=source_type,
        text=text,
        metadata=metadata,
        sections=sections,
        natural_key=str(path.name),
    )


def _parse_body(path: Path, body: str) -> tuple[str, list[Section]]:
    """Parse an in-memory body using the parser matching ``path``'s suffix."""
    from ..features.parsing import parse_markdown
    from ..utils.text import normalise_whitespace

    if path.suffix.lower() in {".md", ".markdown"}:
        return parse_markdown(body)
    return normalise_whitespace(body), []


def _title_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def load_directory(source_dir: Path, *, recursive: bool = True) -> list[Document]:
    """Load every supported document under ``source_dir``, sorted by path.

    Sorting makes ingestion order deterministic, which matters because the
    corpus-fitted embedding backend and the chunk ordering both depend on it.
    """
    directory = Path(source_dir)
    if not directory.is_dir():
        raise ParsingError(f"corpus directory not found: {directory}")

    pattern = "**/*" if recursive else "*"
    paths = sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in supported_suffixes()
    )
    if not paths:
        raise ParsingError(f"no supported documents found under {directory}")

    documents: list[Document] = []
    for path in paths:
        try:
            documents.append(load_document(path))
        except ParsingError as exc:
            logger.warning("document_skipped", path=str(path), error=str(exc))
    logger.info("corpus_loaded", directory=str(directory), documents=len(documents))
    return documents
