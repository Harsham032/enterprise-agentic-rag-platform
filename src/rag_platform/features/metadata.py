"""Metadata enrichment applied after parsing.

Values recoverable from the document body are filled in only when the source
did not already supply them, so upstream metadata always wins over inference.
"""

from __future__ import annotations

import re

from ..data.models import Document, DocumentMetadata, SourceType
from ..utils.text import truncate

_FISCAL_YEAR_RE = re.compile(
    r"fiscal year (?:ended|ending)[^.]{0,40}?(19|20)(\d{2})", re.IGNORECASE
)
_FORM_TYPE_RE = re.compile(r"\bform\s+(10-K|10-Q|8-K|20-F|S-1)\b", re.IGNORECASE)
_CIK_RE = re.compile(r"central index key[^0-9]{0,20}(\d{4,10})", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def infer_fiscal_year(text: str) -> int | None:
    """Recover a fiscal year from filing cover-page language."""
    match = _FISCAL_YEAR_RE.search(text)
    if match:
        return int(f"{match.group(1)}{match.group(2)}")
    return None


def infer_form_type(text: str) -> str | None:
    """Recover the SEC form type from the document body."""
    match = _FORM_TYPE_RE.search(text)
    return match.group(1).upper() if match else None


def derive_title(document: Document) -> str:
    """Produce a display title when the source supplied none."""
    meta = document.metadata
    if meta.title:
        return meta.title
    if document.source_type is SourceType.SEC_FILING and meta.company_name:
        year = meta.fiscal_year or ""
        return f"{meta.company_name} {meta.form_type or 'filing'} {year}".strip()
    first_line = next((line for line in document.text.split("\n") if line.strip()), "")
    return truncate(first_line.strip(), 120)


def enrich_metadata(document: Document) -> Document:
    """Return a copy of ``document`` with inferred metadata filled in."""
    meta: DocumentMetadata = document.metadata.model_copy(deep=True)
    head = document.text[:4000]

    if document.source_type is SourceType.SEC_FILING:
        if meta.fiscal_year is None:
            meta.fiscal_year = infer_fiscal_year(head)
        if meta.form_type is None:
            meta.form_type = infer_form_type(head)
        if meta.cik is None:
            cik_match = _CIK_RE.search(head)
            if cik_match:
                meta.cik = cik_match.group(1).zfill(10)

    if document.source_type is SourceType.BIOMEDICAL_ABSTRACT and meta.publication_year is None:
        year_match = _YEAR_RE.search(head)
        if year_match:
            meta.publication_year = int(year_match.group(0))

    enriched = document.model_copy(update={"metadata": meta})
    if not meta.title:
        meta.title = derive_title(enriched)
    return enriched
