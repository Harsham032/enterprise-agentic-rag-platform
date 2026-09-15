"""Format-specific text extraction.

Every supported input is reduced to plain text plus, where the format carries
it, a list of headed sections. Section structure matters downstream: a citation
that names ``Item 1A. Risk Factors`` is far more useful than one naming a page
number, and section-aware chunking avoids splitting a risk factor in half.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from bs4 import BeautifulSoup

from ..data.models import Section
from ..errors import ParsingError
from ..logging_utils import get_logger
from ..utils.text import normalise_whitespace

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".htm", ".html", ".pdf"})

_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*$", re.MULTILINE)
_BLOCK_TAGS = ("p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br")


def supported_suffixes() -> frozenset[str]:
    """File extensions the ingestion pipeline can read."""
    return SUPPORTED_SUFFIXES


def parse_html(html: str) -> str:
    """Strip markup, scripts and styling from an HTML document.

    EDGAR filings are HTML with heavy inline styling and table layout; block
    elements are converted to newlines so that table rows and paragraphs do not
    run together into one unsplittable line.
    """
    soup = BeautifulSoup(html, "lxml")
    for node in soup(["script", "style", "noscript", "head"]):
        node.decompose()
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_after("\n")
    text = soup.get_text(separator=" ")
    return normalise_whitespace(text)


def parse_pdf(data: bytes) -> str:
    """Extract text from a PDF byte stream.

    Only the embedded text layer is read. Scanned documents without one yield an
    empty string rather than silently producing garbage; callers should route
    those through OCR before ingestion.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - pypdf is a hard dependency
        raise ParsingError("pypdf is required to read PDF documents") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise ParsingError(f"could not read PDF: {exc}") from exc
    return normalise_whitespace("\n\n".join(pages))


def parse_markdown(text: str) -> tuple[str, list[Section]]:
    """Return cleaned markdown text and its heading-delimited sections."""
    matches = list(_MARKDOWN_HEADING_RE.finditer(text))
    cleaned = normalise_whitespace(text)
    if not matches:
        return cleaned, []

    sections: list[Section] = []
    for order, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[order + 1].start() if order + 1 < len(matches) else len(text)
        body = normalise_whitespace(text[start:end])
        if body:
            sections.append(Section(heading=heading, text=body, order=order))
    return cleaned, sections


def extract_text(path: Path) -> tuple[str, list[Section]]:
    """Read ``path`` and return ``(text, sections)``.

    Raises :class:`ParsingError` for unsupported extensions so that a malformed
    corpus directory fails loudly instead of silently skipping documents.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ParsingError(f"unsupported document type {suffix!r}: {path}")

    if suffix == ".pdf":
        return parse_pdf(path.read_bytes()), []
    if suffix in {".htm", ".html"}:
        return parse_html(path.read_text(encoding="utf-8", errors="replace")), []

    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".md", ".markdown"}:
        return parse_markdown(raw)
    return normalise_whitespace(raw), []
