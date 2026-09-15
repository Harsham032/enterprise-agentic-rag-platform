"""Noise removal applied after format parsing and before chunking.

Retrieval quality degrades when boilerplate is indexed: page furniture repeats
across every chunk of a filing, which inflates lexical document frequencies and
pulls unrelated chunks toward every query.
"""

from __future__ import annotations

import re

from ..utils.text import normalise_whitespace

_PAGE_ARTIFACT_RE = re.compile(
    r"^\s*(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s*\|\s*page|-\s*\d+\s*-)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TABLE_OF_CONTENTS_RE = re.compile(r"^\s*table of contents\s*$", re.IGNORECASE | re.MULTILINE)
_DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d*")
_REPEATED_PUNCT_RE = re.compile(r"([^\w\s])\1{3,}")
_COPYRIGHT_RE = re.compile(r"^\s*(?:©|\(c\))\s*\d{4}.*$", re.IGNORECASE | re.MULTILINE)


def clean_document_text(text: str) -> str:
    """Remove page furniture, dot leaders and repeated punctuation runs."""
    text = _PAGE_ARTIFACT_RE.sub("", text)
    text = _TABLE_OF_CONTENTS_RE.sub("", text)
    text = _COPYRIGHT_RE.sub("", text)
    text = _DOT_LEADER_RE.sub(" ", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    return normalise_whitespace(text)


def drop_boilerplate_lines(text: str, *, min_repeats: int = 3, max_words: int = 12) -> str:
    """Remove short lines that repeat throughout a document.

    Running headers and footers survive format parsing as identical short lines.
    Lines longer than ``max_words`` are never removed, so genuine repeated prose
    such as a recurring risk-factor lead-in is preserved.
    """
    lines = text.split("\n")
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped.split()) <= max_words:
            counts[stripped] = counts.get(stripped, 0) + 1

    noisy = {line for line, count in counts.items() if count >= min_repeats}
    if not noisy:
        return text
    kept = [line for line in lines if line.strip() not in noisy]
    return normalise_whitespace("\n".join(kept))
