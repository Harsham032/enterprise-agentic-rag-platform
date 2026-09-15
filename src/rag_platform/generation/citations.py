"""Citation construction and verification.

A citation is only worth anything if it can be checked, so every citation this
platform emits carries the exact quote it was derived from and is verified
against the cited chunk before the answer is returned. Verification is the step
that turns "the answer mentions a source" into "the answer's claim is present in
that source".
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..data.models import Citation, ScoredChunk
from ..utils.text import containment, content_tokens, truncate

CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


def build_citations(
    sentences: Sequence[tuple[str, ScoredChunk]],
    *,
    quote_chars: int = 240,
) -> tuple[list[Citation], dict[str, int]]:
    """Assign stable markers to the chunks backing an answer.

    Returns the citations and a ``chunk_id -> marker`` map. Chunks are numbered
    in order of first use so markers read in the order the reader meets them.
    """
    markers: dict[str, int] = {}
    citations: list[Citation] = []
    for sentence, scored in sentences:
        chunk_id = scored.chunk_id
        if chunk_id not in markers:
            markers[chunk_id] = len(markers) + 1
            citations.append(
                Citation(
                    marker=markers[chunk_id],
                    chunk_id=chunk_id,
                    doc_id=scored.chunk.doc_id,
                    label=scored.chunk.citation_label(),
                    quote=truncate(sentence, quote_chars),
                )
            )
    return citations, markers


def verify_citations(
    citations: Sequence[Citation],
    retrieved: Sequence[ScoredChunk],
    *,
    min_overlap: float = 0.35,
) -> list[Citation]:
    """Mark each citation verified when its quote is present in the cited chunk.

    Verification is lexical containment of the quote's content words, which
    tolerates the whitespace and punctuation differences introduced by chunking
    while still catching a citation pointing at the wrong passage.
    """
    by_id = {scored.chunk_id: scored.chunk.text for scored in retrieved}
    verified: list[Citation] = []
    for citation in citations:
        source_text = by_id.get(citation.chunk_id)
        if source_text is None:
            verified.append(citation.model_copy(update={"verified": False, "support_score": 0.0}))
            continue
        score = containment(
            content_tokens(citation.quote),
            content_tokens(source_text),
        )
        verified.append(
            citation.model_copy(update={"support_score": score, "verified": score >= min_overlap})
        )
    return verified


def extract_markers(text: str) -> list[int]:
    """Return the citation markers referenced by ``text`` in order of appearance."""
    return [int(match.group(1)) for match in CITATION_MARKER_RE.finditer(text)]


def strip_unresolvable_markers(text: str, valid_markers: set[int]) -> str:
    """Remove bracketed markers that do not correspond to a real citation.

    Model-generated answers occasionally invent marker numbers. Leaving them in
    would present an unverifiable reference as though it were a source.
    """

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in valid_markers else ""

    return re.sub(
        r"\s*\[(\d+)\]", lambda m: (" " + replace(m)).rstrip() if replace(m) else "", text
    )
