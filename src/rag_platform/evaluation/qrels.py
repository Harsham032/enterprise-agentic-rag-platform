"""Relevance judgements and their resolution onto chunk ids.

Judgements address documents by source filename and by section heading. Chunk
ids are content-derived, so a change to chunk size would invalidate any
judgement written against them; addressing sections instead keeps one set of
labels valid across every chunking configuration compared in the benchmark.

Resolution is by character span, not by section heading equality. A judgement's
sections are converted into character ranges of the source document, and a chunk
counts as relevant when enough of it falls inside one of those ranges. This is
what makes the comparison between chunking strategies fair: a strategy that does
not carry section headings - a fixed sliding window, for instance - is scored on
the same labels as one that does, rather than scoring zero because its chunks
have no heading to match.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from ..data.models import Chunk, Document
from ..errors import EvaluationError

# A chunk counts as relevant when either test passes:
#
#   MIN_CHUNK_COVERAGE  - this share of the chunk lies inside the judged region,
#                         i.e. the chunk is mostly about the right thing;
#   MIN_TARGET_COVERAGE - the chunk contains this share of the judged region,
#                         i.e. the chunk carries the answer even though it also
#                         carries a lot of surrounding material.
#
# Only the first test would penalise a coarse chunker for including context, and
# only the second would reward a chunker that returns whole documents. Requiring
# either keeps the comparison between chunking strategies meaningful in both
# directions.
MIN_CHUNK_COVERAGE = 0.4
MIN_TARGET_COVERAGE = 0.6

# (source filename, section heading) -> character range in the document text
SpanIndex = dict[tuple[str, str], tuple[int, int]]


class RelevanceTarget(BaseModel):
    """One relevant region of the corpus."""

    document: str
    sections: list[str] = Field(default_factory=list)

    def matches(self, chunk: Chunk, spans: SpanIndex | None = None) -> bool:
        """Whether ``chunk`` falls inside this target."""
        if getattr(chunk.metadata, "source_file", None) != self.document:
            return False
        if not self.sections:
            return True
        if spans is not None:
            ranges = [
                span
                for section in self.sections
                if (span := spans.get((self.document, section))) is not None
            ]
            if ranges:
                # The relevant region is the union of the named sections. Testing
                # each range separately would reject a chunk that straddles two
                # adjacent relevant sections without reaching the threshold on
                # either - exactly the case a short abstract produces.
                covered = _covered_chars(chunk, ranges)
                chunk_length = max(chunk.char_end - chunk.char_start, 1)
                target_length = max(sum(end - start for start, end in _merge_ranges(ranges)), 1)
                return (
                    covered / chunk_length >= MIN_CHUNK_COVERAGE
                    or covered / target_length >= MIN_TARGET_COVERAGE
                )
        # Without a span index the heading carried on the chunk is the only
        # signal available.
        return chunk.section_heading in self.sections


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching character ranges."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _covered_chars(chunk: Chunk, ranges: Sequence[tuple[int, int]]) -> int:
    """Characters of ``chunk`` lying inside the union of ``ranges``."""
    return sum(
        max(min(chunk.char_end, end) - max(chunk.char_start, start), 0)
        for start, end in _merge_ranges(ranges)
    )


def build_span_index(documents: Sequence[Document]) -> SpanIndex:
    """Map ``(source filename, section heading)`` to a character range.

    Section bodies are located in the parent document text with a forward-only
    cursor, matching how chunk offsets are assigned, so the two coordinate
    systems agree.
    """
    spans: SpanIndex = {}
    for document in documents:
        source_file = getattr(document.metadata, "source_file", None)
        if not source_file:
            continue
        cursor = 0
        for section in document.sections:
            probe = section.text[:64]
            start = document.text.find(probe, cursor)
            if start < 0:
                start = document.text.find(probe)
            if start < 0:
                continue
            end = start + len(section.text)
            spans[(source_file, section.heading)] = (start, end)
            cursor = end
    return spans


class RelevanceJudgement(BaseModel):
    """A query together with the regions that answer it."""

    query_id: str
    query: str
    type: str = "single_hop"
    relevant: list[RelevanceTarget]
    answer_terms: list[str] = Field(default_factory=list)

    def relevant_chunk_ids(
        self, chunks: Sequence[Chunk], spans: SpanIndex | None = None
    ) -> set[str]:
        """Resolve this judgement against a concrete chunk collection."""
        return {
            chunk.chunk_id
            for chunk in chunks
            if any(target.matches(chunk, spans) for target in self.relevant)
        }

    def relevant_documents(self) -> set[str]:
        return {target.document for target in self.relevant}


class QrelSet(BaseModel):
    """A loaded judgement file."""

    version: str = "1.0"
    corpus: str = ""
    description: str = ""
    queries: list[RelevanceJudgement]

    def __len__(self) -> int:
        return len(self.queries)

    def by_type(self, query_type: str) -> list[RelevanceJudgement]:
        return [judgement for judgement in self.queries if judgement.type == query_type]

    def resolve(
        self, chunks: Sequence[Chunk], spans: SpanIndex | None = None
    ) -> dict[str, set[str]]:
        """Return ``query_id -> relevant chunk ids`` and fail loudly on empty sets.

        A judgement that resolves to nothing silently deflates every metric, so
        it is treated as a corrupt label file rather than as a hard query.
        """
        resolved: dict[str, set[str]] = {}
        empty: list[str] = []
        for judgement in self.queries:
            ids = judgement.relevant_chunk_ids(chunks, spans)
            if not ids:
                empty.append(judgement.query_id)
            resolved[judgement.query_id] = ids
        if empty:
            raise EvaluationError(
                "these judgements match no chunk in the index, so the corpus and the "
                f"judgement file are out of sync: {', '.join(empty)}"
            )
        return resolved

    def training_split(
        self, fraction: float = 0.5, seed: int = 20260101
    ) -> tuple[list[str], list[str]]:
        """Deterministic query-level split used to train and evaluate the reranker.

        Splitting by query rather than by candidate is essential: the reranker
        sees several candidates per query, so a candidate-level split would leak
        every training query into the evaluation set.
        """
        import random

        if not 0.0 < fraction < 1.0:
            raise EvaluationError("fraction must be strictly between 0 and 1")
        ids = sorted(judgement.query_id for judgement in self.queries)
        rng = random.Random(seed)
        rng.shuffle(ids)
        cut = max(1, int(len(ids) * fraction))
        return sorted(ids[:cut]), sorted(ids[cut:])


def load_qrels(path: str | Path) -> QrelSet:
    """Load and validate a judgement file."""
    qrels_path = Path(path)
    if not qrels_path.is_file():
        raise EvaluationError(f"judgement file not found: {qrels_path}")
    try:
        payload = json.loads(qrels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"could not parse {qrels_path}: {exc}") from exc
    try:
        qrels = QrelSet.model_validate(payload)
    except Exception as exc:
        raise EvaluationError(f"invalid judgement file {qrels_path}: {exc}") from exc
    if not qrels.queries:
        raise EvaluationError(f"{qrels_path} contains no queries")
    return qrels
