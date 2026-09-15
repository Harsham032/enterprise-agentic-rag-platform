"""Answer synthesis.

The default synthesiser is extractive: it selects sentences verbatim from the
retrieved evidence and attaches a citation to each one. Nothing is paraphrased,
so every claim is attributable by construction and groundedness is a property of
the output rather than a hope about it.

Sentence selection uses maximal marginal relevance. Relevance alone returns near
duplicates - the same figure restated in an adjacent chunk - which wastes the
answer budget without adding information. MMR trades relevance against novelty
with a single ``lambda`` parameter.

An LLM-backed synthesiser is also provided for deployments that want fluent
prose. Its output passes through the same citation verification, and any marker
it invents is stripped before the answer is returned.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from ..config import GenerationConfig
from ..data.models import Answer, ScoredChunk
from ..logging_utils import get_logger
from ..models.llm import SYNTHESIS_SYSTEM_PROMPT, LLMClient
from ..utils.text import content_tokens, jaccard_overlap, split_sentences
from .citations import (
    build_citations,
    extract_markers,
    strip_unresolvable_markers,
    verify_citations,
)

logger = get_logger(__name__)

MMR_LAMBDA = 0.72
MIN_SENTENCE_TOKENS = 4
# A sentence sharing no weighted query term with the query is filler regardless
# of how well the chunk it came from scored, so it is never selectable while a
# sentence that does share one remains.
MIN_QUERY_OVERLAP = 1e-9
# Stop once the best remaining candidate's query coverage falls this far below
# the first selected span. Without it the answer pads out to max_spans
# with progressively less relevant material.
RELEVANCE_CUTOFF_RATIO = 0.25
INSUFFICIENT_EVIDENCE = "The indexed corpus does not contain evidence that answers this question."


class Synthesizer(Protocol):
    """Interface implemented by every answer synthesiser."""

    def synthesise(self, query: str, retrieved: Sequence[ScoredChunk]) -> Answer: ...


def _candidate_spans(
    retrieved: Sequence[ScoredChunk],
    span_sentences: int,
) -> list[tuple[str, ScoredChunk]]:
    """Flatten retrieved chunks into (span, source) pairs.

    A span is up to ``span_sentences`` consecutive sentences taken from a single
    chunk, sliding one sentence at a time. Spans never cross a chunk boundary,
    so the citation attached to a span always covers all of its text.
    """
    candidates: list[tuple[str, ScoredChunk]] = []
    seen: set[str] = set()
    for scored in retrieved:
        sentences = [
            sentence
            for sentence in split_sentences(scored.chunk.text)
            if len(content_tokens(sentence)) >= MIN_SENTENCE_TOKENS
        ]
        for start in range(len(sentences)):
            span = " ".join(sentences[start : start + span_sentences])
            key = " ".join(span.lower().split())
            if key in seen:
                continue
            seen.add(key)
            candidates.append((span, scored))
    return candidates


def _term_weights(query_tokens: set[str], chunk_texts: Sequence[str]) -> dict[str, float]:
    """Inverse-document-frequency weights for the query's content words.

    Computed over the retrieved candidate pool rather than the whole corpus.
    Within a result set every chunk tends to contain the entity the query names,
    so an unweighted overlap treats "cascade" as being as informative as
    "margin". Weighting by rarity inside the pool is exactly the discrimination
    the selection step needs, and it needs no corpus-wide statistics at request
    time.
    """
    if not chunk_texts:
        return dict.fromkeys(query_tokens, 1.0)
    token_sets = [set(content_tokens(text)) for text in chunk_texts]
    total = len(token_sets)
    weights: dict[str, float] = {}
    for token in query_tokens:
        document_frequency = sum(1 for tokens in token_sets if token in tokens)
        # Unseen terms keep the maximum weight: a query term absent from every
        # candidate is the most discriminative signal available if one appears.
        weights[token] = math.log(1.0 + total / (1.0 + document_frequency))
    return weights


def _weighted_coverage(sentence_tokens: Sequence[str], weights: dict[str, float]) -> float:
    """Share of the query's total term weight present in a sentence."""
    total = sum(weights.values())
    if total <= 0.0:
        return 0.0
    present = set(sentence_tokens)
    return sum(weight for token, weight in weights.items() if token in present) / total


def _relevance(coverage: float, scored: ScoredChunk, rank_span: int) -> float:
    """Blend weighted query coverage with the rank of the chunk it came from."""
    rank_prior = 1.0 - ((scored.rank - 1) / rank_span if rank_span else 0.0)
    return 0.82 * coverage + 0.18 * max(rank_prior, 0.0)


class ExtractiveSynthesizer:
    """Selects supporting sentences verbatim and cites each one."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def synthesise(self, query: str, retrieved: Sequence[ScoredChunk]) -> Answer:
        if not retrieved:
            return Answer(query=query, text=INSUFFICIENT_EVIDENCE, groundedness=0.0)

        candidates = _candidate_spans(retrieved, self.config.span_sentences)
        if not candidates:
            return Answer(query=query, text=INSUFFICIENT_EVIDENCE, groundedness=0.0)

        selected = self._select(query, candidates)
        if not selected:
            return Answer(query=query, text=INSUFFICIENT_EVIDENCE, groundedness=0.0)

        citations, markers = build_citations(selected)
        citations = verify_citations(
            citations, retrieved, min_overlap=self.config.min_support_overlap
        )

        text = " ".join(f"{span} [{markers[scored.chunk_id]}]" for span, scored in selected)

        from ..evaluation.groundedness import groundedness_score

        # Measured with the same function the evaluation harness uses, so the
        # number reported at request time and the number in the benchmark mean
        # the same thing. For this backend it is 1.0 by construction - every
        # sentence is copied out of a retrieved chunk - which is the design
        # guarantee rather than a result.
        grounded = groundedness_score(
            text,
            [scored.chunk.text for scored in retrieved],
            threshold=self.config.min_support_overlap,
        )
        verified = sum(1 for citation in citations if citation.verified)
        return Answer(
            query=query,
            text=text,
            citations=citations,
            supporting_chunks=list(retrieved),
            groundedness=grounded,
            diagnostics={
                "mode": "extractive",
                "candidate_spans": len(candidates),
                "selected_spans": len(selected),
                "distinct_sources": len({scored.chunk.doc_id for _, scored in selected}),
                "citations_verified": verified,
            },
        )

    def _select(
        self,
        query: str,
        candidates: list[tuple[str, ScoredChunk]],
    ) -> list[tuple[str, ScoredChunk]]:
        """Maximal marginal relevance selection over candidate spans.

        Overlapping spans from the same chunk are highly redundant by
        construction, so the MMR redundancy term does most of the work of
        keeping the answer from repeating itself.
        """
        query_tokens = set(content_tokens(query))
        rank_span = max((scored.rank for _, scored in candidates), default=1)
        weights = _term_weights(query_tokens, [scored.chunk.text for _, scored in candidates])

        scored_candidates = []
        for span, scored in candidates:
            coverage = _weighted_coverage(content_tokens(span), weights)
            scored_candidates.append(
                (span, scored, _relevance(coverage, scored, rank_span), coverage)
            )

        # Prefer sentences that carry some of the query's discriminative terms;
        # fall back to the whole pool only when none does.
        on_topic = [item for item in scored_candidates if item[3] > MIN_QUERY_OVERLAP]
        remaining = sorted(on_topic or scored_candidates, key=lambda item: item[2], reverse=True)
        if not remaining:
            return []

        cutoff = remaining[0][3] * RELEVANCE_CUTOFF_RATIO

        selected: list[tuple[str, ScoredChunk]] = []
        selected_tokens: list[list[str]] = []

        while remaining and len(selected) < self.config.max_spans:
            best_index = 0
            best_value = float("-inf")
            for index, (span, _, relevance, _coverage) in enumerate(remaining):
                tokens = content_tokens(span)
                redundancy = max(
                    (jaccard_overlap(tokens, chosen) for chosen in selected_tokens), default=0.0
                )
                value = MMR_LAMBDA * relevance - (1.0 - MMR_LAMBDA) * redundancy
                if value > best_value:
                    best_value, best_index = value, index

            span, scored, _relevance_value, coverage = remaining.pop(best_index)
            if selected and coverage < cutoff:
                break
            selected.append((span, scored))
            selected_tokens.append(content_tokens(span))
        return selected


class LLMSynthesizer:
    """Abstractive synthesis over the retrieved evidence."""

    def __init__(self, config: GenerationConfig, client: LLMClient) -> None:
        self.config = config
        self.client = client

    def synthesise(self, query: str, retrieved: Sequence[ScoredChunk]) -> Answer:
        if not retrieved:
            return Answer(query=query, text=INSUFFICIENT_EVIDENCE, groundedness=0.0)

        markers = {scored.chunk_id: index for index, scored in enumerate(retrieved, start=1)}
        evidence_block = "\n\n".join(
            f"[{markers[scored.chunk_id]}] ({scored.chunk.citation_label()})\n{scored.chunk.text}"
            for scored in retrieved
        )
        prompt = (
            f"Question: {query}\n\nEvidence passages:\n{evidence_block}\n\n"
            "Answer the question using only these passages, citing each claim with its "
            "bracketed passage number."
        )
        raw = self.client.complete(prompt, system=SYNTHESIS_SYSTEM_PROMPT)

        cited_markers = set(extract_markers(raw))
        text = strip_unresolvable_markers(raw, set(markers.values()))

        by_marker = {
            marker: scored for scored, marker in zip(retrieved, markers.values(), strict=True)
        }
        used = [
            (text, by_marker[marker])
            for marker in sorted(cited_markers & set(by_marker))
            if marker in by_marker
        ]
        citations, _ = build_citations(used)
        citations = verify_citations(
            citations, retrieved, min_overlap=self.config.min_support_overlap
        )

        from ..evaluation.groundedness import groundedness_score

        grounded = groundedness_score(
            text,
            [scored.chunk.text for scored in retrieved],
            threshold=self.config.min_support_overlap,
        )
        return Answer(
            query=query,
            text=text.strip(),
            citations=citations,
            supporting_chunks=list(retrieved),
            groundedness=grounded,
            diagnostics={"mode": "abstractive", "markers_cited": len(cited_markers)},
        )


def build_synthesizer(config: GenerationConfig, client: LLMClient | None = None) -> Synthesizer:
    """Return the synthesiser matching ``config.backend``."""
    if config.backend == "extractive" or client is None:
        return ExtractiveSynthesizer(config)
    return LLMSynthesizer(config, client)
