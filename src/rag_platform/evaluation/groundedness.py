"""Answer-quality measures: groundedness, citation correctness, completeness.

These are the metrics that separate a retrieval demo from something worth
trusting. All three are computed lexically, without a judge model. That is a
deliberate trade-off:

* the measurements are deterministic and reproducible by anyone who clones the
  repository, with no credentials and no per-run cost;
* they cannot detect a claim that is faithfully paraphrased but wrong, and they
  cannot reward a correct answer phrased in vocabulary absent from the evidence.

In other words they are a conservative lower bound on faithfulness, not a
substitute for human adjudication. ``docs/methodology.md`` states the limitation
in the same terms.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..data.models import Answer, Citation, ScoredChunk
from ..utils.text import containment, content_tokens, split_sentences, tokenize

# A sentence carrying fewer than this many content words is too short for token
# containment to say anything. Connective sentences are skipped rather than
# counted as unsupported, which would understate groundedness.
MIN_CONTENT_TOKENS = 3


def sentence_support(sentence: str, evidence_texts: Sequence[str]) -> float:
    """Best containment of a sentence's content words in any evidence passage."""
    sentence_tokens = content_tokens(sentence)
    if len(sentence_tokens) < MIN_CONTENT_TOKENS:
        return 1.0
    return max(
        (containment(sentence_tokens, content_tokens(text)) for text in evidence_texts),
        default=0.0,
    )


def groundedness_score(answer_text: str, evidence: Sequence[str], threshold: float = 0.35) -> float:
    """Fraction of answer sentences supported by at least one evidence passage.

    A sentence counts as supported when the share of its content words present
    in some passage reaches ``threshold``.
    """
    sentences = split_sentences(answer_text)
    if not sentences:
        return 0.0
    supported = sum(
        1 for sentence in sentences if sentence_support(sentence, evidence) >= threshold
    )
    return supported / len(sentences)


def citation_correctness(
    citations: Sequence[Citation],
    retrieved: Sequence[ScoredChunk],
    threshold: float = 0.35,
) -> dict[str, float]:
    """Check that every citation points at a retrieved chunk that supports its quote.

    Two failure modes are separated, because they have different causes:

    ``resolvable`` - the cited chunk id exists in the result set. A failure here
    is a plumbing bug: the answer references evidence that was never retrieved.

    ``supported`` - the quoted text is actually present in the cited chunk. A
    failure here means the citation resolves but points at the wrong passage.
    """
    if not citations:
        return {"count": 0.0, "resolvable": 0.0, "supported": 0.0, "precision": 0.0}

    by_id = {scored.chunk_id: scored.chunk.text for scored in retrieved}
    resolvable = 0
    supported = 0
    for citation in citations:
        text = by_id.get(citation.chunk_id)
        if text is None:
            continue
        resolvable += 1
        if sentence_support(citation.quote, [text]) >= threshold:
            supported += 1

    total = len(citations)
    return {
        "count": float(total),
        "resolvable": resolvable / total,
        "supported": supported / total,
        "precision": supported / resolvable if resolvable else 0.0,
    }


def answer_completeness(answer_text: str, expected_terms: Sequence[str]) -> float:
    """Fraction of the expected content elements present in the answer.

    ``expected_terms`` come from the judgement file and describe what a complete
    answer must contain (a figure, a named entity, a direction of change). A
    multi-word term is matched as a substring of the normalised answer; a single
    token is matched against the answer's token set so that ``0.71`` does not
    match ``0.718``.
    """
    if not expected_terms:
        return 0.0
    lowered = " ".join(answer_text.lower().split())
    answer_tokens = set(tokenize(answer_text))
    found = 0
    for term in expected_terms:
        normalised = " ".join(term.lower().split())
        if " " in normalised:
            found += normalised in lowered
        else:
            found += normalised in answer_tokens or normalised in lowered.split()
    return found / len(expected_terms)


def evaluate_answer(
    answer: Answer,
    expected_terms: Sequence[str],
    *,
    threshold: float = 0.35,
) -> dict[str, float]:
    """All answer-quality metrics for a single response."""
    evidence = [scored.chunk.text for scored in answer.supporting_chunks]
    citation_metrics = citation_correctness(answer.citations, answer.supporting_chunks, threshold)
    return {
        "groundedness": groundedness_score(answer.text, evidence, threshold),
        "citation_count": citation_metrics["count"],
        "citation_resolvable": citation_metrics["resolvable"],
        "citation_supported": citation_metrics["supported"],
        "answer_completeness": answer_completeness(answer.text, expected_terms),
        "has_answer": 1.0 if answer.text.strip() else 0.0,
    }
