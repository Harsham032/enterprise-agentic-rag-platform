"""Text normalisation and lightweight linguistic helpers.

The platform deliberately avoids a heavyweight NLP dependency for these
operations: the corpora are long-form English documents where regular
expressions over normalised whitespace are accurate enough, and keeping the
dependency surface small means the ingestion path runs anywhere.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'0-9])")

# Abbreviations that must not terminate a sentence. Financial and biomedical
# prose is dense with these, and splitting on them fragments the evidence spans
# that citations are attached to.
_ABBREVIATIONS = frozenset(
    {
        "inc.",
        "corp.",
        "ltd.",
        "co.",
        "llc.",
        "plc.",
        "u.s.",
        "u.k.",
        "e.g.",
        "i.e.",
        "vs.",
        "approx.",
        "no.",
        "fig.",
        "dr.",
        "mr.",
        "ms.",
        "st.",
        "jan.",
        "feb.",
        "mar.",
        "apr.",
        "jun.",
        "jul.",
        "aug.",
        "sep.",
        "sept.",
        "oct.",
        "nov.",
        "dec.",
    }
)

# Very common English words carry no retrieval signal but dominate token
# overlap scores, so support checks and keyword extraction drop them.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "had",
        "he",
        "her",
        "his",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
        "our",
        "we",
        "us",
    ]
)


def normalise_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace and excess blank lines."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKLINES_RE.sub("\n\n", text).strip()


def tokenize(text: str, *, drop_stopwords: bool = False, stemming: bool = False) -> list[str]:
    """Lowercase word tokenisation used by lexical retrieval and overlap checks."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    if stemming:
        tokens = [stem(token) for token in tokens]
    return tokens


def content_tokens(text: str) -> list[str]:
    """Stemmed content words: the token form used by every overlap measure.

    Support checks, citation verification and answer completeness all compare a
    short claim against a passage, where inflectional mismatch is the dominant
    source of false negatives. Retrieval scoring keeps stemming configurable so
    its contribution can be measured; overlap measures always stem.
    """
    return tokenize(text, drop_stopwords=True, stemming=True)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping common abbreviations intact."""
    if not text.strip():
        return []

    pieces = _SENTENCE_END_RE.split(normalise_whitespace(text).replace("\n", " "))
    sentences: list[str] = []
    for piece in pieces:
        candidate = piece.strip()
        if not candidate:
            continue
        if sentences:
            previous = sentences[-1]
            last_word = previous.rsplit(" ", 1)[-1].lower()
            # Re-join when the previous fragment ended on an abbreviation or a
            # single capitalised initial such as "J. Smith".
            if last_word in _ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last_word):
                sentences[-1] = f"{previous} {candidate}"
                continue
        sentences.append(candidate)
    return sentences


def jaccard_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    """Return the Jaccard similarity of two token collections."""
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return intersection / union if union else 0.0


def containment(needle: Iterable[str], haystack: Iterable[str]) -> float:
    """Fraction of ``needle`` tokens that also appear in ``haystack``.

    Asymmetric by design: a claim is supported when its own content words are
    present in the evidence, regardless of how much extra text the evidence has.
    """
    needle_set = set(needle)
    if not needle_set:
        return 0.0
    return len(needle_set & set(haystack)) / len(needle_set)


def stable_hash(*parts: str, length: int = 16) -> str:
    """Deterministic short identifier, stable across processes and platforms."""
    digest = hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


def truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    """Shorten ``text`` on a word boundary."""
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - len(suffix)]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped + suffix


# --- stemming -------------------------------------------------------------
#
# A deliberately conservative suffix stripper rather than a full Porter
# implementation. Retrieval over filings and abstracts is dominated by
# inflectional mismatch ("turbine" vs "turbines", "edit" vs "editing") which
# these rules cover, while aggressive derivational stemming would conflate
# domain terms that must stay distinct - "operating" and "operation" mean
# different things in a financial statement, and "calibration" is not
# "calibrated". Every rule below is inflectional only.

_DOUBLE_CONSONANT_RE = re.compile(r"([bdfglmnprt])\1$")
_VOWEL_RE = re.compile(r"[aeiouy]")

_IRREGULAR = {
    "was": "be",
    "were": "be",
    "is": "be",
    "are": "be",
    "been": "be",
    "has": "have",
    "had": "have",
    "having": "have",
    "does": "do",
    "did": "do",
    "done": "do",
    "data": "datum",
    "criteria": "criterion",
    "phenomena": "phenomenon",
    "bacteria": "bacterium",
    "indices": "index",
    "matrices": "matrix",
    "media": "medium",
    # Greek-origin "-sis" plurals: the generic "-ses" rule mangles these.
    "analyses": "analysis",
    "bases": "basis",
    "crises": "crisis",
    "diagnoses": "diagnosis",
    "hypotheses": "hypothesis",
    "theses": "thesis",
    "prognoses": "prognosis",
    "parentheses": "parenthesis",
    "syntheses": "synthesis",
    "metastases": "metastasis",
    "anastomoses": "anastomosis",
}

# Words whose plural-looking ending is part of the stem.
_PROTECTED_SUFFIXES = ("sis", "ss", "us", "is", "ies")


def _has_vowel(word: str) -> bool:
    return bool(_VOWEL_RE.search(word))


def stem(word: str) -> str:
    """Reduce an English word to a conservative inflectional stem.

    >>> stem("turbines"), stem("editing"), stem("reported"), stem("analyses")
    ('turbine', 'edit', 'report', 'analysis')
    """
    if len(word) <= 3 or not word.isalpha():
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]

    # Plurals.
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif (
        word.endswith("sses")
        or word.endswith(("ches", "shes", "xes", "zes", "ses"))
        and len(word) > 4
    ):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(_PROTECTED_SUFFIXES):
        word = word[:-1]

    # Verb inflections, only when a vowel survives in the stem.
    if word.endswith("ing") and len(word) > 5 and _has_vowel(word[:-3]):
        word = word[:-3]
        word = _DOUBLE_CONSONANT_RE.sub(r"\1", word)
        if word.endswith(("at", "iz", "bl", "us", "iv")):
            word += "e"
    elif word.endswith("ed") and len(word) > 4 and _has_vowel(word[:-2]):
        candidate = word[:-2]
        candidate = _DOUBLE_CONSONANT_RE.sub(r"\1", candidate)
        # "increased" -> "increase", not "increas"
        word = (
            candidate + "e"
            if candidate.endswith(("at", "iz", "bl", "us", "iv", "s", "c"))
            else candidate
        )

    return word
