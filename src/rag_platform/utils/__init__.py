"""Shared helpers: text normalisation, deterministic seeding and timing."""

from .seeds import set_global_seed
from .text import (
    containment,
    content_tokens,
    jaccard_overlap,
    normalise_whitespace,
    split_sentences,
    stable_hash,
    stem,
    tokenize,
    truncate,
)
from .timing import Stopwatch, latency_summary, percentile

__all__ = [
    "Stopwatch",
    "containment",
    "content_tokens",
    "jaccard_overlap",
    "latency_summary",
    "normalise_whitespace",
    "percentile",
    "set_global_seed",
    "split_sentences",
    "stable_hash",
    "stem",
    "tokenize",
    "truncate",
]
