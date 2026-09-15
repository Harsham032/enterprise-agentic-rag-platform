"""Exception hierarchy for the platform.

Every failure raised by library code derives from :class:`RagPlatformError`, so
callers can distinguish expected domain failures from genuine bugs.
"""

from __future__ import annotations


class RagPlatformError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(RagPlatformError):
    """Raised when configuration is missing, malformed or internally inconsistent."""


class DataAcquisitionError(RagPlatformError):
    """Raised when an upstream document source cannot be reached or returns an unusable payload."""


class ParsingError(RagPlatformError):
    """Raised when a document cannot be decoded into text."""


class IndexNotBuiltError(RagPlatformError):
    """Raised when a retriever is queried before its index has been built."""


class EmbeddingError(RagPlatformError):
    """Raised when an embedding backend fails or is misconfigured."""


class GenerationError(RagPlatformError):
    """Raised when an answer cannot be synthesised from the retrieved evidence."""


class EvaluationError(RagPlatformError):
    """Raised when an evaluation run is given inconsistent inputs."""
