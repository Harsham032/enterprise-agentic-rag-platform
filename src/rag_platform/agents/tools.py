"""Specialised retrieval tools.

Each tool wraps the same index with a different scope or a different backing
store, and each returns evidence in the same shape so the orchestrator can merge
results without knowing which tool produced them.

``document_search``   unrestricted hybrid retrieval over every source.
``financial_lookup``  hybrid retrieval restricted to filings, plus exact XBRL
                      figures from the structured store when the query names a
                      company and a fiscal year.
``biomedical_search`` hybrid retrieval restricted to the biomedical source.
``sql_lookup``        direct structured query against the financial facts table,
                      for questions a text index answers badly ("largest
                      revenue in 2024").
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ..data.models import ScoredChunk, SourceType
from ..logging_utils import get_logger
from ..retrieval.index import RetrievalIndex
from ..utils.timing import Stopwatch

logger = get_logger(__name__)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


class ToolResult(BaseModel):
    """What a tool returns to the orchestrator."""

    tool: str
    query: str
    chunks: list[ScoredChunk] = Field(default_factory=list)
    structured: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


ToolFn = Callable[[str, int, "str | None"], ToolResult]


class ToolRegistry:
    """Holds the callable tools available to the orchestrator."""

    def __init__(
        self,
        index: RetrievalIndex,
        *,
        fact_lookup: Callable[[str | None, int | None], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.index = index
        self._fact_lookup = fact_lookup
        self._tools: dict[str, ToolFn] = {
            "document_search": self.document_search,
            "financial_lookup": self.financial_lookup,
            "biomedical_search": self.biomedical_search,
            "sql_lookup": self.sql_lookup,
        }

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, tool: str, query: str, top_k: int, entity: str | None = None) -> ToolResult:
        """Dispatch to ``tool``, degrading to document search if it is unknown."""
        handler = self._tools.get(tool)
        if handler is None:
            logger.warning("unknown_tool", tool=tool)
            return self.document_search(query, top_k, entity)
        try:
            return handler(query, top_k, entity)
        except Exception as exc:  # defensive: one failing tool must not fail the query
            logger.warning("tool_failed", tool=tool, error=str(exc))
            return ToolResult(tool=tool, query=query, error=str(exc))

    # ------------------------------------------------------------------ tools

    def document_search(self, query: str, top_k: int, entity: str | None = None) -> ToolResult:
        with Stopwatch() as sw:
            chunks = self.index.search(query, top_k=top_k, entity=entity)
        return ToolResult(
            tool="document_search", query=query, chunks=chunks, latency_ms=sw.elapsed_ms
        )

    def financial_lookup(self, query: str, top_k: int, entity: str | None = None) -> ToolResult:
        with Stopwatch() as sw:
            chunks = self.index.search(
                query, top_k=top_k, source_types={SourceType.SEC_FILING.value}, entity=entity
            )
            structured = self._structured_facts(query, chunks)
        return ToolResult(
            tool="financial_lookup",
            query=query,
            chunks=chunks,
            structured=structured,
            latency_ms=sw.elapsed_ms,
        )

    def biomedical_search(self, query: str, top_k: int, entity: str | None = None) -> ToolResult:
        with Stopwatch() as sw:
            chunks = self.index.search(
                query,
                top_k=top_k,
                source_types={SourceType.BIOMEDICAL_ABSTRACT.value},
                entity=entity,
            )
        return ToolResult(
            tool="biomedical_search", query=query, chunks=chunks, latency_ms=sw.elapsed_ms
        )

    def sql_lookup(self, query: str, top_k: int, entity: str | None = None) -> ToolResult:
        with Stopwatch() as sw:
            # Entity resolution goes through retrieval rather than through a
            # name-matching rule: the CIK is carried on the document metadata,
            # and retrieval is already the component that maps a name onto a
            # document.
            resolved = (
                self.index.search(
                    entity, top_k=3, source_types={SourceType.SEC_FILING.value}, entity=entity
                )
                if entity
                else []
            )
            rows = self._structured_facts(query, resolved)
        return ToolResult(
            tool="sql_lookup", query=query, structured=rows[:top_k], latency_ms=sw.elapsed_ms
        )

    # ------------------------------------------------------------------ facts

    def _structured_facts(self, query: str, chunks: list[ScoredChunk]) -> list[dict[str, Any]]:
        """Fetch exact figures from the structured store when one is wired up.

        The CIK is taken from the retrieved chunks rather than parsed out of the
        query: entity resolution is what retrieval is already good at, and
        reusing it avoids a brittle name-matching rule.
        """
        if self._fact_lookup is None:
            return []
        years: list[int | None] = [int(match.group(0)) for match in _YEAR_RE.finditer(query)]
        ciks = [chunk.chunk.metadata.cik for chunk in chunks if chunk.chunk.metadata.cik]
        cik = ciks[0] if ciks else None
        rows: list[dict[str, Any]] = []
        for year in years or [None]:
            rows.extend(self._fact_lookup(cik, year))
        return rows
