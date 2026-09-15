"""Query planning.

A single embedding of "compare Northwind revenue growth in 2024 against 2023"
sits between the two things it asks about and retrieves neither well. The
planner therefore decomposes a query into sub-queries, each of which is a
well-formed retrieval request on its own, and records which tool should serve
each one.

Decomposition is rule-based. Rules are legible, deterministic, free, and add no
latency, which matters because a planner that costs a model call per query
doubles the latency of every request. The rules cover the decomposable shapes
this corpus actually contains: comparisons between named entities, comparisons
across periods, and conjunctions of independent questions. An LLM planner is
provided for queries outside those shapes.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from ..config import PlanningConfig
from ..logging_utils import get_logger
from ..models.llm import LLMClient
from ..utils.text import content_tokens, stem

logger = get_logger(__name__)

ToolName = Literal["document_search", "financial_lookup", "biomedical_search", "sql_lookup"]

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_FISCAL_RE = re.compile(r"\b(?:fy|fiscal(?:\s+year)?)\s*(\d{2,4})\b", re.IGNORECASE)
_COMPARISON_RE = re.compile(
    r"\b(compare|compared|comparison|versus|vs\.?|against|between|difference|"
    r"changed?\s+(?:from|between)|relative\s+to)\b",
    re.IGNORECASE,
)
# Multi-word proper nouns, which in this corpus means company names.
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z.]+)+")
_CONJUNCTION_RE = re.compile(
    r"\s+and\s+(?=(?:what|how|why|which|who|when|does|did|is|are|was|were)\b)", re.IGNORECASE
)

# Vocabulary used to route a sub-query to a specialised tool. Kept explicit
# rather than learned: the routing decision must be auditable, and a misroute is
# far more expensive than a slightly wider document search.
# Stemmed at import so the vocabularies match what content_tokens produces.
_FINANCIAL_TERMS = frozenset(
    stem(term)
    for term in [
        "revenue",
        "income",
        "earnings",
        "ebitda",
        "margin",
        "profit",
        "loss",
        "eps",
        "dividend",
        "cash",
        "flow",
        "debt",
        "leverage",
        "covenant",
        "assets",
        "liabilities",
        "capex",
        "expenditure",
        "guidance",
        "filing",
        "quarter",
        "fiscal",
        "segment",
        "shareholders",
        "chargeback",
    ]
)
_BIOMEDICAL_TERMS = frozenset(
    stem(term)
    for term in [
        "patients",
        "trial",
        "cohort",
        "randomised",
        "randomized",
        "placebo",
        "efficacy",
        "dose",
        "therapy",
        "clinical",
        "endpoint",
        "mortality",
        "diagnosis",
        "prevalence",
        "incidence",
        "sensitivity",
        "specificity",
        "biomarker",
        "gene",
        "protein",
        "vaccine",
        "treatment",
        "disease",
        "enrolled",
    ]
)


def primary_entity(text: str) -> str | None:
    """Return the first multi-word proper noun in ``text``, if any.

    In this corpus a multi-word proper noun is a company name. A single
    capitalised word is not used: it is usually a sentence-initial verb.
    """
    for raw in _ENTITY_RE.findall(text):
        cleaned = _trim_leading_keywords(raw)
        if cleaned:
            return cleaned
    return None


def _trim_leading_keywords(phrase: str) -> str:
    """Drop leading comparison verbs from a candidate entity name.

    Returns an empty string when fewer than two words survive, since a
    single-word remainder is usually a sentence-initial capital rather than a
    company name.
    """
    words = phrase.split()
    while words and _COMPARISON_RE.fullmatch(words[0]):
        words.pop(0)
    return " ".join(words) if len(words) >= 2 else ""


class SubQuery(BaseModel):
    """One retrieval request derived from the user's question."""

    text: str
    tool: ToolName = "document_search"
    rationale: str = ""
    # The entity this sub-query is about, when decomposition identified one.
    # Retrieval uses it to keep a sub-query's evidence inside the right
    # document: "Northwind Energy revenue growth in 2024" and "Atlas Payments
    # revenue growth in 2024" share almost all their content words, so lexical
    # and dense scores alone do not reliably separate them.
    entity: str | None = None


class QueryPlan(BaseModel):
    """The decomposition of a query into executable sub-queries."""

    original: str
    subqueries: list[SubQuery] = Field(default_factory=list)
    is_decomposed: bool = False
    strategy: str = "single_hop"

    @property
    def tools(self) -> list[str]:
        return sorted({subquery.tool for subquery in self.subqueries})


def route(text: str) -> ToolName:
    """Pick the tool best suited to ``text``."""
    tokens = set(content_tokens(text))
    financial = len(tokens & _FINANCIAL_TERMS)
    biomedical = len(tokens & _BIOMEDICAL_TERMS)
    if financial > biomedical and financial > 0:
        return "financial_lookup"
    if biomedical > 0:
        return "biomedical_search"
    return "document_search"


class QueryPlanner:
    """Rule-based decomposition with an optional model-backed fallback."""

    def __init__(self, config: PlanningConfig, client: LLMClient | None = None) -> None:
        self.config = config
        self.client = client

    def plan(self, query: str) -> QueryPlan:
        """Return the plan for ``query``."""
        if not self.config.enabled:
            return QueryPlan(
                original=query,
                subqueries=[SubQuery(text=query, tool=route(query))],
                strategy="single_hop",
            )

        entity = primary_entity(query)
        for strategy, splitter in (
            ("temporal_comparison", self._split_on_periods),
            ("entity_comparison", self._split_on_entities),
            ("conjunction", self._split_on_conjunction),
        ):
            parts = splitter(query)
            if len(parts) > 1:
                subqueries = [
                    SubQuery(
                        text=part,
                        tool=route(part),
                        rationale=strategy,
                        # A temporal comparison is about one entity across
                        # periods, so every sub-query inherits it. An entity
                        # comparison names a different entity per sub-query.
                        entity=primary_entity(part) if strategy == "entity_comparison" else entity,
                    )
                    for part in parts[: self.config.max_subqueries]
                ]
                logger.info("query_decomposed", strategy=strategy, parts=len(subqueries))
                return QueryPlan(
                    original=query, subqueries=subqueries, is_decomposed=True, strategy=strategy
                )

        return QueryPlan(
            original=query,
            subqueries=[
                SubQuery(text=query, tool=route(query), rationale="single_hop", entity=entity)
            ],
            strategy="single_hop",
        )

    @staticmethod
    def _split_on_periods(query: str) -> list[str]:
        """Split a comparison that names two or more periods into one query per period."""
        if not _COMPARISON_RE.search(query):
            return [query]
        years = [match.group(0) for match in _YEAR_RE.finditer(query)]
        fiscal = [match.group(0) for match in _FISCAL_RE.finditer(query)]
        periods = list(dict.fromkeys(years + fiscal))
        if len(periods) < 2:
            return [query]

        # Strip the comparison scaffolding so each sub-query reads as a direct
        # question about one period.
        residual = _COMPARISON_RE.sub(" ", query)
        for period in periods:
            residual = residual.replace(period, " ")
        residual = " ".join(residual.replace(" and ", " ").split()).strip(" ?,")
        return [f"{residual} {period}".strip() for period in periods]

    @staticmethod
    def _split_on_entities(query: str) -> list[str]:
        """Split a comparison naming two capitalised entities."""
        if not _COMPARISON_RE.search(query):
            return [query]
        # Entities are matched in the original query, then any leading
        # comparison verb is trimmed off. Removing the keywords first instead
        # would merge the two entity names into one contiguous match.
        entities: list[str] = []
        for raw in _ENTITY_RE.findall(query):
            cleaned = _trim_leading_keywords(raw)
            if cleaned and cleaned not in entities:
                entities.append(cleaned)
        if len(entities) < 2:
            return [query]

        residual = _COMPARISON_RE.sub(" ", query)
        for entity in entities:
            residual = residual.replace(entity, " ")
        residual = " ".join(residual.replace(" and ", " ").split()).strip(" ?,")
        return [f"{entity} {residual}".strip() for entity in entities]

    @staticmethod
    def _split_on_conjunction(query: str) -> list[str]:
        """Split two independent questions joined by ``and``."""
        parts = [part.strip(" ?,") for part in _CONJUNCTION_RE.split(query) if part.strip()]
        if len(parts) < 2:
            return [query]
        return [part if part.endswith("?") else f"{part}?" for part in parts]
