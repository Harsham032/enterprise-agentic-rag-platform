"""Relational persistence.

The same SQLAlchemy Core schema runs on SQLite and PostgreSQL. SQLite is the
default so tests and the local demo need no running service; PostgreSQL is what
the compose stack and the deployment target use. Keeping one schema definition
rather than two avoids the usual drift where the test database and the
production database diverge.

``sql/schema.sql`` is the PostgreSQL-native form, including the pgvector column
and index that this Core definition cannot express portably.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from ..logging_utils import get_logger
from .models import Answer, Chunk, Document

logger = get_logger(__name__)

metadata = MetaData()

documents_table = Table(
    "documents",
    metadata,
    Column("doc_id", String(32), primary_key=True),
    Column("source_type", String(48), nullable=False, index=True),
    Column("title", Text, nullable=False, default=""),
    Column("source_file", String(255)),
    Column("url", Text),
    Column("company_name", String(255)),
    Column("ticker", String(16)),
    Column("cik", String(10), index=True),
    Column("form_type", String(16)),
    Column("fiscal_year", Integer),
    Column("filing_date", Date),
    Column("pmid", String(32), index=True),
    Column("journal", String(255)),
    Column("publication_year", Integer),
    Column("char_length", Integer, nullable=False, default=0),
)

chunks_table = Table(
    "chunks",
    metadata,
    Column("chunk_id", String(32), primary_key=True),
    Column("doc_id", String(32), nullable=False, index=True),
    Column("position", Integer, nullable=False),
    Column("section_heading", Text),
    Column("text", Text, nullable=False),
    Column("token_count", Integer, nullable=False, default=0),
    Column("char_start", Integer, nullable=False, default=0),
    Column("char_end", Integer, nullable=False, default=0),
)

financial_facts_table = Table(
    "financial_facts",
    metadata,
    Column("cik", String(10), primary_key=True),
    Column("concept", String(64), primary_key=True),
    Column("fiscal_year", Integer, primary_key=True),
    Column("unit", String(16), primary_key=True),
    Column("entity_name", String(255), nullable=False, default=""),
    Column("value", Float, nullable=False),
    Column("end_date", String(10)),
    Column("accession_number", String(32)),
)

query_log_table = Table(
    "query_log",
    metadata,
    Column("query_id", String(36), primary_key=True),
    Column("query", Text, nullable=False),
    Column("strategy", String(32), nullable=False, default="single_hop"),
    Column("tools_used", Text, nullable=False, default=""),
    Column("evidence_count", Integer, nullable=False, default=0),
    Column("groundedness", Float, nullable=False, default=0.0),
    Column("latency_ms", Float, nullable=False, default=0.0),
)


class MetadataStore:
    """Read/write access to document, chunk, fact and query-log tables."""

    def __init__(self, database_url: str, *, echo: bool = False) -> None:
        if database_url.startswith("sqlite:///"):
            db_path = Path(database_url.removeprefix("sqlite:///"))
            if str(db_path) != ":memory:":
                db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(database_url, echo=echo, future=True)
        self.database_url = database_url

    def create_all(self) -> None:
        """Create any missing tables."""
        metadata.create_all(self.engine)

    def drop_all(self) -> None:
        """Drop every table. Intended for tests and for a clean re-ingest."""
        metadata.drop_all(self.engine)

    # ------------------------------------------------------------- documents

    def upsert_documents(self, documents: Sequence[Document]) -> int:
        """Replace the stored rows for ``documents`` and return the count written."""
        if not documents:
            return 0
        rows = [
            {
                "doc_id": document.doc_id,
                "source_type": document.source_type.value,
                "title": document.metadata.title,
                "source_file": getattr(document.metadata, "source_file", None),
                "url": document.metadata.url,
                "company_name": document.metadata.company_name,
                "ticker": document.metadata.ticker,
                "cik": document.metadata.cik,
                "form_type": document.metadata.form_type,
                "fiscal_year": document.metadata.fiscal_year,
                "filing_date": document.metadata.filing_date,
                "pmid": document.metadata.pmid,
                "journal": document.metadata.journal,
                "publication_year": document.metadata.publication_year,
                "char_length": len(document.text),
            }
            for document in documents
        ]
        doc_ids = [row["doc_id"] for row in rows]
        with self.engine.begin() as connection:
            connection.execute(delete(documents_table).where(documents_table.c.doc_id.in_(doc_ids)))
            connection.execute(insert(documents_table), rows)
        return len(rows)

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Replace the stored chunks for the documents ``chunks`` belong to."""
        if not chunks:
            return 0
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "position": chunk.position,
                "section_heading": chunk.section_heading,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
            for chunk in chunks
        ]
        doc_ids = sorted({chunk.doc_id for chunk in chunks})
        with self.engine.begin() as connection:
            connection.execute(delete(chunks_table).where(chunks_table.c.doc_id.in_(doc_ids)))
            connection.execute(insert(chunks_table), rows)
        return len(rows)

    def document_count(self) -> int:
        with self.engine.connect() as connection:
            return len(connection.execute(select(documents_table.c.doc_id)).fetchall())

    def chunk_count(self) -> int:
        with self.engine.connect() as connection:
            return len(connection.execute(select(chunks_table.c.chunk_id)).fetchall())

    # ------------------------------------------------------------ financials

    def upsert_financial_facts(self, rows: Iterable[dict[str, Any]]) -> int:
        """Write XBRL fact rows, replacing any existing row with the same key."""
        payload = [
            {
                "cik": row["cik"],
                "concept": row["concept"],
                "fiscal_year": int(row["fiscal_year"]),
                "unit": row["unit"],
                "entity_name": row.get("entity_name", ""),
                "value": float(row["value"]),
                "end_date": row.get("end_date"),
                "accession_number": row.get("accession_number"),
            }
            for row in rows
        ]
        if not payload:
            return 0
        with self.engine.begin() as connection:
            for row in payload:
                connection.execute(
                    delete(financial_facts_table).where(
                        financial_facts_table.c.cik == row["cik"],
                        financial_facts_table.c.concept == row["concept"],
                        financial_facts_table.c.fiscal_year == row["fiscal_year"],
                        financial_facts_table.c.unit == row["unit"],
                    )
                )
            connection.execute(insert(financial_facts_table), payload)
        return len(payload)

    def lookup_facts(
        self,
        cik: str | None = None,
        fiscal_year: int | None = None,
        *,
        concepts: Sequence[str] | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Return financial facts matching the supplied filters."""
        statement = select(financial_facts_table)
        if cik:
            statement = statement.where(financial_facts_table.c.cik == cik)
        if fiscal_year:
            statement = statement.where(financial_facts_table.c.fiscal_year == fiscal_year)
        if concepts:
            statement = statement.where(financial_facts_table.c.concept.in_(list(concepts)))
        statement = statement.order_by(
            financial_facts_table.c.fiscal_year.desc(), financial_facts_table.c.concept
        ).limit(limit)
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    # ------------------------------------------------------------- query log

    def log_query(self, answer: Answer) -> str:
        """Record a served query and return its log id."""
        query_id = str(uuid.uuid4())
        with self.engine.begin() as connection:
            connection.execute(
                insert(query_log_table),
                {
                    "query_id": query_id,
                    "query": answer.query,
                    "strategy": str(answer.diagnostics.get("strategy", "single_hop")),
                    "tools_used": ",".join(answer.tools_used),
                    "evidence_count": len(answer.supporting_chunks),
                    "groundedness": answer.groundedness,
                    "latency_ms": answer.latency_ms,
                },
            )
        return query_id

    def recent_queries(self, limit: int = 50) -> list[dict[str, Any]]:
        statement = select(query_log_table).limit(limit)
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    def close(self) -> None:
        self.engine.dispose()


def redact_url(url: str) -> str:
    """Remove credentials from a database URL before it is displayed or logged.

    ``postgresql+psycopg://rag:hunter2@db:5432/rag`` becomes
    ``postgresql+psycopg://rag:***@db:5432/rag``. The health endpoint reports
    which database is configured, and without this that report would publish the
    password to anyone who can reach the endpoint.
    """
    import re

    return re.sub(r"(//[^:/@]+):[^@]*@", r"\1:***@", url)


def postgres_schema_sql() -> str:
    """Return the PostgreSQL-native DDL shipped with the package."""
    return (Path(__file__).parent / "sql" / "schema.sql").read_text(encoding="utf-8")
