-- Relational schema for document provenance, chunk metadata and structured
-- financial facts. Written for PostgreSQL; the SQLAlchemy layer in store.py
-- creates an equivalent schema on SQLite for local development and tests.
--
-- Embeddings live in the vector store rather than here. The optional pgvector
-- column below keeps them alongside the chunks when PostgreSQL is the vector
-- backend, which removes a consistency problem: a chunk and its embedding are
-- then inserted and deleted in the same transaction.

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source_type     TEXT        NOT NULL,
    title           TEXT        NOT NULL DEFAULT '',
    source_file     TEXT,
    url             TEXT,
    company_name    TEXT,
    ticker          TEXT,
    cik             CHAR(10),
    form_type       TEXT,
    fiscal_year     INTEGER,
    filing_date     DATE,
    pmid            TEXT,
    journal         TEXT,
    publication_year INTEGER,
    char_length     INTEGER     NOT NULL DEFAULT 0,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_source_type_idx ON documents (source_type);
CREATE INDEX IF NOT EXISTS documents_cik_year_idx    ON documents (cik, fiscal_year);
CREATE INDEX IF NOT EXISTS documents_pmid_idx        ON documents (pmid);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT        NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    position        INTEGER     NOT NULL,
    section_heading TEXT,
    text            TEXT        NOT NULL,
    token_count     INTEGER     NOT NULL DEFAULT 0,
    char_start      INTEGER     NOT NULL DEFAULT 0,
    char_end        INTEGER     NOT NULL DEFAULT 0
    -- With the pgvector extension installed and PostgreSQL as the vector
    -- backend, add:
    --   ALTER TABLE chunks ADD COLUMN embedding vector(768);
    --   CREATE INDEX chunks_embedding_idx ON chunks
    --       USING hnsw (embedding vector_cosine_ops);
);

CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id, position);

-- Structured XBRL facts, used by the financial lookup tool for questions where
-- an exact figure is wanted rather than the sentence that mentions it.
CREATE TABLE IF NOT EXISTS financial_facts (
    cik              CHAR(10)      NOT NULL,
    entity_name      TEXT          NOT NULL DEFAULT '',
    concept          TEXT          NOT NULL,
    unit             TEXT          NOT NULL,
    fiscal_year      INTEGER       NOT NULL,
    value            DOUBLE PRECISION NOT NULL,
    end_date         DATE,
    accession_number TEXT,
    PRIMARY KEY (cik, concept, fiscal_year, unit)
);

CREATE INDEX IF NOT EXISTS financial_facts_lookup_idx
    ON financial_facts (cik, fiscal_year);

-- Query log, which is what makes retrieval quality measurable in production
-- rather than only at benchmark time.
CREATE TABLE IF NOT EXISTS query_log (
    query_id        TEXT PRIMARY KEY,
    query           TEXT        NOT NULL,
    strategy        TEXT        NOT NULL DEFAULT 'single_hop',
    tools_used      TEXT        NOT NULL DEFAULT '',
    evidence_count  INTEGER     NOT NULL DEFAULT 0,
    groundedness    REAL        NOT NULL DEFAULT 0.0,
    latency_ms      REAL        NOT NULL DEFAULT 0.0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS query_log_created_idx ON query_log (created_at DESC);
