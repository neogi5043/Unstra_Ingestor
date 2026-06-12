-- schema.sql — PostgreSQL DDL for the PDF Ingestor POC
-- Run once:  psql -U postgres -d ingestor -f schema.sql

CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- for gen_random_uuid()

-- ── Master table: one row per ingested document ──────────────────
CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        VARCHAR(255) NOT NULL,
    content_hash    VARCHAR(64) UNIQUE,
    template_type   VARCHAR(100),
    page_count      INTEGER,
    classification  JSONB,          -- {"1": "text", "2": "scanned", …}
    status          VARCHAR(20) DEFAULT 'processing',
    error_log       TEXT,
    quality_score   FLOAT,
    uploaded_at     TIMESTAMP DEFAULT NOW()
);

-- ── Extracted key-value pairs ────────────────────────────────────
CREATE TABLE IF NOT EXISTS key_value_pairs (
    id              SERIAL PRIMARY KEY,
    doc_id          UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    key_name        VARCHAR(255),
    value           TEXT,
    page_number     INTEGER,
    confidence      FLOAT
);

-- ── Checkbox detections ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS checkboxes (
    id              SERIAL PRIMARY KEY,
    doc_id          UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    label           TEXT,
    is_checked      BOOLEAN,
    page_number     INTEGER
);

-- ── Extracted tables (stored as JSON) ────────────────────────────
CREATE TABLE IF NOT EXISTS extracted_tables (
    id              SERIAL PRIMARY KEY,
    doc_id          UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    table_index     INTEGER,
    page_number     INTEGER,
    headers         JSONB,
    rows            JSONB
);

-- ── Raw text per page (for debugging / re-extraction) ────────────
CREATE TABLE IF NOT EXISTS raw_pages (
    id              SERIAL PRIMARY KEY,
    doc_id          UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_number     INTEGER,
    page_type       VARCHAR(20),    -- 'text', 'scanned', 'text_with_images'
    raw_text        TEXT
);

-- ── Signature / image flags ──────────────────────────────────────
DROP TABLE IF EXISTS image_flags CASCADE;
CREATE TABLE image_flags (
    id              SERIAL PRIMARY KEY,
    doc_id          UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_number     INTEGER,
    image_index     INTEGER,
    width           INTEGER,
    height          INTEGER,
    x0              FLOAT,
    y0              FLOAT,
    x1              FLOAT,
    y1              FLOAT,
    image_data      TEXT,
    flag            VARCHAR(50) DEFAULT 'signature_candidate'
);

-- ── Database Migrations (for existing DBs) ───────────────────────
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64) UNIQUE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'processing';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS error_log TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS quality_score FLOAT;
