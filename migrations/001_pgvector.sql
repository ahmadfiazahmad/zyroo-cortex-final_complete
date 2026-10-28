-- Migration 001 — pgvector storage for chunk embeddings (PostgreSQL only)
--
-- Run this ONCE against an existing PostgreSQL database that was created
-- by the old JSON version of the app (fresh databases don't need it:
-- init_db() creates the Vector(384) column directly).
--
--   psql "$DATABASE_URL" -f migrations/001_pgvector.sql
--   python scripts/reembed_chunks.py   # re-embed with the real model
--
-- Step 1: enable the extension.
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: convert the old JSON-text column to a real vector column.
-- The old values are JSON like "[0.01, -0.02, ...]", which casts to vector.
ALTER TABLE chunks
  ALTER COLUMN embedding TYPE vector(384)
  USING embedding::vector(384);

-- Step 3: HNSW index for fast cosine search (used by ORDER BY embedding <=> ...).
CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
