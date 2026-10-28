"""Database — SQLAlchemy. SQLite by default, PostgreSQL + pgvector via DATABASE_URL.

Serverless (Vercel + Neon) notes:
- Use Neon's *pooled* connection string in DATABASE_URL
  (``...-pooler.<region>.aws.neon.tech/...?sslmode=require``). The direct
  string is for migrations only — serverless functions can open hundreds
  of connections and exhaust Postgres without the pooler.
- On Vercel the engine uses NullPool: each invocation opens a connection
  and returns it immediately, so no stale pooled connection survives a
  Neon scale-to-zero. pool_pre_ping discards stale connections anyway.
- Neon requires SSL (sslmode=require).
"""
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

from .config import settings

connect_args = {}
engine_kwargs: dict = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
elif settings.database_url.startswith(("postgres://", "postgresql://")):
    connect_args = {"sslmode": "require"}
    engine_kwargs["pool_pre_ping"] = True  # discard stale connections after scale-to-zero
    if os.getenv("VERCEL") == "1":
        engine_kwargs["poolclass"] = NullPool
    else:
        engine_kwargs["pool_size"] = 5
        engine_kwargs["max_overflow"] = 10
        engine_kwargs["pool_recycle"] = 300

engine = create_engine(settings.database_url, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def is_postgres() -> bool:
    """True when the app is running against PostgreSQL (pgvector mode)."""
    return settings.database_url.startswith(("postgres://", "postgresql://"))


def get_db():
    """FastAPI dependency: one DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables (no migrations needed for this project size).

    On PostgreSQL this also enables the pgvector extension first, so the
    Chunk.embedding Vector(384) column and its HNSW index can be created.
    Neon supports pgvector — CREATE EXTENSION vector works there.

    Serverless safety: every Vercel cold start runs this function, and
    several instances can start at the same moment. Without a lock, two
    of them can both pass create_all's "does the table exist?" check and
    both issue CREATE TABLE users — creating a table also creates a
    composite type of the same name, so Postgres fails the loser with
    "duplicate key value violates unique constraint
    pg_type_typname_nsp_index ... (users, 2200) already exists", and the
    failed lifespan kills that whole instance ("Application startup
    failed"). A transaction-scoped advisory lock serializes concurrent
    startups (it is released at commit, so it is safe with Neon's
    transaction-mode pooler), and the retry below covers any leftover
    "already exists" race.
    """
    import time

    import sqlalchemy.exc

    from . import models  # noqa: F401  (import registers the models)

    last_exc = None
    for attempt in range(3):
        try:
            if is_postgres():
                with engine.begin() as conn:
                    # 727001 is an arbitrary fixed key for this app's init lock.
                    conn.execute(text("SELECT pg_advisory_xact_lock(727001)"))
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    Base.metadata.create_all(bind=conn)
            else:
                Base.metadata.create_all(bind=engine)
            return
        except sqlalchemy.exc.DBAPIError as exc:
            # Only retry the table-creation race; any other DB error is real.
            if "already exists" not in str(exc):
                raise
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    raise last_exc
