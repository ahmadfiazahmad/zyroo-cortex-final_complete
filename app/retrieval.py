"""Vector retrieval — the shared top-k search used by /api/search and RAG chat.

PostgreSQL (production): the ranking happens *in the database* with
pgvector — ``ORDER BY embedding <=> :query_vector LIMIT k`` uses the
HNSW index, so only the k best chunks ever leave the database.

SQLite (local dev / tests): the same top-k is computed in Python with
cosine similarity over the JSON-stored vectors (see chunking.py).
Both paths return the same shape: [{"score", "chunk"}, ...] with
score = cosine similarity (1 = identical).
"""
from sqlalchemy.orm import Session

from .chunking import top_k_chunks
from .database import is_postgres
from .models import Chunk


def retrieve_top_k(
    db: Session, query_embedding: list, doc_ids: list, k: int = 5
) -> list:
    """Return the k most similar chunks among ``doc_ids``."""
    if not doc_ids:
        return []
    if is_postgres():
        # pgvector: cosine distance (0 = identical) -> similarity = 1 - distance.
        # Chunk.embedding.cosine_distance() is pgvector's SQLAlchemy operator.
        rows = (
            db.query(Chunk, (1 - Chunk.embedding.cosine_distance(query_embedding)).label("score"))
            .filter(Chunk.document_id.in_(doc_ids))
            .order_by(Chunk.embedding.cosine_distance(query_embedding))
            .limit(k)
            .all()
        )
        return [
            {"score": round(float(score), 4), "chunk": chunk} for chunk, score in rows
        ]
    chunks = db.query(Chunk).filter(Chunk.document_id.in_(doc_ids)).all()
    return top_k_chunks(query_embedding, chunks, k=k)
