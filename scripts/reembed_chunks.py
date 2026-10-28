"""Re-embed every chunk with the currently configured embedding provider.

Run after migrating to pgvector, or after switching EMBEDDING_PROVIDER
(e.g. from the hash fallback to fastembed) — old vectors were produced
by a different model, so they must be regenerated for similarity search
to be meaningful.

Usage:
    EMBEDDING_PROVIDER=fastembed python scripts/reembed_chunks.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import SessionLocal, init_db
from app.embeddings import embed_texts
from app.models import Chunk

BATCH = 32


def main():
    init_db()
    db = SessionLocal()
    try:
        chunks = db.query(Chunk).order_by(Chunk.document_id, Chunk.chunk_index).all()
        print(f"Re-embedding {len(chunks)} chunks...")
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            vectors = embed_texts([c.text for c in batch])
            for chunk, vector in zip(batch, vectors):
                chunk.set_embedding(vector)
            db.commit()
            print(f"  {min(i + BATCH, len(chunks))}/{len(chunks)} done")
        print("Done.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
