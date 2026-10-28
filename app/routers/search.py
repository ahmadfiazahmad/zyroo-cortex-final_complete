"""Semantic search — Weeks 3-4: cosine similarity over chunk embeddings."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user, is_workspace_member
from ..embeddings import embed_text
from ..models import Document, User, WorkspaceMember
from ..retrieval import retrieve_top_k
from ..schemas import SearchRequest, SearchResponse, SearchResult

router = APIRouter(prefix="/api/search", tags=["Search"])


@router.post("", response_model=SearchResponse, summary="Semantic search over my documents")
def search(data: SearchRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Embed the query and return the top-k most similar chunks, scoped to
    one document, one workspace, or all documents I can access."""
    query_embedding = embed_text(data.query)

    if data.document_id:
        doc = db.query(Document).filter(Document.id == data.document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        if doc.owner_id != current_user.id and not (
            doc.workspace_id and is_workspace_member(db, doc.workspace_id, current_user.id)
        ):
            raise HTTPException(status_code=403, detail="You cannot search this document")
        doc_ids = [doc.id]
    elif data.workspace_id:
        if not is_workspace_member(db, data.workspace_id, current_user.id):
            raise HTTPException(status_code=403, detail="You are not a member of this workspace")
        doc_ids = [
            d.id for d in db.query(Document).filter(Document.workspace_id == data.workspace_id).all()
        ]
    else:
        ws_ids = [
            m.workspace_id
            for m in db.query(WorkspaceMember).filter(WorkspaceMember.user_id == current_user.id).all()
        ]
        docs = db.query(Document).filter(Document.owner_id == current_user.id).all()
        if ws_ids:
            docs += db.query(Document).filter(Document.workspace_id.in_(ws_ids)).all()
        doc_ids = list({d.id for d in docs})

    # pgvector (Postgres): ranked in the database. SQLite: ranked in Python.
    ranked = retrieve_top_k(db, query_embedding, doc_ids, k=data.top_k)
    return SearchResponse(
        query=data.query,
        results=[
            SearchResult(
                document_id=item["chunk"].document_id,
                chunk_index=item["chunk"].chunk_index,
                text=item["chunk"].text,
                score=item["score"],
            )
            for item in ranked
        ],
    )
