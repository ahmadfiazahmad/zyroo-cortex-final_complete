"""Documents — Weeks 3-4: upload, extract, chunk, embed, CRUD.

Upload limits: max 10 MB, only .pdf / .docx / .txt.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..chunking import chunk_text
from ..config import settings
from ..database import get_db
from ..deps import (
    enforce_personal_quota,
    enforce_workspace_quota,
    get_current_user,
    get_workspace,
    is_workspace_member,
)
from ..embeddings import embed_texts
from ..models import Chunk, Document, User
from ..schemas import DocumentOut

router = APIRouter(prefix="/api/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {"pdf": "pdf", "docx": "docx", "txt": "txt"}


def extract_text(filename: str, data: bytes) -> tuple:
    """Extract plain text + page count by file type."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "txt":
        return data.decode("utf-8", errors="ignore"), 0
    if ext == "pdf":
        from pypdf import PdfReader
        import io

        reader = PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages), len(reader.pages)
    if ext == "docx":
        from docx import Document as DocxDocument
        import io

        doc = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs), 0
    raise HTTPException(status_code=400, detail="Unsupported file type")


@router.post("", response_model=DocumentOut, summary="Upload a document")
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: str = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Multipart upload: validate, extract, chunk (~500 tokens, ~50 overlap),
    embed each chunk, mark the document ready. Usage counts against the
    workspace's shared Team quota when workspace_id is given, otherwise
    against the user's own plan quota."""
    filename = (file.filename or "").strip()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail="Only .pdf, .docx and .txt files are allowed"
        )
    data = await file.read()
    # effective_max_upload_mb is 4 MB on Vercel (its request-body limit
    # is ~4.5 MB) and MAX_UPLOAD_MB elsewhere.
    max_bytes = settings.effective_max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large — max {settings.effective_max_upload_mb} MB",
        )

    workspace = None
    if workspace_id:
        workspace = get_workspace(db, workspace_id)
        if not is_workspace_member(db, workspace_id, current_user.id):
            raise HTTPException(status_code=403, detail="You are not a member of this workspace")
        enforce_workspace_quota(workspace, "document")
    else:
        enforce_personal_quota(db, current_user, "document")

    doc = Document(
        owner_id=current_user.id,
        workspace_id=workspace_id or None,
        filename=filename,
        file_type=ALLOWED_EXTENSIONS[ext],
        status="processing",
        size_bytes=len(data),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    try:
        text, page_count = extract_text(filename, data)
        doc.page_count = page_count
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No readable text found in file")
        embeddings = embed_texts([c["text"] for c in chunks])
        for i, chunk_data in enumerate(chunks):
            chunk = Chunk(
                document_id=doc.id,
                chunk_index=i,
                text=chunk_data["text"],
                token_count=chunk_data["token_count"],
            )
            chunk.set_embedding(embeddings[i])
            db.add(chunk)
        doc.status = "ready"
        # Count usage (personal or shared workspace).
        if workspace is not None:
            workspace.documents_uploaded_count += 1
            workspace.storage_used_bytes += len(data)
        else:
            current_user.documents_uploaded_count += 1
            current_user.storage_used_bytes += len(data)
        db.commit()
    except Exception:
        doc.status = "failed"
        db.commit()
        raise HTTPException(status_code=400, detail="Could not process this file")
    db.refresh(doc)
    return doc


@router.get("", response_model=list[DocumentOut], summary="List my documents")
def list_documents(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Documents I own, plus documents in workspaces I belong to."""
    from ..models import WorkspaceMember

    ws_ids = [
        m.workspace_id
        for m in db.query(WorkspaceMember)
        .filter(WorkspaceMember.user_id == current_user.id)
        .all()
    ]
    docs = db.query(Document).filter(Document.owner_id == current_user.id).all()
    if ws_ids:
        docs += db.query(Document).filter(Document.workspace_id.in_(ws_ids)).all()
    # De-duplicate (a doc can match both) and sort newest first.
    seen = {}
    for d in docs:
        seen[d.id] = d
    return sorted(seen.values(), key=lambda d: d.uploaded_at or d.id, reverse=True)


@router.get("/{doc_id}", response_model=DocumentOut, summary="Get one document")
def get_document(
    doc_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.owner_id != current_user.id and not (
        doc.workspace_id and is_workspace_member(db, doc.workspace_id, current_user.id)
    ):
        raise HTTPException(status_code=403, detail="You cannot access this document")
    return doc


@router.delete("/{doc_id}", summary="Delete a document and its chunks")
def delete_document(
    doc_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the owner can delete this document")
    db.delete(doc)  # chunks cascade-delete
    db.commit()
    return {"message": "Document deleted", "id": doc_id}
