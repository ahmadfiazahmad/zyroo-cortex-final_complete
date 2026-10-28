"""RAG chat — Week 5: sessions, messages, retrieval + Groq, SSE streaming."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import (
    enforce_personal_quota,
    enforce_workspace_quota,
    get_current_user,
    get_workspace,
    is_workspace_member,
)
from ..embeddings import embed_text
from ..llm import generate_answer
from ..models import ChatSession, Document, Message, User, WorkspaceMember
from ..retrieval import retrieve_top_k
from ..schemas import CreateSessionRequest, MessageOut, SendMessageRequest, SessionOut

router = APIRouter(prefix="/api/chat", tags=["Chat"])


def _session_visible_to(db, session, user_id: str) -> bool:
    if session.owner_id == user_id:
        return True
    return bool(session.workspace_id) and is_workspace_member(db, session.workspace_id, user_id)


def _can_access_document(db, user: User, doc: Document) -> bool:
    """True when the user owns the document, or it lives in a workspace
    they are a member of. This is the same rule /api/search uses."""
    if doc.owner_id == user.id:
        return True
    return bool(doc.workspace_id) and is_workspace_member(db, doc.workspace_id, user.id)


def _validate_scoped_documents(db, user: User, document_ids: list) -> None:
    """Reject a chat session that scopes to a document the user cannot
    access. Without this, anyone could read another user's private
    documents by passing their IDs when creating a session."""
    for doc_id in document_ids or []:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        if not _can_access_document(db, user, doc):
            raise HTTPException(
                status_code=403, detail="You cannot access one of the scoped documents"
            )


def _accessible_scoped_ids(db, user: User, document_ids: list) -> list:
    """Defence in depth: re-filter stored scoped IDs at retrieval time,
    so a session can never return chunks from a document the user lost
    access to (e.g. removed from the workspace)."""
    accessible = []
    for doc_id in document_ids or []:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc and _can_access_document(db, user, doc):
            accessible.append(doc.id)
    return accessible


def _get_session(db, session_id: str, user: User) -> ChatSession:
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    if not _session_visible_to(db, session, user.id):
        raise HTTPException(status_code=403, detail="You cannot access this chat session")
    return session


def _retrieve(db, user: User, session: ChatSession, question: str, k: int = 5) -> list:
    """Top-k chunks scoped to the session's documents / workspace / owner."""
    scoped = session.get_scoped_document_ids()
    if scoped:
        # Re-check access at retrieval time — never trust stored IDs alone.
        doc_ids = _accessible_scoped_ids(db, user, scoped)
    elif session.workspace_id:
        doc_ids = [
            d.id for d in db.query(Document).filter(Document.workspace_id == session.workspace_id).all()
        ]
    else:
        ws_ids = [
            m.workspace_id
            for m in db.query(WorkspaceMember).filter(WorkspaceMember.user_id == user.id).all()
        ]
        docs = db.query(Document).filter(Document.owner_id == user.id).all()
        if ws_ids:
            docs += db.query(Document).filter(Document.workspace_id.in_(ws_ids)).all()
        doc_ids = list({d.id for d in docs})
    # pgvector (Postgres): ranked in the database. SQLite: ranked in Python.
    ranked = retrieve_top_k(db, embed_text(question), doc_ids, k=k)
    return [
        {
            "text": item["chunk"].text,
            "document_id": item["chunk"].document_id,
            "chunk_index": item["chunk"].chunk_index,
            "score": item["score"],
        }
        for item in ranked
    ]


def _answer_and_store(db, user: User, session: ChatSession, content: str) -> Message:
    """Quota -> retrieve -> LLM/fallback -> store user + assistant messages."""
    workspace = None
    if session.workspace_id:
        workspace = get_workspace(db, session.workspace_id)
        enforce_workspace_quota(workspace, "message")
    else:
        enforce_personal_quota(db, user, "message")

    user_msg = Message(session_id=session.id, role="user", content=content)
    db.add(user_msg)
    db.commit()

    retrieved = _retrieve(db, user, session, content)
    answer, prompt_tokens, completion_tokens = generate_answer(content, retrieved)
    sources = [
        {"document_id": r["document_id"], "chunk_index": r["chunk_index"], "text": r["text"][:500], "score": r["score"]}
        for r in retrieved
    ]
    assistant_msg = Message(
        session_id=session.id,
        role="assistant",
        content=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_used=prompt_tokens + completion_tokens,
    )
    assistant_msg.set_sources(sources)
    db.add(assistant_msg)
    if workspace is not None:
        workspace.messages_sent_count += 1
    else:
        user.messages_sent_count += 1
    db.commit()
    db.refresh(assistant_msg)
    return assistant_msg


@router.post("/sessions", response_model=SessionOut, summary="Create a chat session")
def create_session(
    data: CreateSessionRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    if data.workspace_id:
        workspace = get_workspace(db, data.workspace_id)
        if not is_workspace_member(db, data.workspace_id, current_user.id):
            raise HTTPException(status_code=403, detail="You are not a member of this workspace")
    # Every scoped document must be one the creator can actually access.
    _validate_scoped_documents(db, current_user, data.document_ids or [])
    session = ChatSession(
        owner_id=current_user.id,
        workspace_id=data.workspace_id or None,
        title=data.title or "New chat",
    )
    session.set_scoped_document_ids(data.document_ids or [])
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=list[SessionOut], summary="List my chat sessions")
def list_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    ws_ids = [
        m.workspace_id
        for m in db.query(WorkspaceMember).filter(WorkspaceMember.user_id == current_user.id).all()
    ]
    sessions = db.query(ChatSession).filter(ChatSession.owner_id == current_user.id).all()
    if ws_ids:
        sessions += db.query(ChatSession).filter(ChatSession.workspace_id.in_(ws_ids)).all()
    seen = {s.id: s for s in sessions}
    return sorted(seen.values(), key=lambda s: s.created_at or s.id, reverse=True)


@router.delete("/sessions/{session_id}", summary="Delete a chat session")
def delete_session(
    session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    if session.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the owner can delete this session")
    db.delete(session)  # messages cascade-delete
    db.commit()
    return {"message": "Chat session deleted", "id": session_id}


@router.post("/sessions/{session_id}/messages", response_model=MessageOut, summary="Send a chat message (RAG)")
def send_message(
    session_id: str,
    data: SendMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve relevant chunks, call Groq (or the grounded fallback), and
    return the assistant reply with its cited sources."""
    session = _get_session(db, session_id, current_user)
    return _answer_and_store(db, current_user, session, data.content)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut], summary="Get message history")
def get_messages(
    session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    session = _get_session(db, session_id, current_user)
    return (
        db.query(Message)
        .filter(Message.session_id == session.id)
        .order_by(Message.created_at)
        .all()
    )


@router.post("/sessions/{session_id}/messages/stream", summary="Send a message, streamed (SSE)")
def send_message_stream(
    session_id: str,
    data: SendMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Same RAG flow as POST .../messages, streamed back token-by-token
    (word-by-word here) as Server-Sent Events."""
    session = _get_session(db, session_id, current_user)
    assistant_msg = _answer_and_store(db, current_user, session, data.content)
    words = assistant_msg.content.split(" ")

    def event_stream():
        for word in words:
            yield f"data: {word} \n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
