"""Database models — Weeks 1-8.

User, Document, Chunk, ChatSession, Message (Weeks 1-5),
billing fields on User (Week 6), Workspace + WorkspaceMember (Weeks 7-8).
"""
import json
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from .config import settings
from .database import Base
from .embeddings import DIM
from .plans import BILLING_PERIOD_DAYS


def is_postgres_database(url: str | None = None) -> bool:
    """True when the configured database is PostgreSQL (pgvector lives there)."""
    url = url if url is not None else settings.database_url
    return (url or "").startswith(("postgres://", "postgresql://"))


def embedding_column_type(url: str | None = None):
    """The storage type for chunk embeddings.

    PostgreSQL -> pgvector Vector(384), the real vector column with a
    similarity index. Anything else (SQLite locally / in tests) ->
    plain Text holding a JSON list, so the project still runs anywhere
    with zero setup.
    """
    if is_postgres_database(url):
        from pgvector.sqlalchemy import Vector

        return Vector(DIM)
    return Text()


def chunk_embedding_index():
    """HNSW index for fast cosine search — PostgreSQL only.

    Valid pgvector parameters are m and ef_construction (m2 does not
    exist — see the regression test in tests/test_unit.py).
    """
    return Index(
        "ix_chunks_embedding_hnsw",
        "embedding",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


# Only attached to the table on PostgreSQL; SQLite has no pgvector.
_CHUNK_TABLE_ARGS = (chunk_embedding_index(),) if is_postgres_database() else ()


def _id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_id)
    name = Column(String(200), nullable=False)
    email = Column(String(320), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user")  # user | admin
    is_suspended = Column(Boolean(), default=False)
    plan_tier = Column(String(20), default="free")  # free | pro | team
    stripe_customer_id = Column(String(100), nullable=True)
    subscription_id = Column(String(100), nullable=True)
    # active | past_due | cancelling | canceled
    subscription_status = Column(String(20), default="active")
    documents_uploaded_count = Column(Integer(), default=0)
    messages_sent_count = Column(Integer(), default=0)
    storage_used_bytes = Column(Integer(), default=0)
    current_period_start = Column(DateTime(timezone=True), default=_now)
    current_period_end = Column(
        DateTime(timezone=True),
        default=lambda: _now() + timedelta(days=BILLING_PERIOD_DAYS),
    )
    created_at = Column(DateTime(timezone=True), default=_now)

    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="owner", cascade="all, delete-orphan")
    workspaces_owned = relationship("Workspace", back_populates="owner", cascade="all, delete-orphan")


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(String(36), primary_key=True, default=_id)
    name = Column(String(200), nullable=False)
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    plan_tier = Column(String(20), default="team")
    documents_uploaded_count = Column(Integer(), default=0)
    messages_sent_count = Column(Integer(), default=0)
    storage_used_bytes = Column(Integer(), default=0)
    created_at = Column(DateTime(timezone=True), default=_now)

    owner = relationship("User", back_populates="workspaces_owned")
    members = relationship("WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="workspace")
    chat_sessions = relationship("ChatSession", back_populates="workspace")


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"

    id = Column(String(36), primary_key=True, default=_id)
    workspace_id = Column(String(36), ForeignKey("workspaces.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    role = Column(String(20), default="member")  # owner | member
    joined_at = Column(DateTime(timezone=True), default=_now)

    workspace = relationship("Workspace", back_populates="members")
    user = relationship("User")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=_id)
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    workspace_id = Column(String(36), ForeignKey("workspaces.id"), nullable=True)
    filename = Column(String(500), nullable=False)
    file_type = Column(String(20), nullable=False)  # pdf | docx | txt
    status = Column(String(20), default="processing")  # processing | ready | failed
    page_count = Column(Integer(), default=0)
    size_bytes = Column(Integer(), default=0)
    uploaded_at = Column(DateTime(timezone=True), default=_now)

    owner = relationship("User", back_populates="documents")
    workspace = relationship("Workspace", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = _CHUNK_TABLE_ARGS

    id = Column(String(36), primary_key=True, default=_id)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer(), nullable=False)
    text = Column(Text(), nullable=False)
    token_count = Column(Integer(), default=0)
    # PostgreSQL: pgvector Vector(384). SQLite: Text holding a JSON list.
    # See embedding_column_type() above.
    embedding = Column(
        embedding_column_type(), default=(list if is_postgres_database() else "[]")
    )

    document = relationship("Document", back_populates="chunks")

    def get_embedding(self) -> list:
        """Return the embedding as a plain list of floats, either way
        it is stored (pgvector returns a list already; SQLite JSON
        needs parsing)."""
        value = self.embedding
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        if isinstance(value, str):
            try:
                return [float(v) for v in json.loads(value or "[]")]
            except (json.JSONDecodeError, TypeError, ValueError):
                return []
        try:
            return [float(v) for v in list(value)]
        except TypeError:
            return []

    def set_embedding(self, vector: list) -> None:
        if is_postgres_database():
            self.embedding = [float(v) for v in vector]
        else:
            self.embedding = json.dumps([float(v) for v in vector])


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=_id)
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    workspace_id = Column(String(36), ForeignKey("workspaces.id"), nullable=True)
    title = Column(String(500), default="New chat")
    # JSON list of document ids this session is scoped to (optional).
    scoped_document_ids = Column(Text(), default="[]")
    created_at = Column(DateTime(timezone=True), default=_now)

    owner = relationship("User", back_populates="chat_sessions")
    workspace = relationship("Workspace", back_populates="chat_sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

    def get_scoped_document_ids(self) -> list:
        try:
            return json.loads(self.scoped_document_ids or "[]")
        except json.JSONDecodeError:
            return []

    def set_scoped_document_ids(self, ids: list) -> None:
        self.scoped_document_ids = json.dumps(ids or [])


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=_id)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user | assistant
    content = Column(Text(), nullable=False)
    # JSON list of {"document_id", "chunk_index", "text", "score"}.
    sources = Column(Text(), default="[]")
    tokens_used = Column(Integer(), default=0)
    prompt_tokens = Column(Integer(), default=0)
    completion_tokens = Column(Integer(), default=0)
    created_at = Column(DateTime(timezone=True), default=_now)

    session = relationship("ChatSession", back_populates="messages")

    def get_sources(self) -> list:
        try:
            return json.loads(self.sources or "[]")
        except json.JSONDecodeError:
            return []

    def set_sources(self, sources: list) -> None:
        self.sources = json.dumps(sources or [])
