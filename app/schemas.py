"""Pydantic schemas — request validation + response models for every route."""
from datetime import datetime
from typing import List, Optional

import json

from pydantic import BaseModel, EmailStr, Field, field_validator


def _parse_json_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v or "[]")
        except json.JSONDecodeError:
            return []
    return v or []

from .config import settings
from .security import sanitize_text


class _Sanitized(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def _strip_control(cls, v):
        return sanitize_text(v) if isinstance(v, str) else v


# ---------------------------------------------------------------- Auth
class RegisterRequest(_Sanitized):
    name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(_Sanitized):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    role: str
    is_suspended: bool
    plan_tier: str
    subscription_status: str
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ---------------------------------------------------------------- Documents / search
class DocumentOut(BaseModel):
    id: str
    owner_id: str
    workspace_id: Optional[str] = None
    filename: str
    file_type: str
    status: str
    page_count: int
    size_bytes: int
    uploaded_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class SearchRequest(_Sanitized):
    query: str = Field(..., min_length=1, max_length=1000)
    document_id: Optional[str] = None
    workspace_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


# ---------------------------------------------------------------- Chat
class CreateSessionRequest(_Sanitized):
    title: str = Field(default="New chat", max_length=500)
    document_ids: Optional[List[str]] = None
    workspace_id: Optional[str] = None


class SessionOut(BaseModel):
    id: str
    owner_id: str
    workspace_id: Optional[str] = None
    title: str
    scoped_document_ids: List[str] = []
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}

    @field_validator("scoped_document_ids", mode="before")
    @classmethod
    def _parse_ids(cls, v):
        return _parse_json_list(v)


class SendMessageRequest(_Sanitized):
    content: str = Field(..., min_length=1, max_length=settings.chat_max_chars)


class MessageOut(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    sources: List[dict] = []
    tokens_used: int = 0
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}

    @field_validator("sources", mode="before")
    @classmethod
    def _parse_sources(cls, v):
        return _parse_json_list(v)


# ---------------------------------------------------------------- Billing
class CheckoutRequest(_Sanitized):
    plan: str = Field(..., description="'pro' or 'team'")
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


class PlanOut(BaseModel):
    name: str
    price_monthly_usd: int
    limits: dict


class BillingMeOut(BaseModel):
    user_id: str
    plan: str
    plan_name: str
    subscription_status: str
    current_period_end: Optional[datetime] = None
    usage: dict
    limits: dict


# ---------------------------------------------------------------- Workspaces
class CreateWorkspaceRequest(_Sanitized):
    name: str = Field(..., min_length=1, max_length=200)


class InviteRequest(_Sanitized):
    email: EmailStr


class WorkspaceMemberOut(BaseModel):
    user_id: str
    name: str
    email: str
    role: str
    joined_at: Optional[datetime] = None


class WorkspaceOut(BaseModel):
    id: str
    name: str
    owner_id: str
    plan_tier: str
    documents_uploaded_count: int
    messages_sent_count: int
    storage_used_bytes: int
    created_at: Optional[datetime] = None
    members: List[WorkspaceMemberOut] = []

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------- Admin
class SuspendRequest(BaseModel):
    is_suspended: Optional[bool] = None  # None = toggle


class AdminUserOut(BaseModel):
    id: str
    name: str
    email: str
    role: str
    plan_tier: str
    is_suspended: bool
    subscription_status: str
    documents_uploaded_count: int
    messages_sent_count: int
    storage_used_bytes: int
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AnalyticsOut(BaseModel):
    daily_active_users: dict
    messages_per_day: dict
    plan_distribution: dict
    subscriptions: dict
    llm_usage_monitoring: dict
