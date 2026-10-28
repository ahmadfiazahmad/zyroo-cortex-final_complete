"""Admin — Weeks 7-8: users, workspaces, suspend, analytics."""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import require_admin
from ..llm import estimate_cost_usd
from ..plans import as_aware
from ..models import ChatSession, Document, Message, User, Workspace, WorkspaceMember
from ..schemas import AdminUserOut, AnalyticsOut, SuspendRequest, WorkspaceOut
from .workspaces import _workspace_out

router = APIRouter(prefix="/api/admin", tags=["Admin"])


@router.get("/users", response_model=list[AdminUserOut], summary="List all users (admin)")
def list_users(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """All registered users (never password hashes) with plan, usage,
    suspension status and creation date. Paginated with skip/limit."""
    return db.query(User).order_by(User.created_at.desc()).offset(skip).limit(limit).all()


@router.get("/workspaces", response_model=list[WorkspaceOut], summary="List all workspaces (admin)")
def list_all_workspaces(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    workspaces = db.query(Workspace).order_by(Workspace.created_at.desc()).offset(skip).limit(limit).all()
    return [_workspace_out(db, w) for w in workspaces]


@router.patch("/users/{user_id}/suspend", response_model=AdminUserOut, summary="Suspend or reactivate a user")
def suspend_user(
    user_id: str,
    data: SuspendRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Set is_suspended (or toggle it when omitted). Suspended users are
    immediately rejected with 403 by get_current_user everywhere."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_suspended = not user.is_suspended if data.is_suspended is None else data.is_suspended
    db.commit()
    db.refresh(user)
    return user


@router.get("/analytics", response_model=AnalyticsOut, summary="Platform analytics (admin)")
def analytics(db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    """Last-30-days summary: daily active users, messages per day, plan
    distribution, subscription counts, and LLM token usage / cost."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=30)
    days = [(start + timedelta(days=i)).date() for i in range(31)]
    day_keys = [d.isoformat() for d in days]

    # Daily active users: distinct users who sent a message or uploaded
    # a document that day.
    daily_active = {k: 0 for k in day_keys}
    active_sets = {k: set() for k in day_keys}
    for msg in db.query(Message).join(ChatSession).all():
        if msg.created_at and as_aware(msg.created_at) >= start:
            key = as_aware(msg.created_at).date().isoformat()
            if key in active_sets and msg.role == "user":
                active_sets[key].add(msg.session.owner_id)
    for doc in db.query(Document).all():
        if doc.uploaded_at and as_aware(doc.uploaded_at) >= start:
            key = as_aware(doc.uploaded_at).date().isoformat()
            if key in active_sets:
                active_sets[key].add(doc.owner_id)
    for k, users in active_sets.items():
        daily_active[k] = len(users)

    messages_per_day = {k: 0 for k in day_keys}
    for msg in db.query(Message).filter(Message.role == "user").all():
        if msg.created_at and as_aware(msg.created_at) >= start:
            key = as_aware(msg.created_at).date().isoformat()
            if key in messages_per_day:
                messages_per_day[key] += 1

    plan_distribution = {"Free": 0, "Pro": 0, "Team": 0}
    for (tier,) in db.query(User.plan_tier).all():
        plan_distribution[{"free": "Free", "pro": "Pro", "team": "Team"}.get(tier, "Free")] += 1

    # Paid-plan users still active vs. users whose subscription was cancelled
    # (a cancelled user is downgraded to Free, so plan alone can't tell).
    paid_active = (
        db.query(func.count(User.id))
        .filter(User.plan_tier != "free", User.subscription_status != "canceled")
        .scalar()
        or 0
    )
    cancelled = db.query(func.count(User.id)).filter(User.subscription_status == "canceled").scalar() or 0
    subscriptions = {"active": int(paid_active), "cancelled": int(cancelled)}

    total_prompt = db.query(func.coalesce(func.sum(Message.prompt_tokens), 0)).scalar() or 0
    total_completion = db.query(func.coalesce(func.sum(Message.completion_tokens), 0)).scalar() or 0
    total_requests = db.query(func.count(Message.id)).filter(Message.role == "assistant").scalar() or 0
    total_tokens = int(total_prompt) + int(total_completion)

    return AnalyticsOut(
        daily_active_users=daily_active,
        messages_per_day=messages_per_day,
        plan_distribution=plan_distribution,
        subscriptions=subscriptions,
        llm_usage_monitoring={
            "total_requests": int(total_requests),
            "total_prompt_tokens": int(total_prompt),
            "total_completion_tokens": int(total_completion),
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimate_cost_usd(total_tokens),
        },
    )
