"""Plan model — Free / Pro / Team (same limits as Week 6).

Limits are the single source of truth for quota checks, for personal
accounts and for shared workspaces (which always use the Team plan).
"""
from datetime import datetime, timedelta, timezone

MB = 1024 * 1024
GB = 1024 * MB
BILLING_PERIOD_DAYS = 30

PLANS = {
    "free": {
        "name": "Free",
        "price_monthly_usd": 0,
        "limits": {"max_documents": 10, "max_messages": 100, "max_storage_bytes": 100 * MB},
    },
    "pro": {
        "name": "Pro",
        "price_monthly_usd": 19,
        "limits": {"max_documents": 500, "max_messages": 5_000, "max_storage_bytes": 10 * GB},
    },
    "team": {
        "name": "Team",
        "price_monthly_usd": 49,
        "limits": {"max_documents": 5_000, "max_messages": 50_000, "max_storage_bytes": 100 * GB},
    },
}


def get_plan(tier: str) -> dict:
    """Return the plan dict for a tier, falling back to Free."""
    return PLANS.get((tier or "free").lower(), PLANS["free"])


def plan_limits(tier: str) -> dict:
    return get_plan(tier)["limits"]


def quota_ok(used: int, limit: int) -> bool:
    """True when one more item is still allowed."""
    return used < limit


def check_quota(kind: str, used: int, tier: str) -> None:
    """Raise a clear 429 error when a plan limit is reached.

    ``kind`` is "document" or "message". Kept as a pure function so unit
    tests can check it without HTTP.
    """
    from fastapi import HTTPException

    limits = plan_limits(tier)
    limit_key = "max_documents" if kind == "document" else "max_messages"
    limit = limits[limit_key]
    if used >= limit:
        plan_name = get_plan(tier)["name"]
        raise HTTPException(
            status_code=429,
            detail=(
                f"{kind.capitalize()} limit reached for your {plan_name} plan "
                f"({used}/{limit} used this billing period). "
                f"Upgrade here: GET /api/billing/plans"
            ),
        )


def reset_usage_counters(obj) -> None:
    """Zero the usage counters on a user or workspace and roll the period.

    Pure logic on the object — the caller commits. Used by the scheduled
    job and by the lazy reset in ``get_usage``-style helpers.
    """
    obj.documents_uploaded_count = 0
    obj.messages_sent_count = 0
    obj.storage_used_bytes = 0
    if hasattr(obj, "current_period_start"):
        now = datetime.now(timezone.utc)
        obj.current_period_start = now
        obj.current_period_end = now + timedelta(days=BILLING_PERIOD_DAYS)


def as_aware(dt):
    """SQLite stores datetimes without tzinfo — treat naive values as UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def reset_due_usage(db) -> int:
    """Reset counters for every user whose billing period has ended.

    Also resets any workspace owned by such a user. Returns how many
    users were reset. Called by the daily scheduler job.
    """
    from .models import User, Workspace

    now = datetime.now(timezone.utc)
    count = 0
    for user in db.query(User).all():
        if user.current_period_end and now >= as_aware(user.current_period_end):
            reset_usage_counters(user)
            for ws in db.query(Workspace).filter(Workspace.owner_id == user.id).all():
                reset_usage_counters(ws)
            count += 1
    if count:
        db.commit()
    return count
