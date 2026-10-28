"""Scheduled jobs for serverless deployments (Vercel Cron).

Vercel has no continuously running process, so the APScheduler job in
app/main.py does not run there. Instead Vercel Cron calls this endpoint
once a day (see vercel.json), sending
``Authorization: Bearer <CRON_SECRET>``. The job itself is idempotent
(reset only users whose billing period has ended), so a duplicate or
missed run is safe — get_usage() also resets lazily on every request.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..plans import reset_due_usage

router = APIRouter(prefix="/api/cron", tags=["Cron"])


@router.get("/usage-reset", summary="Daily usage reset (Vercel Cron)")
def usage_reset(request: Request, db: Session = Depends(get_db)):
    """Reset usage counters for users whose billing period has ended.

    Protected by CRON_SECRET. Vercel Cron sends it automatically when
    the secret is set in the Vercel dashboard.
    """
    if not settings.cron_secret:
        raise HTTPException(
            status_code=503, detail="CRON_SECRET is not configured on the server"
        )
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {settings.cron_secret}":
        raise HTTPException(status_code=401, detail="Invalid cron secret")
    reset_count = reset_due_usage(db)
    return {"reset_users": reset_count}
