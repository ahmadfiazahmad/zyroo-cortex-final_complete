"""Stripe webhooks — Week 6: signature-verified, 3 events handled."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import User
from ..plans import reset_usage_counters
from .billing import stripe

router = APIRouter(prefix="/api/webhooks", tags=["Billing"])


def _find_user_by_customer(db: Session, customer_id: str):
    return db.query(User).filter(User.stripe_customer_id == customer_id).first()


def _handle_checkout_completed(db: Session, obj: dict):
    meta = obj.get("metadata") or {}
    user = db.query(User).filter(User.id == meta.get("user_id", "")).first()
    if not user:
        return
    user.stripe_customer_id = obj.get("customer")
    user.subscription_id = obj.get("subscription")
    user.subscription_status = "active"
    user.plan_tier = meta.get("plan", "pro")
    reset_usage_counters(user)  # new paid billing period starts now
    db.commit()


def _handle_subscription_updated(db: Session, obj: dict):
    user = _find_user_by_customer(db, obj.get("customer", ""))
    if not user:
        return
    status = obj.get("status", "")
    user.subscription_status = status
    period_end = obj.get("current_period_end")
    if period_end:
        user.current_period_end = datetime.fromtimestamp(period_end, tz=timezone.utc)
    if status == "canceled":
        user.plan_tier = "free"
        user.subscription_id = None
        reset_usage_counters(user)
    db.commit()


def _handle_subscription_deleted(db: Session, obj: dict):
    user = _find_user_by_customer(db, obj.get("customer", ""))
    if not user:
        return
    user.plan_tier = "free"
    user.subscription_id = None
    user.subscription_status = "canceled"
    reset_usage_counters(user)
    db.commit()


@router.post("/stripe", summary="Stripe webhook (signature verified)")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Rejects missing/invalid Stripe-Signature headers with 400."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    if stripe is None or not settings.stripe_webhook_secret:
        raise HTTPException(
            status_code=503,
            detail="Stripe webhook not configured. Set STRIPE_SECRET_KEY and "
            "STRIPE_WEBHOOK_SECRET in .env (see README).",
        )
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid webhook signature: {e}")

    event_type = event["type"]
    obj = event["data"]["object"]
    if event_type == "checkout.session.completed":
        _handle_checkout_completed(db, obj)
    elif event_type == "customer.subscription.updated":
        _handle_subscription_updated(db, obj)
    elif event_type == "customer.subscription.deleted":
        _handle_subscription_deleted(db, obj)
    # Other events: acknowledged but ignored.
    return {"received": True, "type": event_type}
