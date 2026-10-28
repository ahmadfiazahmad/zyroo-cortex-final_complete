"""Billing — Week 6: plans, checkout, me, cancel."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..plans import PLANS, get_plan
from ..schemas import BillingMeOut, CheckoutRequest, PlanOut

router = APIRouter(prefix="/api/billing", tags=["Billing"])

stripe = None
if settings.stripe_secret_key:
    try:
        import stripe as _stripe

        _stripe.api_key = settings.stripe_secret_key
        stripe = _stripe
    except ImportError:
        stripe = None


@router.get("/plans", response_model=dict[str, PlanOut], summary="List all plans")
def list_plans():
    """Free / Pro / Team with prices and limits."""
    return {
        key: PlanOut(name=p["name"], price_monthly_usd=p["price_monthly_usd"], limits=p["limits"])
        for key, p in PLANS.items()
    }


@router.get("/me", response_model=BillingMeOut, summary="My plan, usage and limits")
def billing_me(current_user: User = Depends(get_current_user)):
    plan = get_plan(current_user.plan_tier)
    return BillingMeOut(
        user_id=current_user.id,
        plan=current_user.plan_tier,
        plan_name=plan["name"],
        subscription_status=current_user.subscription_status,
        current_period_end=current_user.current_period_end,
        usage={
            "documents": current_user.documents_uploaded_count,
            "messages": current_user.messages_sent_count,
            "storage_bytes": current_user.storage_used_bytes,
        },
        limits=plan["limits"],
    )


@router.post("/checkout", summary="Create a Stripe Checkout session to upgrade")
def create_checkout(data: CheckoutRequest, current_user: User = Depends(get_current_user)):
    plan_key = data.plan.lower()
    if plan_key not in PLANS or plan_key == "free":
        raise HTTPException(status_code=400, detail="Choose 'pro' or 'team'.")
    if stripe is None:
        raise HTTPException(
            status_code=503,
            detail="Stripe is not configured. Set STRIPE_SECRET_KEY in .env (see README).",
        )
    price_id = {"pro": settings.stripe_price_pro, "team": settings.stripe_price_team}[plan_key]
    if not price_id:
        raise HTTPException(
            status_code=503,
            detail=f"No Stripe price configured for '{plan_key}'. "
            f"Set STRIPE_PRICE_{plan_key.upper()} in .env (see README).",
        )
    success_url = data.success_url or (
        f"{settings.app_base_url.rstrip('/')}/billing/success?session_id={{CHECKOUT_SESSION_ID}}"
    )
    cancel_url = data.cancel_url or f"{settings.app_base_url.rstrip('/')}/billing/cancelled"
    kwargs = {
        "mode": "subscription",
        "line_items": [{"price": price_id, "quantity": 1}],
        "success_url": success_url,
        "cancel_url": cancel_url,
        "metadata": {"user_id": current_user.id, "plan": plan_key},
    }
    if current_user.stripe_customer_id:
        kwargs["customer"] = current_user.stripe_customer_id
    else:
        kwargs["customer_email"] = current_user.email
    try:
        session = stripe.checkout.Session.create(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Stripe error: {e}")
    return {"checkout_url": session.url, "session_id": session.id}


@router.post("/cancel", summary="Cancel my subscription at period end")
def cancel_subscription(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if current_user.plan_tier == "free":
        raise HTTPException(status_code=400, detail="You are on the Free plan — nothing to cancel.")
    if stripe is not None and current_user.subscription_id:
        try:
            stripe.Subscription.modify(current_user.subscription_id, cancel_at_period_end=True)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Stripe error: {e}")
    current_user.subscription_status = "cancelling"
    db.commit()
    return {
        "message": "Your subscription will cancel at the end of the billing period.",
        "subscription_status": current_user.subscription_status,
        "current_period_end": current_user.current_period_end.isoformat()
        if current_user.current_period_end
        else None,
    }
