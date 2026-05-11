"""
Stripe billing integration.

If STRIPE_SECRET_KEY is empty (dev mode), checkout returns a mock response
so the rest of the app works without real keys.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend import database as db
from backend.auth_router import get_current_active_user
from backend.config import settings

router = APIRouter(prefix="/api/billing", tags=["billing"])

PLAN_LIMITS = {
    "starter": {"max_cameras": 2,  "max_incidents_month": 2000},
    "pro":     {"max_cameras": 10, "max_incidents_month": 999999},
    "enterprise": {"max_cameras": 999999, "max_incidents_month": 999999},
}


@router.post("/checkout")
async def create_checkout(
    body: dict,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    """
    Create a Stripe Checkout session for upgrading the org plan.
    Returns {checkout_url} — redirect the browser there.
    Dev mode: returns a mock URL when STRIPE_SECRET_KEY is empty.
    """
    plan = (body.get("plan") or "").lower()
    if plan not in ("starter", "pro", "enterprise"):
        raise HTTPException(status_code=422, detail="Plan invalid. Alege: starter, pro, enterprise")

    # Resolve org
    org_obj = await db.get_or_create_default_org(session)
    if current_user.organization_id:
        fetched = await db.get_org_by_id(session, current_user.organization_id)
        if fetched:
            org_obj = fetched

    if not settings.STRIPE_SECRET_KEY:
        # Dev mode — simulate successful checkout redirect
        return {
            "checkout_url": None,
            "dev_mode": True,
            "message": f"Dev mode: Stripe neconfigurat. Activând planul '{plan}' direct...",
            "plan": plan,
        }

    try:
        import stripe
        stripe.api_key = settings.STRIPE_SECRET_KEY

        price_id_map = {
            "starter": settings.STRIPE_PRICE_STARTER,
            "pro": settings.STRIPE_PRICE_PRO,
        }
        price_id = price_id_map.get(plan, "")
        if not price_id:
            raise HTTPException(status_code=422, detail=f"Price ID neconfigurat pentru planul '{plan}'")

        checkout = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{settings.APP_BASE_URL}/app?checkout=success&plan={plan}",
            cancel_url=f"{settings.APP_BASE_URL}/app?checkout=cancelled",
            client_reference_id=str(org_obj.id),
            customer_email=current_user.email,
            metadata={"org_id": str(org_obj.id), "plan": plan},
        )
        return {"checkout_url": checkout.url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare Stripe: {e}")


@router.post("/activate-dev")
async def activate_plan_dev(
    body: dict,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    """Dev-only: activate a plan without Stripe (when STRIPE_SECRET_KEY is empty)."""
    if settings.STRIPE_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Disponibil doar în dev mode")
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar admin")

    plan = (body.get("plan") or "trial").lower()
    if plan not in ("trial", "starter", "pro", "enterprise"):
        raise HTTPException(status_code=422, detail="Plan invalid")

    org = await db.get_or_create_default_org(session)
    if current_user.organization_id:
        fetched = await db.get_org_by_id(session, current_user.organization_id)
        if fetched:
            org = fetched

    org.plan = plan
    limits = PLAN_LIMITS.get(plan, {})
    if limits:
        org.max_cameras = limits["max_cameras"]
        org.max_incidents_month = limits["max_incidents_month"]
    org.subscription_active = True

    await session.commit()
    await session.refresh(org)
    return {"ok": True, "plan": org.plan, "max_cameras": org.max_cameras}


@router.post("/webhook")
async def stripe_webhook(request: Request, session: AsyncSession = Depends(db.get_db)):
    """
    Stripe sends events here after payment.
    Verifies HMAC signature and activates plan on checkout.session.completed.
    """
    if not settings.STRIPE_SECRET_KEY:
        return {"ok": True, "dev_mode": True}

    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig, settings.STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Semnătură Stripe invalidă")

    if event["type"] == "checkout.session.completed":
        checkout_session = event["data"]["object"]
        org_id = int(checkout_session.get("client_reference_id") or 0)
        plan = (checkout_session.get("metadata") or {}).get("plan", "starter")

        if org_id:
            org = await db.get_org_by_id(session, org_id)
            if org:
                org.plan = plan
                limits = PLAN_LIMITS.get(plan, {})
                if limits:
                    org.max_cameras = limits["max_cameras"]
                    org.max_incidents_month = limits["max_incidents_month"]
                org.subscription_active = True
                org.stripe_customer_id = checkout_session.get("customer")
                await session.commit()

    return {"ok": True}
