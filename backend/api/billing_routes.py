"""
Billing API Routes — Stripe Checkout, Customer Portal, and Webhook.

Endpoints:
    POST /v1/billing/checkout  — Create a Stripe Checkout session (returns redirect URL)
    POST /v1/billing/portal    — Create a Stripe Customer Portal session
    POST /v1/billing/webhook   — Stripe webhook receiver (verified by signature)
    GET  /v1/billing/usage     — Current period API usage for the authenticated tenant
    GET  /v1/billing/plans     — Public plan definitions (no auth required)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.data.auth import TenantContext, resolve_tenant_context
from backend.data.billing import (
    DAILY_LIMITS,
    create_checkout_session,
    create_portal_session,
    handle_webhook,
    is_available,
    process_webhook_event,
)
from backend.data.database import Tenant, get_db
from backend.data.usage import summarize_usage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/billing", tags=["billing"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CheckoutRequest(BaseModel):
    plan: str  # "pro" or "enterprise"
    success_url: str
    cancel_url: str
    email: str | None = None


class CheckoutResponse(BaseModel):
    checkout_url: str
    plan: str


class PortalRequest(BaseModel):
    return_url: str


class PortalResponse(BaseModel):
    portal_url: str


class PlanDefinition(BaseModel):
    name: str
    plan_tier: str
    price_monthly_usd: int | None
    daily_request_limit: int | None
    serving_tier: str
    support: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/plans", response_model=list[PlanDefinition], summary="List available plans")
async def list_plans() -> list[PlanDefinition]:
    """
    Returns public plan definitions. No authentication required.
    Used by the pricing page frontend component.
    """
    return [
        PlanDefinition(
            name="Free",
            plan_tier="free",
            price_monthly_usd=0,
            daily_request_limit=100,
            serving_tier="Tier 3 — FAISS + TF-IDF",
            support="Community",
        ),
        PlanDefinition(
            name="Pro",
            plan_tier="pro",
            price_monthly_usd=299,
            daily_request_limit=10_000,
            serving_tier="Tier 2 — ONNX Ensemble (200–800 ms)",
            support="Email (48 h SLA)",
        ),
        PlanDefinition(
            name="Enterprise",
            plan_tier="enterprise",
            price_monthly_usd=None,  # contact sales
            daily_request_limit=None,
            serving_tier="Tier 1 — GPU Ensemble (50–200 ms)",
            support="Dedicated + 4 h SLA",
        ),
    ]


@router.post("/checkout", response_model=CheckoutResponse, summary="Create Stripe Checkout session")
async def billing_checkout(
    body: CheckoutRequest,
    context: TenantContext = Depends(resolve_tenant_context),
    db: Session = Depends(get_db),
) -> CheckoutResponse:
    """
    Creates a Stripe Checkout session for the given plan.
    Returns a URL that the frontend redirects the user to.

    Requires a valid API key (X-Nova-API-Key header).
    """
    if not is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Billing is not configured on this deployment.",
        )
    if body.plan not in ("pro", "enterprise"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown plan '{body.plan}'. Valid values: pro, enterprise.",
        )

    # Fetch the tenant email to pre-fill checkout if not provided
    email = body.email
    if not email:
        tenant = db.query(Tenant).filter_by(tenant_id=context.tenant_id).first()
        if tenant:
            # Try to get email from associated user records
            from backend.data.database import User

            user = db.query(User).filter_by(tenant_id=context.tenant_id).first()
            if user and user.email:
                email = user.email

    try:
        url = create_checkout_session(
            tenant_id=context.tenant_id,
            plan=body.plan,
            success_url=body.success_url,
            cancel_url=body.cancel_url,
            customer_email=email,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return CheckoutResponse(checkout_url=url, plan=body.plan)


@router.post("/portal", response_model=PortalResponse, summary="Open Stripe Customer Portal")
async def billing_portal(
    body: PortalRequest,
    context: TenantContext = Depends(resolve_tenant_context),
    db: Session = Depends(get_db),
) -> PortalResponse:
    """
    Creates a Stripe Customer Portal session for self-serve plan management
    (upgrade, downgrade, cancel, update payment method).

    Requires a valid API key for an authenticated tenant that has an active
    Stripe subscription.
    """
    if not is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Billing is not configured on this deployment.",
        )

    tenant = db.query(Tenant).filter_by(tenant_id=context.tenant_id).first()
    if not tenant or not tenant.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Stripe customer record found for this tenant. "
            "Start a subscription first via /v1/billing/checkout.",
        )

    try:
        url = create_portal_session(
            stripe_customer_id=tenant.stripe_customer_id,
            return_url=body.return_url,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return PortalResponse(portal_url=url)


@router.post("/webhook", summary="Stripe webhook receiver", include_in_schema=False)
async def billing_webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None, alias="stripe-signature"),
    db: Session = Depends(get_db),
) -> dict:
    """
    Receives and verifies Stripe webhook events.

    This endpoint has no API key auth — it is verified by the Stripe-Signature
    header using the STRIPE_WEBHOOK_SECRET. Stripe retries on non-2xx responses.

    Configure this URL in your Stripe Dashboard:
        https://dashboard.stripe.com/webhooks → Add endpoint → /v1/billing/webhook
    """
    if not is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Billing not configured.",
        )
    if not stripe_signature:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Stripe-Signature header.",
        )

    payload = await request.body()

    try:
        event = handle_webhook(payload, stripe_signature)
    except Exception as exc:
        logger.warning("Stripe webhook signature verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Webhook signature verification failed: {exc}",
        ) from exc

    result = process_webhook_event(event, db)
    return {"status": "ok", **result}


@router.get("/usage", summary="Current period API usage")
async def billing_usage(
    context: TenantContext = Depends(resolve_tenant_context),
) -> dict:
    """
    Returns API usage summary for the authenticated tenant in the current period.
    Includes daily limit, current consumption, and upgrade prompt if near limit.
    """
    summary = summarize_usage()
    daily_limit = DAILY_LIMITS.get(context.plan)

    # Filter summary to this tenant only
    tenant_key = f"{context.tenant_id}:{context.catalog_id}"
    tenant_requests = summary.get("tenant_counts", {}).get(tenant_key, 0)

    response: dict = {
        "tenant_id": context.tenant_id,
        "plan": context.plan,
        "daily_limit": daily_limit,
        "requests_today": tenant_requests,
        "limit_remaining": (daily_limit - tenant_requests) if daily_limit is not None else None,
        "upgrade_url": None,
    }

    # Surface upgrade prompt when at 80%+ of daily limit
    if daily_limit is not None and tenant_requests >= daily_limit * 0.8:
        response["upgrade_url"] = "/v1/billing/checkout"
        response["upgrade_message"] = (
            f"You've used {tenant_requests}/{daily_limit} requests today. Upgrade to Pro for 10,000 requests/day."
        )

    return response
