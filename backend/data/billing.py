"""
Stripe Billing Integration.

Handles plan checkout, customer portal, webhook processing, and plan-tier
synchronisation back to the PostgreSQL tenant table.

Required environment variables (set in Render dashboard — never committed):
    STRIPE_SECRET_KEY      — Stripe secret key (sk_live_... or sk_test_...)
    STRIPE_WEBHOOK_SECRET  — Webhook signing secret (whsec_...)
    STRIPE_PRICE_PRO       — Stripe Price ID for the Pro plan
    STRIPE_PRICE_ENT       — Stripe Price ID for the Enterprise plan
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stripe client — gracefully absent when not configured
# ---------------------------------------------------------------------------
try:
    import stripe as _stripe

    _stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
    _STRIPE_AVAILABLE = bool(_stripe.api_key)
    if not _STRIPE_AVAILABLE:
        logger.warning("STRIPE_SECRET_KEY not set. Billing endpoints will return 503.")
except ImportError:
    _stripe = None  # type: ignore[assignment]
    _STRIPE_AVAILABLE = False
    logger.warning("stripe package not installed. Run: pip install stripe")

# ---------------------------------------------------------------------------
# Price ID mapping  (Free has no Stripe price — it's the default plan)
# ---------------------------------------------------------------------------
PRICE_IDS: dict[str, str | None] = {
    "pro": os.getenv("STRIPE_PRICE_PRO"),
    "enterprise": os.getenv("STRIPE_PRICE_ENT"),
}

# Daily request limits per plan tier
DAILY_LIMITS: dict[str, int | None] = {
    "free": 100,
    "pro": 10_000,
    "enterprise": None,  # unlimited
}

# Plan tier label returned by Stripe metadata → our internal plan_tier value
STRIPE_PLAN_MAP: dict[str, str] = {
    "pro": "pro",
    "enterprise": "enterprise",
}


# ---------------------------------------------------------------------------
# Checkout & portal helpers
# ---------------------------------------------------------------------------


def create_checkout_session(
    tenant_id: str,
    plan: str,
    success_url: str,
    cancel_url: str,
    customer_email: str | None = None,
) -> str:
    """
    Create a Stripe Checkout Session and return the redirect URL.

    Args:
        tenant_id:      Internal tenant UUID — stored in Stripe metadata for
                        webhook reconciliation.
        plan:           One of "pro" or "enterprise".
        success_url:    Where Stripe redirects after successful payment.
        cancel_url:     Where Stripe redirects on cancellation.
        customer_email: Pre-fill the checkout email field (optional).

    Returns:
        Stripe Checkout Session URL (redirect the user here).

    Raises:
        ValueError:  If plan is unknown or price ID is not configured.
        RuntimeError: If Stripe is not available.
    """
    _require_stripe()
    price_id = PRICE_IDS.get(plan)
    if not price_id:
        raise ValueError(f"Unknown plan '{plan}' or STRIPE_PRICE_{plan.upper()} not configured.")

    params: dict = {
        "mode": "subscription",
        "line_items": [{"price": price_id, "quantity": 1}],
        "metadata": {"tenant_id": tenant_id, "plan": plan},
        "success_url": success_url,
        "cancel_url": cancel_url,
    }
    if customer_email:
        params["customer_email"] = customer_email

    session = _stripe.checkout.Session.create(**params)  # type: ignore[union-attr]
    logger.info("Created Stripe Checkout session %s for tenant %s (plan=%s)", session.id, tenant_id, plan)
    return session.url  # type: ignore[return-value]


def create_portal_session(stripe_customer_id: str, return_url: str) -> str:
    """
    Create a Stripe Customer Portal session for self-serve plan management.

    Args:
        stripe_customer_id: Stripe customer ID (cus_...) stored on the tenant record.
        return_url:         Where the portal redirects when the customer is done.

    Returns:
        Stripe Customer Portal URL.

    Raises:
        RuntimeError: If Stripe is not available.
    """
    _require_stripe()
    session = _stripe.billing_portal.Session.create(  # type: ignore[union-attr]
        customer=stripe_customer_id,
        return_url=return_url,
    )
    logger.info("Created Stripe Portal session for customer %s", stripe_customer_id)
    return session.url  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Webhook processing
# ---------------------------------------------------------------------------


def handle_webhook(payload: bytes, sig_header: str) -> dict:
    """
    Verify and parse a Stripe webhook payload.

    Args:
        payload:    Raw request body bytes.
        sig_header: Value of the `Stripe-Signature` HTTP header.

    Returns:
        Parsed Stripe event dict.

    Raises:
        stripe.error.SignatureVerificationError: On invalid signature.
        RuntimeError: If Stripe is not available.
    """
    _require_stripe()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    if not webhook_secret:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET is not configured.")

    event = _stripe.Webhook.construct_event(payload, sig_header, webhook_secret)  # type: ignore[union-attr]
    return dict(event)


def process_webhook_event(event: dict, db) -> dict:
    """
    Handle a verified Stripe webhook event and sync plan state to Postgres.

    Supported event types:
        checkout.session.completed    — link Stripe customer ID to tenant
        invoice.paid                  — upgrade/maintain plan tier
        customer.subscription.deleted — downgrade to free
        customer.subscription.updated — sync plan tier

    Args:
        event: Parsed Stripe event dict (from handle_webhook).
        db:    SQLAlchemy Session (FastAPI dependency).

    Returns:
        Dict with `{"handled": bool, "event_type": str, "tenant_id": str | None}`.
    """
    from backend.data.database import Tenant  # local import avoids circular dep at module load

    event_type: str = event.get("type", "")
    data_obj: dict = event.get("data", {}).get("object", {})
    tenant_id: str | None = None

    try:
        if event_type == "checkout.session.completed":
            tenant_id = data_obj.get("metadata", {}).get("tenant_id")
            stripe_customer_id = data_obj.get("customer")
            plan = data_obj.get("metadata", {}).get("plan", "pro")

            if tenant_id and stripe_customer_id:
                tenant = db.query(Tenant).filter_by(tenant_id=tenant_id).first()
                if tenant:
                    tenant.stripe_customer_id = stripe_customer_id
                    tenant.plan_tier = STRIPE_PLAN_MAP.get(plan, "pro")
                    db.commit()
                    logger.info(
                        "Linked stripe_customer=%s to tenant=%s, plan=%s",
                        stripe_customer_id,
                        tenant_id,
                        plan,
                    )

        elif event_type == "invoice.paid":
            stripe_customer_id = data_obj.get("customer")
            if stripe_customer_id:
                tenant = db.query(Tenant).filter_by(stripe_customer_id=stripe_customer_id).first()
                if tenant and tenant.plan_tier == "free":
                    # Reactivate if they were downgraded and have paid again
                    tenant.plan_tier = "pro"
                    tenant.is_active = True
                    db.commit()
                    logger.info("Reactivated tenant %s after invoice.paid", tenant.tenant_id)
                    tenant_id = str(tenant.tenant_id)

        elif event_type == "customer.subscription.deleted":
            stripe_customer_id = data_obj.get("customer")
            if stripe_customer_id:
                tenant = db.query(Tenant).filter_by(stripe_customer_id=stripe_customer_id).first()
                if tenant:
                    tenant.plan_tier = "free"
                    db.commit()
                    logger.info(
                        "Downgraded tenant %s to free after subscription cancellation",
                        tenant.tenant_id,
                    )
                    tenant_id = str(tenant.tenant_id)

        elif event_type == "customer.subscription.updated":
            stripe_customer_id = data_obj.get("customer")
            # Determine the new plan from the subscription's price ID
            items = data_obj.get("items", {}).get("data", [])
            new_price_id = items[0].get("price", {}).get("id") if items else None
            new_plan = _price_id_to_plan(new_price_id)

            if stripe_customer_id and new_plan:
                tenant = db.query(Tenant).filter_by(stripe_customer_id=stripe_customer_id).first()
                if tenant:
                    tenant.plan_tier = new_plan
                    db.commit()
                    logger.info("Updated tenant %s plan to %s", tenant.tenant_id, new_plan)
                    tenant_id = str(tenant.tenant_id)

        else:
            logger.debug("Unhandled Stripe event type: %s", event_type)
            return {"handled": False, "event_type": event_type, "tenant_id": None}

    except Exception as exc:
        logger.error("Error processing Stripe webhook %s: %s", event_type, exc)
        db.rollback()
        raise

    return {"handled": True, "event_type": event_type, "tenant_id": tenant_id}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _price_id_to_plan(price_id: str | None) -> str | None:
    """Reverse-map a Stripe Price ID to an internal plan tier string."""
    if not price_id:
        return None
    for plan, pid in PRICE_IDS.items():
        if pid and pid == price_id:
            return plan
    return None


def _require_stripe() -> None:
    """Raise RuntimeError if Stripe is not available."""
    if not _STRIPE_AVAILABLE:
        raise RuntimeError("Stripe is not available. Install stripe (`pip install stripe`) and set STRIPE_SECRET_KEY.")


def is_available() -> bool:
    """Return True if Stripe is configured and the client is available."""
    return _STRIPE_AVAILABLE
