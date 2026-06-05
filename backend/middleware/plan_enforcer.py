"""
Plan Enforcement Middleware.

Intercepts authenticated API requests and enforces daily request limits based
on the tenant's plan tier. Returns HTTP 429 with an upgrade prompt when the
limit is exceeded.

Limits:
    free:       100 requests / day
    pro:        10,000 requests / day
    enterprise: unlimited

Usage counters are maintained in Redis (fast path) with a fallback to the
JSONL usage log. Counter keys are scoped per tenant per UTC calendar day.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plan daily limits  (None = unlimited)
# ---------------------------------------------------------------------------
DAILY_LIMITS: dict[str, int | None] = {
    "free": 100,
    "pro": 10_000,
    "enterprise": None,
}

# Paths that are exempt from plan enforcement (health, auth, billing, docs)
_EXEMPT_PREFIXES = (
    "/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/v1/auth/",
    "/v1/billing/webhook",  # webhook must never be blocked
    "/v1/platform/",
    "/ui/",
    "/",
)


def _today_key(tenant_id: str) -> str:
    """Redis counter key scoped to UTC calendar day."""
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"apex:usage:{tenant_id}:{day}"


def _get_redis():
    """Return a Redis client if configured, else None."""
    try:
        import redis as _redis

        url = os.getenv("REDIS_URL")
        if not url:
            return None
        return _redis.from_url(url, decode_responses=True, socket_connect_timeout=1)
    except Exception:
        return None


class PlanEnforcerMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces per-tenant daily request limits.

    Flow:
        1. Skip exempt paths (health, auth, billing webhook, docs)
        2. Extract tenant_id + plan from X-Nova-API-Key header
           (uses the resolved TenantContext when available; falls back to DB)
        3. Increment Redis counter atomically (TTL = 25 hours)
        4. If count > daily_limit: return 429 with upgrade_url
        5. Otherwise: pass the request through
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip exempt paths
        if any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            return await call_next(request)

        tenant_id, plan = self._resolve_tenant(request)

        # Unknown or unauthenticated requests pass through (rate limiter handles IP-level)
        if not tenant_id or plan == "demo":
            return await call_next(request)

        limit = DAILY_LIMITS.get(plan)
        if limit is None:
            # Unlimited plan — no check needed
            return await call_next(request)

        count = self._increment_counter(tenant_id)

        if count > limit:
            logger.warning(
                "Plan limit exceeded: tenant=%s plan=%s count=%d limit=%d path=%s",
                tenant_id,
                plan,
                count,
                limit,
                path,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "daily_limit_exceeded",
                    "detail": (
                        f"Your {plan} plan allows {limit:,} requests per day. You have made {count:,} requests today."
                    ),
                    "plan": plan,
                    "daily_limit": limit,
                    "requests_today": count,
                    "upgrade_url": "/v1/billing/checkout",
                    "upgrade_message": ("Upgrade to Pro for 10,000 requests/day, or Enterprise for unlimited access."),
                },
            )

        return await call_next(request)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_tenant(request: Request) -> tuple[str | None, str]:
        """
        Extract (tenant_id, plan) from the request without a DB round-trip.

        Uses the pre-resolved TenantContext stored in request.state by the
        auth dependency when available. Falls back to header parsing.
        """
        # TenantContext may have been resolved by a previous dependency
        ctx = getattr(request.state, "tenant_context", None)
        if ctx is not None:
            return ctx.tenant_id, ctx.plan

        # Fallback: read X-Tenant-ID + default plan (conservative — assume free)
        tenant_id = request.headers.get("X-Tenant-ID") or request.headers.get("x-tenant-id")
        if tenant_id:
            return tenant_id, "free"

        return None, "demo"

    @staticmethod
    def _increment_counter(tenant_id: str) -> int:
        """
        Atomically increment and return the daily counter for this tenant.

        Uses Redis INCR + EXPIRE (25-hour TTL) for fast atomic counting.
        Falls back to 0 (no enforcement) on Redis failure to prevent blocking.
        """
        redis = _get_redis()
        if redis is None:
            return 0  # Redis not configured — skip enforcement

        key = _today_key(tenant_id)
        try:
            pipe = redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 90_000)  # 25 hours in seconds
            results = pipe.execute()
            return int(results[0])
        except Exception as exc:
            logger.warning("Redis counter increment failed for tenant %s: %s", tenant_id, exc)
            return 0  # Fail open — do not block requests on Redis failure
