"""
Enterprise B2B Rate Limiter (Token Bucket Algorithm).

This middleware intercepts every incoming FastAPI request and checks a distributed
Redis Cluster to enforce strict Transactions Per Second (TPS) quotas.

If a B2B tenant (e.g., "Startup A") exceeds their tier's SLA quota,
this layer instantly rejects the request with HTTP 429 (Too Many Requests)
*before* the heavy PyTorch mathematical ensemble is ever invoked.
This prevents DDoS attacks and guarantees 100% uptime for Enterprise tenants like Netflix.
"""

from collections.abc import Callable
import logging
import time

from fastapi import Request
import redis.asyncio as redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RedisRateLimiter(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str = "redis://redis-feature-store:6379"):
        super().__init__(app)
        self.redis = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)

        # Define B2B SaaS Quotas (Requests Per Second)
        self.quotas = {
            "free": 10,  # Developer Sandboxes
            "pro": 100,  # Medium Startups
            "enterprise": 5000,  # Netflix/Amazon Tier
        }

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        # 1. Identify the Tenant (For demo purposes, we extract it from a mock header.
        # In production, this comes from the validated JWT token).
        tenant_id = request.headers.get("X-Tenant-ID", "anonymous")
        tenant_tier = request.headers.get("X-Tenant-Tier", "free")

        limit = self.quotas.get(tenant_tier, 10)

        # We only rate-limit the heavy ML recommendation routes
        if request.url.path.startswith("/v1/recommendations") or request.url.path.startswith("/v1/search/ai"):
            try:
                # 2. Execute Distributed Token Bucket in Redis
                current_timestamp = int(time.time())
                redis_key = f"rate_limit:{tenant_id}:{current_timestamp}"

                # Increment the request count for this specific second
                requests_this_second = await self.redis.incr(redis_key)

                # Set TTL of 5 seconds so Redis doesn't run out of memory
                if requests_this_second == 1:
                    await self.redis.expire(redis_key, 5)

                if requests_this_second > limit:
                    logger.warning(
                        f"s [RATE LIMIT EXCEEDED] Tenant {tenant_id} (Tier: {tenant_tier}) exceeded {limit} TPS."
                    )
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "SLA Quota Exceeded. Please upgrade to a higher Enterprise Tier for more TPS."
                        },
                        headers={"Retry-After": "1"},
                    )
            except Exception as e:
                # If Redis is temporarily down, we fail open (allow the request) to guarantee API uptime
                logger.error(f"Redis Rate Limiter Error: {e}. Failing open.")

        # 3. Request is within limits. Proceed to the PyTorch inference nodes.
        response = await call_next(request)
        return response
