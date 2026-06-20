import sys
import types

import pytest
from starlette.requests import Request
from starlette.responses import PlainTextResponse


def _request(path="/v1/recommendations/id/100", headers=None, client=("203.0.113.10", 12345)):
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "scheme": "http",
            "server": ("testserver", 80),
            "client": client,
            "headers": [
                (name.lower().encode("latin-1"), value.encode("latin-1"))
                for name, value in (headers or {}).items()
            ],
            "query_string": b"",
        }
    )


def _install_fake_redis_module(monkeypatch):
    redis_package = types.ModuleType("redis")
    redis_asyncio = types.ModuleType("redis.asyncio")
    redis_asyncio.from_url = lambda *args, **kwargs: None
    redis_package.asyncio = redis_asyncio
    monkeypatch.setitem(sys.modules, "redis", redis_package)
    monkeypatch.setitem(sys.modules, "redis.asyncio", redis_asyncio)
    sys.modules.pop("backend.middleware.rate_limiter", None)


def test_plan_enforcer_does_not_trust_rotated_tenant_headers():
    from backend.middleware.plan_enforcer import PlanEnforcerMiddleware

    request_a = _request(headers={"X-Tenant-ID": "tenant-a"})
    request_b = _request(headers={"X-Tenant-ID": "tenant-b"})

    assert PlanEnforcerMiddleware._resolve_tenant(request_a) == PlanEnforcerMiddleware._resolve_tenant(request_b)
    assert PlanEnforcerMiddleware._resolve_tenant(request_a) == ("anonymous:203.0.113.10", "free")


@pytest.mark.asyncio
async def test_plan_enforcer_does_not_exempt_every_api_path(monkeypatch):
    from backend.middleware.plan_enforcer import PlanEnforcerMiddleware

    middleware = PlanEnforcerMiddleware(lambda scope, receive, send: None)
    monkeypatch.setattr(middleware, "_increment_counter", lambda tenant_id: 101)

    async def call_next(request):
        return PlainTextResponse("unrestricted")

    response = await middleware.dispatch(_request(path="/v1/recommendations/id/100"), call_next)

    assert response.status_code == 429


@pytest.mark.asyncio
async def test_rate_limiter_uses_client_identity_not_rotated_tenant_headers(monkeypatch):
    _install_fake_redis_module(monkeypatch)
    import backend.middleware.rate_limiter as rate_limiter
    from backend.middleware.rate_limiter import RedisRateLimiter

    class FakeRedis:
        def __init__(self):
            self.keys = []

        async def incr(self, key):
            self.keys.append(key)
            return 1

        async def expire(self, key, seconds):
            return None

    fake_redis = FakeRedis()
    limiter = RedisRateLimiter.__new__(RedisRateLimiter)
    limiter.redis = fake_redis
    limiter.quotas = {"free": 10, "pro": 100, "enterprise": 5000}
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 12345)

    async def call_next(request):
        return PlainTextResponse("ok")

    await limiter.dispatch(_request(headers={"X-Tenant-ID": "tenant-a"}), call_next)
    await limiter.dispatch(_request(headers={"X-Tenant-ID": "tenant-b"}), call_next)

    assert fake_redis.keys == [
        "rate_limit:anonymous:203.0.113.10:12345",
        "rate_limit:anonymous:203.0.113.10:12345",
    ]


@pytest.mark.asyncio
async def test_rate_limiter_ignores_claimed_enterprise_tier(monkeypatch):
    _install_fake_redis_module(monkeypatch)
    import backend.middleware.rate_limiter as rate_limiter
    from backend.middleware.rate_limiter import RedisRateLimiter

    class FakeRedis:
        async def incr(self, key):
            return 11

        async def expire(self, key, seconds):
            return None

    limiter = RedisRateLimiter.__new__(RedisRateLimiter)
    limiter.redis = FakeRedis()
    limiter.quotas = {"free": 10, "pro": 100, "enterprise": 5000}
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 12345)

    async def call_next(request):
        return PlainTextResponse("ok")

    response = await limiter.dispatch(
        _request(headers={"X-Tenant-ID": "tenant-a", "X-Tenant-Tier": "enterprise"}),
        call_next,
    )

    assert response.status_code == 429
