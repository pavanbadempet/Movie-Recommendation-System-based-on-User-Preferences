"""Remote recommender resilience tests."""

import asyncio
import time

import httpx


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def reset_remote_state(remote):
    remote._circuit_states.clear()
    remote._response_cache.clear()


def test_remote_get_json_uses_fresh_cache(monkeypatch):
    import backend.remote_recommender as remote

    reset_remote_state(remote)
    monkeypatch.setenv("NOVA_RECOMMENDER_SERVICE_URL", "https://vector.example")
    monkeypatch.setenv("NOVA_RECOMMENDER_CACHE_READS", "true")

    calls = {"count": 0}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            calls["count"] += 1
            return FakeResponse(200, {"results": ["avatar"]})

    monkeypatch.setattr(remote.httpx, "AsyncClient", FakeAsyncClient)

    first = asyncio.run(remote.remote_get_json("/v1/search", params={"q": "avatar"}))
    second = asyncio.run(remote.remote_get_json("/v1/search", params={"q": "avatar"}))

    assert calls["count"] == 1
    assert first.source == "remote"
    assert second.source == "cache"
    assert second.cache_status == "fresh"
    assert second.payload == {"results": ["avatar"]}


def test_remote_get_json_serves_stale_cache_on_failure(monkeypatch):
    import backend.remote_recommender as remote

    reset_remote_state(remote)
    monkeypatch.setenv("NOVA_RECOMMENDER_SERVICE_URL", "https://vector.example")
    monkeypatch.setenv("NOVA_RECOMMENDER_CACHE_READS", "false")
    monkeypatch.setenv("NOVA_RECOMMENDER_CACHE_TTL_SECONDS", "1")
    monkeypatch.setenv("NOVA_RECOMMENDER_STALE_CACHE_TTL_SECONDS", "100")

    class WarmClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            return FakeResponse(200, {"recommendations": ["cached"]})

    monkeypatch.setattr(remote.httpx, "AsyncClient", WarmClient)
    first = asyncio.run(remote.remote_get_json("/v1/recommendations/id/1", params={"n": 10}))
    assert first.source == "remote"

    for entry in remote._response_cache.values():
        entry.created_at = time.time() - 5

    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            raise httpx.ConnectTimeout("remote asleep")

    monkeypatch.setattr(remote.httpx, "AsyncClient", FailingClient)
    fallback = asyncio.run(remote.remote_get_json("/v1/recommendations/id/1", params={"n": 10}))

    assert fallback.source == "cache"
    assert fallback.cache_status == "stale"
    assert fallback.payload == {"recommendations": ["cached"]}


def test_remote_circuit_opens_after_repeated_failures(monkeypatch):
    import backend.remote_recommender as remote

    reset_remote_state(remote)
    monkeypatch.setenv("NOVA_RECOMMENDER_SERVICE_URL", "https://vector.example")
    monkeypatch.setenv("NOVA_RECOMMENDER_CACHE_READS", "false")
    monkeypatch.setenv("NOVA_RECOMMENDER_CIRCUIT_FAILURE_THRESHOLD", "1")
    monkeypatch.setenv("NOVA_RECOMMENDER_CIRCUIT_OPEN_SECONDS", "60")

    calls = {"count": 0}

    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            calls["count"] += 1
            raise httpx.ConnectTimeout("remote asleep")

    monkeypatch.setattr(remote.httpx, "AsyncClient", FailingClient)

    assert asyncio.run(remote.remote_get_json("/v1/search", params={"q": "avatar"})) is None
    assert asyncio.run(remote.remote_get_json("/v1/search", params={"q": "avatar"})) is None

    status = remote.remote_recommender_status()
    assert calls["count"] == 1
    assert status["circuit"]["state"] == "open"
    assert status["circuit"]["failure_count"] == 1


def test_remote_429_uses_stale_cache_instead_of_client_error(monkeypatch):
    import backend.remote_recommender as remote

    reset_remote_state(remote)
    monkeypatch.setenv("NOVA_RECOMMENDER_SERVICE_URL", "https://vector.example")
    monkeypatch.setenv("NOVA_RECOMMENDER_CACHE_READS", "false")
    monkeypatch.setenv("NOVA_RECOMMENDER_CACHE_TTL_SECONDS", "1")
    monkeypatch.setenv("NOVA_RECOMMENDER_STALE_CACHE_TTL_SECONDS", "100")

    class RateLimitedClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            return FakeResponse(429, {"detail": "rate limited"})

    remote._response_cache[
        "https://vector.example|/v1/search|q=avatar|||"
    ] = remote._CacheEntry(
        created_at=time.time() - 5,
        status_code=200,
        payload=[{"title": "Avatar"}],
    )
    monkeypatch.setattr(remote.httpx, "AsyncClient", RateLimitedClient)

    fallback = asyncio.run(remote.remote_get_json("/v1/search", params={"q": "avatar"}))

    assert fallback.status_code == 200
    assert fallback.source == "cache"
    assert fallback.cache_status == "stale"
    assert fallback.payload == [{"title": "Avatar"}]
