"""Remote recommender service client.

Render free tier should not have to carry the vector-heavy serving stack when a
Hugging Face Space is already running the recommender API. This module lets the
API gateway call that Space first and fall back to local lite serving if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from backend.auth import TenantContext

logger = logging.getLogger(__name__)

# Fast JSON for Redis cache serialization
try:
    import orjson as _orjson

    def _cache_dumps(obj) -> str:
        return _orjson.dumps(obj).decode()

    def _cache_loads(s):
        return _orjson.loads(s)
except ImportError:

    def _cache_dumps(obj) -> str:
        return json.dumps(obj, separators=(",", ":"))

    def _cache_loads(s):
        return json.loads(s)


@dataclass(frozen=True)
class RemoteResponse:
    """Normalized response from the remote recommender service."""

    status_code: int
    payload: Any
    source: str = "remote"
    cache_status: str | None = None


@dataclass
class _CircuitState:
    failure_count: int = 0
    opened_until: float = 0.0
    last_error: str | None = None


@dataclass
class _CacheEntry:
    created_at: float
    status_code: int
    payload: Any


_circuit_states: dict[str, _CircuitState] = {}
_response_cache: dict[str, _CacheEntry] = {}
_distributed_cache_stats = {
    "hits": 0,
    "misses": 0,
    "writes": 0,
    "errors": 0,
}


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default
    return raw_value in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def remote_recommender_url() -> str | None:
    """Return the configured recommender service base URL, if any."""
    raw_url = (os.getenv("NOVA_RECOMMENDER_SERVICE_URL", "") or os.getenv("NOVA_VECTOR_SERVICE_URL", "")).strip()
    if not raw_url:
        return None
    return raw_url.rstrip("/")


def remote_recommender_headers(context: TenantContext | None = None) -> dict[str, str]:
    """Build headers for a remote recommender request."""
    headers = {
        "Accept": "application/json",
        "X-Nova-Proxy": "render-gateway",
    }

    api_key = os.getenv("NOVA_RECOMMENDER_SERVICE_API_KEY", "").strip()
    if api_key:
        headers["X-Nova-API-Key"] = api_key

    if context is not None:
        headers["X-Tenant-ID"] = context.tenant_id
        headers["X-Catalog-ID"] = context.catalog_id

    return headers


def _circuit_state(base_url: str) -> _CircuitState:
    return _circuit_states.setdefault(base_url, _CircuitState())


def _circuit_open(base_url: str) -> bool:
    state = _circuit_state(base_url)
    return not state.opened_until <= time.time()


def _record_remote_success(base_url: str) -> None:
    state = _circuit_state(base_url)
    state.failure_count = 0
    state.opened_until = 0.0
    state.last_error = None


def _record_remote_failure(base_url: str, error: str) -> None:
    state = _circuit_state(base_url)
    state.failure_count += 1
    state.last_error = error
    threshold = _env_int("NOVA_RECOMMENDER_CIRCUIT_FAILURE_THRESHOLD", 3, minimum=1)
    if state.failure_count >= threshold:
        open_seconds = _env_int("NOVA_RECOMMENDER_CIRCUIT_OPEN_SECONDS", 60, minimum=1)
        state.opened_until = time.time() + open_seconds
        logger.warning(
            "Remote recommender circuit opened for %ss after %s failures: %s",
            open_seconds,
            state.failure_count,
            error,
        )


def _normalized_cache_key(
    *,
    base_url: str,
    path: str,
    params: dict[str, Any] | None,
    context: TenantContext | None,
) -> str:
    normalized_params = []
    for key, value in sorted((params or {}).items()):
        if value is None:
            continue
        normalized_params.append((str(key), str(value)))
    tenant = context.tenant_id if context is not None else ""
    catalog = context.catalog_id if context is not None else ""
    plan = context.plan if context is not None else ""
    query = urlencode(normalized_params)
    return f"{base_url}|{path}|{query}|{tenant}|{catalog}|{plan}"


def _cache_enabled() -> bool:
    return _env_bool("NOVA_RECOMMENDER_CACHE_ENABLED", True)


def _fresh_cache_reads_enabled() -> bool:
    return _env_bool("NOVA_RECOMMENDER_CACHE_READS", True)


def _cache_ttl_seconds() -> int:
    return _env_int("NOVA_RECOMMENDER_CACHE_TTL_SECONDS", 300, minimum=1)


def _stale_cache_ttl_seconds() -> int:
    return _env_int("NOVA_RECOMMENDER_STALE_CACHE_TTL_SECONDS", 21600, minimum=1)


def _distributed_cache_enabled() -> bool:
    return (
        _cache_enabled()
        and _env_bool("NOVA_RECOMMENDER_DISTRIBUTED_CACHE_ENABLED", True)
        and bool(_upstash_rest_url() and _upstash_rest_token())
    )


def _distributed_cache_timeout_seconds() -> float:
    return _env_float("NOVA_RECOMMENDER_DISTRIBUTED_CACHE_TIMEOUT_SECONDS", 1.5, minimum=0.1)


def _upstash_rest_url() -> str | None:
    raw_url = (
        os.getenv("UPSTASH_REDIS_REST_URL", "") or os.getenv("NOVA_RECOMMENDER_DISTRIBUTED_CACHE_URL", "")
    ).strip()
    if not raw_url:
        return None
    return raw_url.rstrip("/")


def _upstash_rest_token() -> str | None:
    return (
        os.getenv("UPSTASH_REDIS_REST_TOKEN", "") or os.getenv("NOVA_RECOMMENDER_DISTRIBUTED_CACHE_TOKEN", "")
    ).strip() or None


def _distributed_cache_key(cache_key: str) -> str:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return f"nova:remote-recommender:{digest}"


def _cache_max_entries() -> int:
    return _env_int("NOVA_RECOMMENDER_CACHE_MAX_ENTRIES", 512, minimum=1)


def _transient_remote_status(status_code: int) -> bool:
    return status_code in {408, 425, 429} or status_code >= 500


def _get_cached_response(cache_key: str, *, allow_stale: bool) -> RemoteResponse | None:
    if not _cache_enabled():
        return None
    entry = _response_cache.get(cache_key)
    if entry is None:
        return None

    age_seconds = time.time() - entry.created_at
    if age_seconds <= _cache_ttl_seconds():
        cache_status = "fresh"
    elif allow_stale and age_seconds <= _stale_cache_ttl_seconds():
        cache_status = "stale"
    else:
        return None

    return RemoteResponse(
        status_code=entry.status_code,
        payload=entry.payload,
        source="cache",
        cache_status=cache_status,
    )


def _prune_local_cache() -> None:
    max_entries = _cache_max_entries()
    if len(_response_cache) <= max_entries:
        return
    oldest_keys = sorted(
        _response_cache,
        key=lambda key: _response_cache[key].created_at,
    )
    for oldest_key in oldest_keys[: len(_response_cache) - max_entries]:
        _response_cache.pop(oldest_key, None)


def _store_local_cache_entry(cache_key: str, entry: _CacheEntry) -> None:
    if not _cache_enabled():
        return
    _response_cache[cache_key] = entry
    _prune_local_cache()


def _store_cached_response(cache_key: str, status_code: int, payload: Any) -> None:
    if status_code < 200 or status_code >= 400:
        return
    _store_local_cache_entry(
        cache_key,
        _CacheEntry(
            created_at=time.time(),
            status_code=status_code,
            payload=payload,
        ),
    )


async def _get_distributed_cached_response(cache_key: str, *, allow_stale: bool) -> RemoteResponse | None:
    if not _distributed_cache_enabled():
        return None

    rest_url = _upstash_rest_url()
    rest_token = _upstash_rest_token()
    if not rest_url or not rest_token:
        return None

    try:
        async with httpx.AsyncClient(timeout=_distributed_cache_timeout_seconds()) as client:
            response = await client.post(
                rest_url,
                headers={"Authorization": f"Bearer {rest_token}"},
                json=["GET", _distributed_cache_key(cache_key)],
            )
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        _distributed_cache_stats["errors"] += 1
        logger.info("Distributed recommender cache read skipped: %s", exc)
        return None

    if response.status_code != 200 or payload.get("error"):
        _distributed_cache_stats["errors"] += 1
        logger.info("Distributed recommender cache read failed: %s", payload.get("error") or response.status_code)
        return None

    raw_entry = payload.get("result")
    if raw_entry is None:
        _distributed_cache_stats["misses"] += 1
        return None

    try:
        parsed_entry = _cache_loads(raw_entry)
        entry = _CacheEntry(
            created_at=float(parsed_entry["created_at"]),
            status_code=int(parsed_entry["status_code"]),
            payload=parsed_entry["payload"],
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        _distributed_cache_stats["errors"] += 1
        logger.info("Distributed recommender cache entry is invalid: %s", exc)
        return None

    age_seconds = time.time() - entry.created_at
    if age_seconds <= _cache_ttl_seconds():
        cache_status = "fresh"
    elif allow_stale and age_seconds <= _stale_cache_ttl_seconds():
        cache_status = "stale"
    else:
        return None

    _distributed_cache_stats["hits"] += 1
    _store_local_cache_entry(cache_key, entry)
    return RemoteResponse(
        status_code=entry.status_code,
        payload=entry.payload,
        source="distributed_cache",
        cache_status=cache_status,
    )


async def _store_distributed_cached_response(cache_key: str, status_code: int, payload: Any) -> None:
    if not _distributed_cache_enabled():
        return
    if status_code < 200 or status_code >= 400:
        return

    rest_url = _upstash_rest_url()
    rest_token = _upstash_rest_token()
    if not rest_url or not rest_token:
        return

    entry = {
        "created_at": time.time(),
        "status_code": int(status_code),
        "payload": payload,
    }
    try:
        async with httpx.AsyncClient(timeout=_distributed_cache_timeout_seconds()) as client:
            response = await client.post(
                rest_url,
                headers={"Authorization": f"Bearer {rest_token}"},
                json=[
                    "SET",
                    _distributed_cache_key(cache_key),
                    _cache_dumps(entry),
                    "EX",
                    _stale_cache_ttl_seconds(),
                ],
            )
        response_payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        _distributed_cache_stats["errors"] += 1
        logger.info("Distributed recommender cache write skipped: %s", exc)
        return

    if response.status_code != 200 or response_payload.get("error"):
        _distributed_cache_stats["errors"] += 1
        logger.info(
            "Distributed recommender cache write failed: %s", response_payload.get("error") or response.status_code
        )
        return
    _distributed_cache_stats["writes"] += 1


def _fallback_cached_response(cache_key: str, reason: str) -> RemoteResponse | None:
    cached = _get_cached_response(cache_key, allow_stale=True)
    if cached is None:
        return None
    logger.warning("Serving %s cached remote recommender response after %s", cached.cache_status, reason)
    return cached


def remote_recommender_status() -> dict[str, Any]:
    """Return circuit/cache state for diagnostics without probing the remote."""
    base_url = remote_recommender_url()
    state = _circuit_state(base_url) if base_url else _CircuitState()
    remaining_open_seconds = max(0.0, state.opened_until - time.time())
    return {
        "configured": bool(base_url),
        "base_url": base_url,
        "circuit": {
            "state": "open" if remaining_open_seconds > 0 else "closed",
            "failure_count": state.failure_count,
            "open_remaining_seconds": round(remaining_open_seconds, 2),
            "last_error": state.last_error,
            "failure_threshold": _env_int("NOVA_RECOMMENDER_CIRCUIT_FAILURE_THRESHOLD", 3, minimum=1),
            "open_seconds": _env_int("NOVA_RECOMMENDER_CIRCUIT_OPEN_SECONDS", 60, minimum=1),
        },
        "cache": {
            "enabled": _cache_enabled(),
            "read_before_remote": _fresh_cache_reads_enabled(),
            "entry_count": len(_response_cache),
            "max_entries": _cache_max_entries(),
            "ttl_seconds": _cache_ttl_seconds(),
            "stale_ttl_seconds": _stale_cache_ttl_seconds(),
            "distributed": {
                "enabled": _distributed_cache_enabled(),
                "provider": "upstash_redis_rest" if _distributed_cache_enabled() else None,
                "timeout_seconds": _distributed_cache_timeout_seconds(),
                **_distributed_cache_stats,
            },
        },
    }


async def remote_get_json(
    path: str,
    params: dict[str, Any] | None = None,
    context: TenantContext | None = None,
) -> RemoteResponse | None:
    """GET JSON from the remote recommender service.

    Returns None when no service is configured or when the remote service is not
    healthy enough to trust. Callers can then use their local fallback path.
    """
    base_url = remote_recommender_url()
    if not base_url:
        return None

    normalized_path = path if path.startswith("/") else f"/{path}"
    cache_key = _normalized_cache_key(
        base_url=base_url,
        path=normalized_path,
        params=params,
        context=context,
    )

    if _fresh_cache_reads_enabled():
        cached = _get_cached_response(cache_key, allow_stale=False)
        if cached is not None:
            return cached
        cached = await _get_distributed_cached_response(cache_key, allow_stale=False)
        if cached is not None:
            return cached

    if _circuit_open(base_url):
        cached = _fallback_cached_response(cache_key, "remote circuit is open")
        if cached is not None:
            return cached
        cached = await _get_distributed_cached_response(cache_key, allow_stale=True)
        if cached is not None:
            return cached
        return None

    timeout = _env_float("NOVA_RECOMMENDER_SERVICE_TIMEOUT_SECONDS", 12.0, minimum=0.1)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(
                f"{base_url}{normalized_path}",
                params=params,
                headers=remote_recommender_headers(context),
            )
    except httpx.HTTPError as exc:
        _record_remote_failure(base_url, str(exc))
        logger.warning("Remote recommender unavailable for %s: %s", normalized_path, exc)
        cached = _fallback_cached_response(cache_key, "remote request failure")
        if cached is not None:
            return cached
        return await _get_distributed_cached_response(cache_key, allow_stale=True)

    if _transient_remote_status(response.status_code):
        _record_remote_failure(base_url, f"HTTP {response.status_code}")
        logger.warning(
            "Remote recommender returned %s for %s; using local fallback.",
            response.status_code,
            normalized_path,
        )
        cached = _fallback_cached_response(cache_key, f"HTTP {response.status_code}")
        if cached is not None:
            return cached
        return await _get_distributed_cached_response(cache_key, allow_stale=True)

    try:
        payload = response.json()
    except ValueError:
        _record_remote_failure(base_url, "non-JSON response")
        logger.warning("Remote recommender returned non-JSON for %s", normalized_path)
        cached = _fallback_cached_response(cache_key, "non-JSON response")
        if cached is not None:
            return cached
        return await _get_distributed_cached_response(cache_key, allow_stale=True)

    _record_remote_success(base_url)
    _store_cached_response(cache_key, response.status_code, payload)
    await _store_distributed_cached_response(cache_key, response.status_code, payload)
    return RemoteResponse(status_code=response.status_code, payload=payload)
