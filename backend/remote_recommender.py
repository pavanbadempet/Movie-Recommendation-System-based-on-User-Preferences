"""Remote recommender service client.

Render free tier should not have to carry the vector-heavy serving stack when a
Hugging Face Space is already running the recommender API. This module lets the
API gateway call that Space first and fall back to local lite serving if needed.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from backend.auth import TenantContext

logger = logging.getLogger(__name__)


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
    raw_url = (
        os.getenv("NOVA_RECOMMENDER_SERVICE_URL", "")
        or os.getenv("NOVA_VECTOR_SERVICE_URL", "")
    ).strip()
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
    if state.opened_until <= time.time():
        return False
    return True


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


def _store_cached_response(cache_key: str, status_code: int, payload: Any) -> None:
    if not _cache_enabled():
        return
    if status_code < 200 or status_code >= 400:
        return
    _response_cache[cache_key] = _CacheEntry(
        created_at=time.time(),
        status_code=status_code,
        payload=payload,
    )
    max_entries = _cache_max_entries()
    if len(_response_cache) <= max_entries:
        return
    oldest_keys = sorted(
        _response_cache,
        key=lambda key: _response_cache[key].created_at,
    )
    for oldest_key in oldest_keys[: len(_response_cache) - max_entries]:
        _response_cache.pop(oldest_key, None)


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

    if _circuit_open(base_url):
        cached = _fallback_cached_response(cache_key, "remote circuit is open")
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
        return _fallback_cached_response(cache_key, "remote request failure")

    if _transient_remote_status(response.status_code):
        _record_remote_failure(base_url, f"HTTP {response.status_code}")
        logger.warning(
            "Remote recommender returned %s for %s; using local fallback.",
            response.status_code,
            normalized_path,
        )
        return _fallback_cached_response(cache_key, f"HTTP {response.status_code}")

    try:
        payload = response.json()
    except ValueError:
        _record_remote_failure(base_url, "non-JSON response")
        logger.warning("Remote recommender returned non-JSON for %s", normalized_path)
        return _fallback_cached_response(cache_key, "non-JSON response")

    _record_remote_success(base_url)
    _store_cached_response(cache_key, response.status_code, payload)
    return RemoteResponse(status_code=response.status_code, payload=payload)
