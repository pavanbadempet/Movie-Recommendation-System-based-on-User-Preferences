"""Remote recommender service client.

Render free tier should not have to carry the vector-heavy serving stack when a
Hugging Face Space is already running the recommender API. This module lets the
API gateway call that Space first and fall back to local lite serving if needed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from backend.auth import TenantContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteResponse:
    """Normalized response from the remote recommender service."""

    status_code: int
    payload: Any


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
    timeout = float(os.getenv("NOVA_RECOMMENDER_SERVICE_TIMEOUT_SECONDS", "12"))

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(
                f"{base_url}{normalized_path}",
                params=params,
                headers=remote_recommender_headers(context),
            )
    except httpx.HTTPError as exc:
        logger.warning("Remote recommender unavailable for %s: %s", normalized_path, exc)
        return None

    if response.status_code >= 500:
        logger.warning(
            "Remote recommender returned %s for %s; using local fallback.",
            response.status_code,
            normalized_path,
        )
        return None

    try:
        payload = response.json()
    except ValueError:
        logger.warning("Remote recommender returned non-JSON for %s", normalized_path)
        return None

    return RemoteResponse(status_code=response.status_code, payload=payload)
