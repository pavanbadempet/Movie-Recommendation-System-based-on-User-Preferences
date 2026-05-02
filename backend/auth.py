"""
Tenant/API-key helpers for the Nova product API.

The public demo stays frictionless when NOVA_API_KEYS is empty. When keys are
configured, the same backend becomes tenant-aware without adding a paid auth
provider or database dependency.
"""

from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass
from typing import Any

from fastapi import Header, HTTPException

DEFAULT_TENANT_ID = os.getenv("NOVA_TENANT_ID", "demo-media-co")
DEFAULT_CATALOG_ID = os.getenv("NOVA_CATALOG_ID", "tmdb-movies")


@dataclass(frozen=True)
class TenantContext:
    """Resolved customer/catalog context for a request."""

    tenant_id: str
    catalog_id: str
    plan: str = "demo"
    authenticated: bool = False
    api_key_label: str | None = None


def _parse_api_keys(raw_value: str | None = None) -> dict[str, dict[str, str]]:
    """
    Parse NOVA_API_KEYS.

    Supported formats:
    - JSON: {"key": {"tenant_id": "acme", "catalog_id": "main", "plan": "free"}}
    - CSV: key:tenant_id:catalog_id:plan,key2:tenant2:catalog2
    """
    raw_value = raw_value if raw_value is not None else os.getenv("NOVA_API_KEYS", "")
    raw_value = raw_value.strip()
    if not raw_value:
        return {}

    if raw_value.startswith("{"):
        parsed = json.loads(raw_value)
        result: dict[str, dict[str, str]] = {}
        for api_key, metadata in parsed.items():
            if not isinstance(metadata, dict):
                raise ValueError("Each NOVA_API_KEYS JSON value must be an object")
            result[str(api_key)] = {
                "tenant_id": str(metadata.get("tenant_id") or DEFAULT_TENANT_ID),
                "catalog_id": str(metadata.get("catalog_id") or DEFAULT_CATALOG_ID),
                "plan": str(metadata.get("plan") or "free"),
                "label": str(metadata.get("label") or str(api_key)[:8]),
            }
        return result

    result = {}
    for item in raw_value.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) < 3 or not parts[0]:
            raise ValueError("NOVA_API_KEYS entries must be key:tenant_id:catalog_id[:plan]")
        api_key, tenant_id, catalog_id = parts[:3]
        plan = parts[3] if len(parts) >= 4 and parts[3] else "free"
        result[api_key] = {
            "tenant_id": tenant_id,
            "catalog_id": catalog_id,
            "plan": plan,
            "label": api_key[:8],
        }
    return result


def configured_api_keys() -> dict[str, dict[str, str]]:
    """Return API-key configuration, failing closed if the env var is malformed."""
    try:
        return _parse_api_keys()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid NOVA_API_KEYS configuration: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Invalid NOVA_API_KEYS JSON") from exc


def resolve_tenant_context(
    x_nova_api_key: str | None = Header(default=None, alias="X-Nova-API-Key"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    x_catalog_id: str | None = Header(default=None, alias="X-Catalog-ID"),
) -> TenantContext:
    """
    Resolve request tenant context.

    If no API keys are configured, headers can provide local demo context. If
    keys are configured, a valid key is required and wins over headers.
    """
    api_keys = configured_api_keys()
    if not api_keys:
        return TenantContext(
            tenant_id=x_tenant_id or DEFAULT_TENANT_ID,
            catalog_id=x_catalog_id or DEFAULT_CATALOG_ID,
            plan="demo",
            authenticated=False,
        )

    if not x_nova_api_key:
        raise HTTPException(status_code=401, detail="X-Nova-API-Key is required")

    for configured_key, metadata in api_keys.items():
        if hmac.compare_digest(configured_key, x_nova_api_key):
            return TenantContext(
                tenant_id=metadata["tenant_id"],
                catalog_id=metadata["catalog_id"],
                plan=metadata.get("plan", "free"),
                authenticated=True,
                api_key_label=metadata.get("label"),
            )

    raise HTTPException(status_code=401, detail="Invalid X-Nova-API-Key")


def enforce_payload_context(
    payload: Any,
    context: TenantContext,
) -> None:
    """Prevent one API key from writing events into another tenant/catalog."""
    if not context.authenticated:
        return
    payload_tenant = getattr(payload, "tenant_id", None)
    payload_catalog = getattr(payload, "catalog_id", None)
    if payload_tenant and payload_tenant != context.tenant_id:
        raise HTTPException(status_code=403, detail="tenant_id does not match API key context")
    if payload_catalog and payload_catalog != context.catalog_id:
        raise HTTPException(status_code=403, detail="catalog_id does not match API key context")
