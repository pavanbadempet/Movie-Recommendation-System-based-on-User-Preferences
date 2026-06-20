"""Pure helpers for the Streamlit backend-telemetry dashboard."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _payload(api_get: Callable[..., Any], path: str, params: dict | None = None) -> dict:
    value = api_get(path, params=params, timeout=15)
    return value if isinstance(value, dict) else {}


def build_monitoring_snapshot(api_get: Callable[..., Any]) -> dict:
    """Fetch a point-in-time monitoring snapshot from real backend endpoints."""
    health = _payload(api_get, "/health")
    platform = _payload(api_get, "/v1/platform/status")
    artifacts = _payload(api_get, "/v1/artifacts/health")
    features = _payload(api_get, "/v1/events/features", {"limit": 20})
    analytics = _payload(api_get, "/v1/events/recommendation-analytics", {"limit": 20})

    event_store = platform.get("event_store") if isinstance(platform.get("event_store"), dict) else {}
    event_type_counts = (
        features.get("event_type_counts") if isinstance(features.get("event_type_counts"), dict) else {}
    )
    return {
        "telemetry_source": "backend_api",
        "health_status": health.get("status", "unavailable"),
        "platform_status": platform.get("status", "unavailable"),
        "artifact_status": artifacts.get("status", "unavailable"),
        "serving_tier": health.get("serving_tier", "unknown"),
        "movie_count": int(health.get("movie_count") or 0),
        "event_store": event_store.get("mode", "unavailable"),
        "durable": bool(event_store.get("durable", False)),
        "total_events": int(event_store.get("total_events") or 0),
        "event_type_counts": {str(key): int(value) for key, value in event_type_counts.items()},
        "top_searches": features.get("top_searches") if isinstance(features.get("top_searches"), list) else [],
        "impression_count": int(analytics.get("impression_count") or 0),
        "click_count": int(analytics.get("click_count") or 0),
        "click_through_rate": analytics.get("click_through_rate"),
        "analytics_available": bool(analytics),
        "event_features_available": bool(features),
        "artifact_rows": artifacts.get("row_counts") if isinstance(artifacts.get("row_counts"), dict) else {},
    }
