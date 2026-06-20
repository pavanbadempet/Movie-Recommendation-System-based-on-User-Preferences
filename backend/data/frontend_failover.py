"""Frontend availability and launch routing.

The API is the durable product surface. Frontends are replaceable clients, so
the backend keeps a small registry and can send users to the healthiest UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urljoin

import httpx

DEFAULT_STREAMLIT_URL = "https://a-movie-recommendation-system.streamlit.app"
DEFAULT_REACT_URL = "/ui/"
_HEALTH_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


@dataclass(frozen=True)
class FrontendTarget:
    """Configured frontend destination."""

    name: str
    label: str
    kind: str
    url: str
    priority: int
    health_url: str | None = None
    local: bool = False


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    if url.startswith("/"):
        return url if url.endswith("/") else f"{url}/"
    return url.rstrip("/")


def _priority_names() -> list[str]:
    raw = os.getenv("NOVA_FRONTEND_PRIORITY", "streamlit,react")
    names = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return names or ["streamlit", "react"]


def configured_frontends(*, frontend_available: bool) -> list[FrontendTarget]:
    """Return frontend registry in preferred failover order."""
    streamlit_url = _normalize_url(
        os.getenv("NOVA_FRONTEND_STREAMLIT_URL", "") or os.getenv("STREAMLIT_FRONTEND_URL", "") or DEFAULT_STREAMLIT_URL
    )
    react_url = _normalize_url(
        os.getenv("NOVA_FRONTEND_REACT_URL", "") or (DEFAULT_REACT_URL if frontend_available else "")
    )
    github_pages_url = _normalize_url(os.getenv("NOVA_FRONTEND_PAGES_URL", ""))

    configured: dict[str, FrontendTarget] = {}
    if streamlit_url and streamlit_url.lower() not in {"off", "disabled", "none"}:
        configured["streamlit"] = FrontendTarget(
            name="streamlit",
            label="Streamlit console",
            kind="streamlit",
            url=streamlit_url,
            health_url=streamlit_url,
            priority=100,
        )
    if react_url and react_url.lower() not in {"off", "disabled", "none"}:
        configured["react"] = FrontendTarget(
            name="react",
            label="React discovery UI",
            kind="react",
            url=react_url,
            health_url=react_url,
            priority=100,
            local=react_url.startswith("/"),
        )
    if github_pages_url:
        configured["github_pages"] = FrontendTarget(
            name="github_pages",
            label="GitHub Pages UI",
            kind="static",
            url=github_pages_url,
            health_url=github_pages_url,
            priority=100,
        )

    ordered: list[FrontendTarget] = []
    for index, name in enumerate(_priority_names()):
        target = configured.pop(name, None)
        if target is not None:
            ordered.append(
                FrontendTarget(
                    name=target.name,
                    label=target.label,
                    kind=target.kind,
                    url=target.url,
                    health_url=target.health_url,
                    priority=index + 1,
                    local=target.local,
                )
            )

    next_priority = len(ordered) + 1
    for target in configured.values():
        ordered.append(
            FrontendTarget(
                name=target.name,
                label=target.label,
                kind=target.kind,
                url=target.url,
                health_url=target.health_url,
                priority=next_priority,
                local=target.local,
            )
        )
        next_priority += 1

    return ordered


def _local_frontend_probe(target: FrontendTarget, frontend_dist_dir: Path) -> dict[str, Any]:
    index_path = frontend_dist_dir / "index.html"
    exists = index_path.exists()
    return {
        "name": target.name,
        "label": target.label,
        "kind": target.kind,
        "url": target.url,
        "health_url": target.health_url,
        "priority": target.priority,
        "local": target.local,
        "status": "ok" if exists else "unavailable",
        "http_status": None,
        "latency_ms": 0,
        "error": None if exists else f"{index_path} is missing",
    }


async def _remote_frontend_probe(target: FrontendTarget) -> dict[str, Any]:
    timeout_seconds = _env_float("NOVA_FRONTEND_HEALTH_TIMEOUT_SECONDS", 2.5)
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
            response = await client.get(
                target.health_url or target.url,
                headers={"Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8"},
            )
    except httpx.HTTPError as exc:
        return {
            "name": target.name,
            "label": target.label,
            "kind": target.kind,
            "url": target.url,
            "health_url": target.health_url,
            "priority": target.priority,
            "local": target.local,
            "status": "unavailable",
            "http_status": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": str(exc),
        }

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    if 200 <= response.status_code < 400:
        status = "ok"
        error = None
    elif response.status_code < 500:
        status = "degraded"
        error = f"HTTP {response.status_code}"
    else:
        status = "unavailable"
        error = f"HTTP {response.status_code}"

    return {
        "name": target.name,
        "label": target.label,
        "kind": target.kind,
        "url": target.url,
        "health_url": target.health_url,
        "priority": target.priority,
        "local": target.local,
        "status": status,
        "http_status": response.status_code,
        "latency_ms": latency_ms,
        "error": error,
    }


async def probe_frontend(
    target: FrontendTarget,
    *,
    frontend_dist_dir: Path,
    include_remote: bool,
) -> dict[str, Any]:
    """Probe one frontend, using a short cache for remote targets."""
    if target.local:
        return _local_frontend_probe(target, frontend_dist_dir)

    if not include_remote:
        return {
            "name": target.name,
            "label": target.label,
            "kind": target.kind,
            "url": target.url,
            "health_url": target.health_url,
            "priority": target.priority,
            "local": target.local,
            "status": "unknown",
            "http_status": None,
            "latency_ms": None,
            "error": "remote probe skipped",
        }

    ttl = max(0, _env_int("NOVA_FRONTEND_HEALTH_CACHE_SECONDS", 30))
    cache_key = f"{target.name}:{target.health_url or target.url}"
    cached = _HEALTH_CACHE.get(cache_key)
    if cached and time.time() - cached[0] <= ttl:
        return {**cached[1], "cached": True}

    report = await _remote_frontend_probe(target)
    _HEALTH_CACHE[cache_key] = (time.time(), report)
    return {**report, "cached": False}


def choose_frontend(frontends: list[dict[str, Any]], preferred: str | None = None) -> dict[str, Any] | None:
    """Pick the first healthy frontend, honoring a requested preference."""
    ordered = sorted(frontends, key=lambda item: int(item.get("priority") or 999))
    if preferred:
        preferred_normalized = preferred.strip().lower()
        for item in ordered:
            if item.get("name") == preferred_normalized and item.get("status") in {"ok", "degraded"}:
                return item
    for item in ordered:
        if item.get("status") == "ok":
            return item
    for item in ordered:
        if item.get("status") == "degraded":
            return item
    for item in ordered:
        if item.get("status") == "unknown":
            return item
    return ordered[0] if ordered else None


def absolute_frontend_url(url: str, base_url: str) -> str:
    """Resolve local frontend paths against the current API host."""
    if url.startswith("/") and base_url:
        return urljoin(base_url.rstrip("/") + "/", url.lstrip("/"))
    return url


async def frontend_status_report(
    *,
    frontend_dist_dir: Path,
    base_url: str,
    include_remote: bool = True,
    preferred: str | None = None,
    app: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return all frontend statuses and the selected launch target."""
    targets = configured_frontends(frontend_available=(frontend_dist_dir / "index.html").exists())
    checks = [
        await probe_frontend(
            target,
            frontend_dist_dir=frontend_dist_dir,
            include_remote=include_remote,
        )
        for target in targets
    ]
    selected = choose_frontend(checks, preferred=preferred)
    if selected is not None:
        selected = {
            **selected,
            "absolute_url": absolute_frontend_url(str(selected["url"]), base_url),
        }

    ok_count = sum(1 for item in checks if item.get("status") == "ok")
    if ok_count == len(checks) and checks:
        status = "ready"
    elif selected is not None and selected.get("status") in {"ok", "degraded", "unknown"}:
        status = "degraded" if ok_count < len(checks) else "ready"
    else:
        status = "unavailable"

    return {
        "status": status,
        "mode": "multi_frontend_failover",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "app": app or {},
        "selected": selected,
        "launch_url": selected.get("absolute_url") if selected else None,
        "frontends": checks,
        "policy": {
            "priority": [target.name for target in targets],
            "remote_checks": include_remote,
            "cache_seconds": max(0, _env_int("NOVA_FRONTEND_HEALTH_CACHE_SECONDS", 30)),
            "timeout_seconds": _env_float("NOVA_FRONTEND_HEALTH_TIMEOUT_SECONDS", 2.5),
        },
    }
