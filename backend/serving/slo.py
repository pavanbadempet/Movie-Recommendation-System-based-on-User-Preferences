"""Lightweight request SLO telemetry for the serving API."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from threading import Lock
import time
from typing import Any

BOOT_TIME = time.time()


DEFAULT_EXCLUDED_ROUTE_PREFIXES = (
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
    "/v1/artifacts",
    "/v1/diagnostics",
    "/v1/evaluation",
    "/v1/platform/readiness",
)
DEFAULT_ROUTE_LATENCY_BUDGETS_MS = {
    "/": 1000.0,
    "/health": 1000.0,
    "/v1/frontends/status": 3000.0,
    "/v1/platform/slo": 1000.0,
    "/v1/recommendations/id/{movie_id}": 25000.0,
    "/v1/search": 2500.0,
}


@dataclass(frozen=True)
class RequestSample:
    """Single HTTP request sample stored in a bounded in-memory window."""

    timestamp: float
    method: str
    path: str
    route: str
    status_code: int
    latency_ms: float


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return max(0, int(value))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        return default


def slo_thresholds() -> dict[str, float | int]:
    """Return the current SLO thresholds from environment configuration."""

    return {
        "window_seconds": _env_int("NOVA_SLO_WINDOW_SECONDS", 3600),
        "min_requests": _env_int("NOVA_SLO_MIN_REQUESTS", 5),
        "min_route_requests": _env_int("NOVA_SLO_MIN_ROUTE_REQUESTS", 20),
        "latency_p95_ms": _env_float("NOVA_SLO_LATENCY_P95_MS", 2500.0),
        "error_rate": _env_float("NOVA_SLO_ERROR_RATE", 0.03),
    }


def slo_excluded_route_prefixes() -> tuple[str, ...]:
    """Return route/path prefixes excluded from user-serving SLO math."""

    configured = os.getenv("NOVA_SLO_EXCLUDED_ROUTE_PREFIXES", "").strip()
    if not configured:
        return DEFAULT_EXCLUDED_ROUTE_PREFIXES
    return tuple(prefix.strip() for prefix in configured.split(",") if prefix.strip())


def slo_route_latency_budgets() -> dict[str, float]:
    """Return per-route p95 latency budgets in milliseconds."""

    configured = os.getenv("NOVA_SLO_ROUTE_LATENCY_BUDGETS", "").strip()
    if not configured:
        return dict(DEFAULT_ROUTE_LATENCY_BUDGETS_MS)

    budgets: dict[str, float] = {}
    for item in configured.split(","):
        route_budget = item.strip()
        if not route_budget or ":" not in route_budget:
            continue
        route, raw_budget = route_budget.rsplit(":", 1)
        try:
            budgets[route.strip()] = max(0.0, float(raw_budget.strip()))
        except ValueError:
            continue
    return budgets or dict(DEFAULT_ROUTE_LATENCY_BUDGETS_MS)


def should_track_request(*, path: str, route: str) -> bool:
    """Return whether this request belongs in the user-serving SLO window."""

    return all(not (path.startswith(prefix) or route.startswith(prefix)) for prefix in slo_excluded_route_prefixes())


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round((ordered[lower] * (1 - weight)) + (ordered[upper] * weight), 2)


class RequestSloTracker:
    """Bounded, process-local request SLO tracker.

    Hosted free tiers can restart at any time, so this intentionally tracks
    process-local health rather than pretending to be durable observability.
    The synthetic monitor and GitHub Actions reports provide the durable record.
    """

    def __init__(self, max_events: int | None = None) -> None:
        self.max_events = max_events or _env_int("NOVA_SLO_MAX_EVENTS", 5000)
        self._samples: deque[RequestSample] = deque(maxlen=max(100, self.max_events))
        self._lock = Lock()

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def record(
        self,
        *,
        method: str,
        path: str,
        route: str,
        status_code: int,
        latency_ms: float,
        timestamp: float | None = None,
    ) -> None:
        sample = RequestSample(
            timestamp=timestamp if timestamp is not None else time.time(),
            method=method.upper(),
            path=path,
            route=route or path,
            status_code=int(status_code),
            latency_ms=max(0.0, float(latency_ms)),
        )
        with self._lock:
            self._samples.append(sample)

    def snapshot(self, window_seconds: int | None = None) -> list[RequestSample]:
        now = time.time()
        cutoff = now - max(0, window_seconds or int(slo_thresholds()["window_seconds"]))
        with self._lock:
            return [sample for sample in self._samples if sample.timestamp >= cutoff]

    def summary(self, window_seconds: int | None = None) -> dict[str, Any]:
        samples = self.snapshot(window_seconds=window_seconds)
        latencies = [sample.latency_ms for sample in samples]
        error_samples = [sample for sample in samples if sample.status_code >= 500]
        route_stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "error_count": 0, "latencies": []}
        )
        for sample in samples:
            key = (sample.method, sample.route)
            route_stats[key]["count"] += 1
            route_stats[key]["error_count"] += int(sample.status_code >= 500)
            route_stats[key]["latencies"].append(sample.latency_ms)

        breakdown = []
        for (method, route), stats in route_stats.items():
            count = int(stats["count"])
            error_count = int(stats["error_count"])
            breakdown.append(
                {
                    "method": method,
                    "route": route,
                    "count": count,
                    "error_count": error_count,
                    "error_rate": round(error_count / count, 4) if count else 0.0,
                    "latency_ms": {
                        "p95": _percentile(stats["latencies"], 0.95),
                        "max": round(max(stats["latencies"]), 2) if stats["latencies"] else None,
                    },
                }
            )
        breakdown.sort(key=lambda item: (item["error_count"], item["count"]), reverse=True)

        request_count = len(samples)
        error_count = len(error_samples)
        return {
            "window_started_at": (
                datetime.fromtimestamp(min(sample.timestamp for sample in samples), UTC).isoformat()
                if samples
                else None
            ),
            "request_count": request_count,
            "error_count": error_count,
            "error_rate": round(error_count / request_count, 4) if request_count else 0.0,
            "latency_ms": {
                "avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
                "max": round(max(latencies), 2) if latencies else None,
            },
            "routes": breakdown[:10],
        }


def _dependency_state(dependencies: dict[str, Any]) -> str:
    artifact_status = ((dependencies.get("artifacts") or {}).get("status") or "").lower()
    remote_rec = dependencies.get("remote_recommender") or {}
    remote_configured = remote_rec.get("configured") is True
    remote_state = ((remote_rec.get("circuit") or {}).get("state") or "").lower()
    frontend_status = ((dependencies.get("frontends") or {}).get("status") or "").lower()

    degraded = False
    if not remote_configured:
        if artifact_status and artifact_status not in {"ready", "degraded"}:
            return "failed"
        if artifact_status == "degraded":
            degraded = True
    else:
        # If remote is configured, we rely on the remote recommender.
        # But we might still be degraded if the remote circuit is open or frontend is unavailable.
        if remote_state in {"open"}:
            degraded = True
        if frontend_status in {"unavailable", "failed"}:
            degraded = True

    return "degraded" if degraded else "ok"


def build_slo_report(
    *,
    tracker: RequestSloTracker,
    app: dict[str, Any],
    dependencies: dict[str, Any],
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a compact SLO report suitable for humans and synthetic checks."""

    thresholds = slo_thresholds()
    summary = tracker.summary(window_seconds=int(thresholds["window_seconds"]))
    request_count = int(summary["request_count"])
    p95_latency = summary["latency_ms"]["p95"]
    route_latency_budgets = slo_route_latency_budgets()
    default_latency_budget = float(thresholds["latency_p95_ms"])
    min_route_requests = int(thresholds["min_route_requests"])
    route_latency_violations = []
    low_sample_routes = []
    for route_summary in summary["routes"]:
        route = str(route_summary.get("route") or "")
        count = int(route_summary.get("count") or 0)
        route_p95 = (route_summary.get("latency_ms") or {}).get("p95")
        route_budget = route_latency_budgets.get(route, default_latency_budget)
        enforced = count >= min_route_requests
        route_summary["latency_ms"]["budget"] = route_budget
        route_summary["latency_ms"]["enforced"] = enforced
        route_summary["latency_ms"]["passed"] = route_p95 is None or route_p95 <= route_budget
        if not enforced:
            low_sample_routes.append(
                {
                    "route": route,
                    "count": count,
                    "minimum": min_route_requests,
                }
            )
        if enforced and not route_summary["latency_ms"]["passed"]:
            route_latency_violations.append(
                {
                    "route": route,
                    "actual": route_p95,
                    "budget": route_budget,
                }
            )

    latency_ok = not route_latency_violations
    error_rate_ok = float(summary["error_rate"]) <= float(thresholds["error_rate"])
    enough_requests = request_count >= int(thresholds["min_requests"])
    dependency_state = _dependency_state(dependencies)

    if dependency_state == "failed":
        status = "failed"
    elif not enough_requests or low_sample_routes:
        status = "warming"
    elif not latency_ok or not error_rate_ok:
        status = "violated"
    elif dependency_state == "degraded":
        status = "degraded"
    else:
        status = "ok"

    elapsed_since_boot = time.time() - BOOT_TIME
    uptime_seconds = int(elapsed_since_boot)
    window_seconds = int(thresholds["window_seconds"])
    elapsed_for_rate = min(float(window_seconds), max(1.0, elapsed_since_boot))
    request_rate = round(request_count / elapsed_for_rate, 2)

    routes_dict = {
        r["route"]: {
            "p95_ms": r["latency_ms"]["p95"],
            "error_rate": r["error_rate"],
            "count": r["count"],
        }
        for r in summary["routes"]
    }

    return {
        "status": status,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "app": app,
        "slo": {
            "window_seconds": thresholds["window_seconds"],
            "min_requests": thresholds["min_requests"],
            "min_route_requests": min_route_requests,
            "excluded_route_prefixes": list(slo_excluded_route_prefixes()),
            "latency_p95_ms": {
                "target": thresholds["latency_p95_ms"],
                "actual": p95_latency,
                "passed": latency_ok,
                "route_violations": route_latency_violations,
                "low_sample_routes": low_sample_routes,
            },
            "route_latency_budgets_ms": route_latency_budgets,
            "error_rate": {
                "target": thresholds["error_rate"],
                "actual": summary["error_rate"],
                "passed": error_rate_ok,
            },
            "has_enough_traffic": enough_requests,
        },
        "traffic": summary,
        "dependencies": dependencies,
        # Flat top-level keys for frontend compatibility
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": summary["latency_ms"]["p99"],
        "error_rate": summary["error_rate"],
        "total_requests": request_count,
        "request_rate": request_rate,
        "uptime_seconds": uptime_seconds,
        "window_seconds": window_seconds,
        "routes": routes_dict,
    }

