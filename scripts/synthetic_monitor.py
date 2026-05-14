"""Lightweight live product monitor for hosted free-tier deployments.

This is intentionally cheaper than the full serving quality gate. It probes
the real user paths often: health, UI failover, SLOs, search, and one canonical
recommendation case.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _get_json(base_url: str, path: str, timeout: int) -> Any:
    request = urllib.request.Request(
        _url(base_url, path),
        headers={"User-Agent": "movie-rec-synthetic-monitor/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured public URLs only
        return json.loads(response.read().decode("utf-8"))


def _fail(check: dict[str, Any], message: str) -> dict[str, Any]:
    check["status"] = "failed"
    check["error"] = message
    return check


def _validate_health(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "/health did not return an object"
    if payload.get("status") != "healthy":
        return f"/health status is {payload.get('status')!r}"
    return None


def _validate_frontends(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "/v1/frontends/status did not return an object"
    if payload.get("status") not in {"ready", "degraded"}:
        return f"/v1/frontends/status is {payload.get('status')!r}"
    selected = payload.get("selected") or {}
    if not selected.get("absolute_url"):
        return "/v1/frontends/status did not select a launch URL"
    return None


def _validate_slo(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "/v1/platform/slo did not return an object"
    if payload.get("status") in {"failed", "violated"}:
        return f"/v1/platform/slo status is {payload.get('status')!r}"
    return None


def _validate_search(payload: Any) -> str | None:
    if not isinstance(payload, list):
        return "/v1/search did not return a list"
    if not payload:
        return "/v1/search returned no results for Avatar"
    first_title = str((payload[0] or {}).get("title") or "").lower()
    if "avatar" not in first_title:
        return f"/v1/search first result was {first_title!r}, expected Avatar"
    return None


def _validate_recommendations(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "/v1/recommendations/id did not return an object"
    recommendations = payload.get("recommendations")
    if not isinstance(recommendations, list):
        return "/v1/recommendations/id did not return recommendations"
    if len(recommendations) < 3:
        return f"/v1/recommendations/id returned {len(recommendations)} results, expected at least 3"
    bad_titles = {"small soldiers", "supergirl", "barbarella", "the last airbender"}
    returned_bad = sorted(
        str((item or {}).get("title") or "").strip()
        for item in recommendations
        if str((item or {}).get("title") or "").strip().lower() in bad_titles
    )
    if returned_bad:
        return "recommendations returned known drift titles: " + ", ".join(returned_bad)
    return None


CHECKS = (
    {
        "name": "health",
        "path": "/health",
        "validator": _validate_health,
    },
    {
        "name": "frontends",
        "path": "/v1/frontends/status?include_remote=false",
        "validator": _validate_frontends,
    },
    {
        "name": "slo",
        "path": "/v1/platform/slo?include_frontends=false",
        "validator": _validate_slo,
    },
    {
        "name": "search_avatar",
        "path": f"/v1/search?{urllib.parse.urlencode({'q': 'Avatar', 'limit': 3})}",
        "validator": _validate_search,
    },
    {
        "name": "recommend_avatar",
        "path": f"/v1/recommendations/id/19995?{urllib.parse.urlencode({'n': 6})}",
        "validator": _validate_recommendations,
    },
)


def evaluate_base_url(base_url: str, *, timeout: int, include_recommendations: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for check_config in CHECKS:
        if check_config["name"] == "recommend_avatar" and not include_recommendations:
            checks.append(
                {
                    "name": check_config["name"],
                    "status": "skipped",
                    "reason": "disabled by synthetic monitor profile",
                }
            )
            continue
        started = time.perf_counter()
        check = {
            "name": check_config["name"],
            "path": check_config["path"],
            "status": "ok",
            "latency_ms": None,
        }
        try:
            payload = _get_json(base_url, check_config["path"], timeout)
            check["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            error = check_config["validator"](payload)
            if error:
                _fail(check, error)
        except Exception as exc:  # pragma: no cover - exercised by live workflow
            check["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            _fail(check, str(exc))
        if check["status"] == "failed":
            failures.append(f"{check['name']}: {check.get('error')}")
        checks.append(check)

    return {
        "base_url": base_url,
        "status": "failed" if failures else "ok",
        "failures": failures,
        "checks": checks,
    }


def evaluate_synthetic_monitor(args: argparse.Namespace) -> dict[str, Any]:
    targets = [
        evaluate_base_url(
            base_url,
            timeout=args.timeout,
            include_recommendations=not args.skip_recommendations,
        )
        for base_url in args.base_url
    ]
    failures = [failure for target in targets for failure in target["failures"]]
    return {
        "status": "failed" if failures else "ok",
        "generated_at": datetime.now(UTC).isoformat(),
        "targets": targets,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", action="append", required=True, help="Base API URL to probe. Repeat for failover targets.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("reports/synthetic_monitor.json"))
    parser.add_argument("--skip-recommendations", action="store_true", help="Skip the heavier recommendation path.")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_synthetic_monitor(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "failures": report["failures"]}, indent=2))
    if args.fail_on_error and report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
