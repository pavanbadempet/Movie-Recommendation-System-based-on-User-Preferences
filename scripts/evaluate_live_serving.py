"""Evaluate the deployed Nova serving API.

This is a production smoke gate: it checks the live service health, artifact
alignment, and human-labeled semantic benchmark metrics.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _get_json(base_url: str, path: str, timeout: int) -> Any:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    request = urllib.request.Request(url, headers={"User-Agent": "nova-serving-quality-gate"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _threshold_failure(message: str, report: dict[str, Any]) -> None:
    report["status"] = "failed"
    report.setdefault("failures", []).append(message)


def evaluate_live_serving(args: argparse.Namespace) -> dict[str, Any]:
    last_report: dict[str, Any] | None = None
    last_error = None
    for attempt in range(1, args.retries + 1):
        report: dict[str, Any] = {
            "base_url": args.base_url,
            "status": "ok",
            "attempt": attempt,
            "failures": [],
            "health": {},
            "artifact_health": {},
            "search_smoke": [],
            "semantic_benchmark": {},
        }
        try:
            report["health"] = _get_json(args.base_url, "/health", args.timeout)
            report["artifact_health"] = _get_json(args.base_url, "/v1/artifacts/health", args.timeout)
            search_params = urllib.parse.urlencode({"q": args.search_query, "limit": args.search_limit})
            report["search_smoke"] = _get_json(args.base_url, f"/v1/search?{search_params}", args.timeout)
            report["semantic_benchmark"] = _get_json(
                args.base_url,
                f"/v1/evaluation/semantic-benchmark?k={args.k}",
                args.timeout,
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt < args.retries:
                time.sleep(args.retry_delay_seconds)
                continue
            _threshold_failure(f"live API did not respond after {args.retries} attempts: {last_error}", report)
            return report

        if report["health"].get("status") != "healthy":
            _threshold_failure(f"/health status is {report['health'].get('status')}", report)

        artifact_status = report["artifact_health"].get("status")
        if artifact_status != "ready":
            _threshold_failure(f"/v1/artifacts/health status is {artifact_status}", report)

        search_results = report.get("search_smoke")
        if not isinstance(search_results, list):
            _threshold_failure("/v1/search did not return a list", report)
        elif len(search_results) < args.min_search_results:
            _threshold_failure(
                f"/v1/search returned {len(search_results)} results, below {args.min_search_results}",
                report,
            )
        elif args.expected_search_title:
            first_title = str((search_results[0] or {}).get("title") or "").lower()
            expected_title = args.expected_search_title.lower()
            if expected_title not in first_title:
                _threshold_failure(
                    f"/v1/search first result {first_title!r} does not contain {expected_title!r}",
                    report,
                )

        benchmark = report["semantic_benchmark"]
        if benchmark.get("status") not in {"ok", "needs_attention"}:
            _threshold_failure(f"semantic benchmark unavailable: {benchmark.get('reason') or benchmark.get('status')}", report)

        metrics = benchmark.get("metrics") or {}
        checks = {
            "bad_match_rate_at_k": (float(metrics.get("bad_match_rate_at_k") or 0.0), "<=", args.max_bad_match_rate),
            "hit_rate_at_k": (float(metrics.get("hit_rate_at_k") or 0.0), ">=", args.min_hit_rate),
            "mrr_at_k": (float(metrics.get("mrr_at_k") or 0.0), ">=", args.min_mrr),
            "ndcg_at_k": (float(metrics.get("ndcg_at_k") or 0.0), ">=", args.min_ndcg),
            "explanation_coverage": (float(metrics.get("explanation_coverage") or 0.0), ">=", args.min_explanation_coverage),
        }
        for name, (actual, op, expected) in checks.items():
            if op == "<=" and actual > expected:
                _threshold_failure(f"{name} {actual} exceeds {expected}", report)
            if op == ">=" and actual < expected:
                _threshold_failure(f"{name} {actual} below {expected}", report)

        if report.get("status") == "ok":
            return report
        last_report = report
        if attempt < args.retries:
            time.sleep(args.retry_delay_seconds)
    else:
        return last_report or {
            "base_url": args.base_url,
            "status": "failed",
            "failures": ["live serving quality gate did not produce a report"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://pavanbadempet-movie-rec-api.hf.space")
    parser.add_argument("--output", type=Path, default=Path("reports/live_serving_quality.json"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--retry-delay-seconds", type=int, default=30)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--search-query", default="Avatar")
    parser.add_argument("--search-limit", type=int, default=5)
    parser.add_argument("--min-search-results", type=int, default=1)
    parser.add_argument("--expected-search-title", default="Avatar")
    parser.add_argument("--max-bad-match-rate", type=float, default=0.05)
    parser.add_argument("--min-hit-rate", type=float, default=0.95)
    parser.add_argument("--min-mrr", type=float, default=0.35)
    parser.add_argument("--min-ndcg", type=float, default=0.25)
    parser.add_argument("--min-explanation-coverage", type=float, default=0.90)
    parser.add_argument("--fail-on-threshold", action="store_true")
    args = parser.parse_args()

    report = evaluate_live_serving(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    benchmark_metrics = (report.get("semantic_benchmark") or {}).get("metrics") or {}
    summary = {
        "status": report.get("status"),
        "failures": report.get("failures"),
        "artifact_status": (report.get("artifact_health") or {}).get("status"),
        "movie_count": (report.get("health") or {}).get("movie_count"),
        "search_result_count": len(report.get("search_smoke") or []) if isinstance(report.get("search_smoke"), list) else None,
        "search_first_title": (
            (report.get("search_smoke") or [{}])[0].get("title")
            if isinstance(report.get("search_smoke"), list) and report.get("search_smoke")
            else None
        ),
        "hit_rate_at_k": benchmark_metrics.get("hit_rate_at_k"),
        "mrr_at_k": benchmark_metrics.get("mrr_at_k"),
        "ndcg_at_k": benchmark_metrics.get("ndcg_at_k"),
        "bad_match_rate_at_k": benchmark_metrics.get("bad_match_rate_at_k"),
        "explanation_coverage": benchmark_metrics.get("explanation_coverage"),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.fail_on_threshold and report.get("status") != "ok":
        raise SystemExit("; ".join(report.get("failures") or ["live serving quality gate failed"]))


if __name__ == "__main__":
    main()
