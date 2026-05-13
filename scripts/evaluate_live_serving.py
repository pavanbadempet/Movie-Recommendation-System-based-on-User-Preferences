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


def _parse_title_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _canonical_title(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


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
            "recommendation_smoke": {},
            "semantic_benchmark": {},
        }
        try:
            report["health"] = _get_json(args.base_url, "/health", args.timeout)
            report["artifact_health"] = _get_json(args.base_url, "/v1/artifacts/health", args.timeout)
            search_params = urllib.parse.urlencode({"q": args.search_query, "limit": args.search_limit})
            report["search_smoke"] = _get_json(args.base_url, f"/v1/search?{search_params}", args.timeout)
            recommendation_params = urllib.parse.urlencode({"n": args.recommendation_smoke_k})
            report["recommendation_smoke"] = _get_json(
                args.base_url,
                f"/v1/recommendations/id/{args.recommendation_smoke_movie_id}?{recommendation_params}",
                args.timeout,
            )
            if args.skip_semantic_benchmark:
                report["semantic_benchmark"] = {"status": "skipped", "reason": "disabled by live gate"}
            else:
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
        accepted_artifact_statuses = {"ready"}
        if args.allow_degraded_artifact_health:
            accepted_artifact_statuses.add("degraded")
        if artifact_status not in accepted_artifact_statuses:
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
        if isinstance(search_results, list):
            search_titles = [_canonical_title((item or {}).get("title")) for item in search_results if isinstance(item, dict)]
            required_search_titles = {_canonical_title(title) for title in _parse_title_csv(args.required_search_titles)}
            required_search_hits = sorted({title for title in search_titles if title in required_search_titles})
            report["search_smoke_summary"] = {
                "required_hit_count": len(required_search_hits),
                "required_hits": required_search_hits,
            }
            if len(required_search_hits) < args.min_required_search_hits:
                _threshold_failure(
                    f"/v1/search found {len(required_search_hits)} required title hits, "
                    f"below {args.min_required_search_hits}",
                    report,
                )

        recommendation_payload = report.get("recommendation_smoke")
        if not isinstance(recommendation_payload, dict):
            _threshold_failure("/v1/recommendations/id did not return an object", report)
            recommendation_results: list[dict[str, Any]] = []
        else:
            raw_recommendations = recommendation_payload.get("recommendations")
            if not isinstance(raw_recommendations, list):
                _threshold_failure("/v1/recommendations/id did not return a recommendations list", report)
                recommendation_results = []
            else:
                recommendation_results = [item for item in raw_recommendations if isinstance(item, dict)]

        if len(recommendation_results) < args.min_recommendation_results:
            _threshold_failure(
                f"/v1/recommendations/id returned {len(recommendation_results)} results, "
                f"below {args.min_recommendation_results}",
                report,
            )

        recommendation_titles = [_canonical_title(item.get("title")) for item in recommendation_results]
        required_titles = {_canonical_title(title) for title in _parse_title_csv(args.required_recommendation_titles)}
        blocked_titles = {_canonical_title(title) for title in _parse_title_csv(args.blocked_recommendation_titles)}
        required_hits = sorted({title for title in recommendation_titles if title in required_titles})
        blocked_hits = sorted({title for title in recommendation_titles if title in blocked_titles})
        report["recommendation_smoke_summary"] = {
            "result_count": len(recommendation_results),
            "titles": [item.get("title") for item in recommendation_results],
            "required_hit_count": len(required_hits),
            "required_hits": required_hits,
            "blocked_hits": blocked_hits,
        }
        if len(required_hits) < args.min_required_recommendation_hits:
            _threshold_failure(
                f"recommendation smoke found {len(required_hits)} required semantic hits, "
                f"below {args.min_required_recommendation_hits}",
                report,
            )
        if blocked_hits:
            _threshold_failure(
                "recommendation smoke returned blocked drift titles: " + ", ".join(blocked_hits),
                report,
            )

        benchmark = report["semantic_benchmark"]
        if not args.skip_semantic_benchmark:
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
    parser.add_argument("--required-search-titles", default="Avatar: Fire and Ash,Avatar: The Way of Water")
    parser.add_argument("--min-required-search-hits", type=int, default=2)
    parser.add_argument("--recommendation-smoke-movie-id", type=int, default=19995)
    parser.add_argument("--recommendation-smoke-k", type=int, default=10)
    parser.add_argument("--min-recommendation-results", type=int, default=5)
    parser.add_argument(
        "--required-recommendation-titles",
        default=(
            "Avatar: The Way of Water,Avatar: Fire and Ash,The Abyss,Pacific Rim,Dune,"
            "The Creator,John Carter,Valerian and the City of a Thousand Planets"
        ),
    )
    parser.add_argument("--min-required-recommendation-hits", type=int, default=2)
    parser.add_argument(
        "--blocked-recommendation-titles",
        default=(
            "Small Soldiers,Supergirl,Barbarella,Kids Next Door: Operation Z.E.R.O.,"
            "The Last Airbender,X-Men: Apocalypse,Knights of the Zodiac,Mystery Men,"
            "Justice League: The Flashpoint Paradox"
        ),
    )
    parser.add_argument("--max-bad-match-rate", type=float, default=0.05)
    parser.add_argument("--min-hit-rate", type=float, default=0.95)
    parser.add_argument("--min-mrr", type=float, default=0.35)
    parser.add_argument("--min-ndcg", type=float, default=0.25)
    parser.add_argument("--min-explanation-coverage", type=float, default=0.90)
    parser.add_argument("--skip-semantic-benchmark", action="store_true")
    parser.add_argument("--allow-degraded-artifact-health", action="store_true")
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
        "app_commit": (report.get("health") or {}).get("app_commit"),
        "app_version": (report.get("health") or {}).get("app_version"),
        "movie_count": (report.get("health") or {}).get("movie_count"),
        "search_result_count": len(report.get("search_smoke") or []) if isinstance(report.get("search_smoke"), list) else None,
        "search_first_title": (
            (report.get("search_smoke") or [{}])[0].get("title")
            if isinstance(report.get("search_smoke"), list) and report.get("search_smoke")
            else None
        ),
        "search_required_hit_count": (report.get("search_smoke_summary") or {}).get("required_hit_count"),
        "recommendation_result_count": (report.get("recommendation_smoke_summary") or {}).get("result_count"),
        "recommendation_required_hit_count": (report.get("recommendation_smoke_summary") or {}).get("required_hit_count"),
        "recommendation_blocked_hits": (report.get("recommendation_smoke_summary") or {}).get("blocked_hits"),
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
