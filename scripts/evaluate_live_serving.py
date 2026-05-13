"""Evaluate the deployed Nova serving API.

This is a production smoke gate: it checks live service health, artifact
alignment, and human-labeled search/recommendation/semantic benchmark metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.search_benchmark import DEFAULT_SEARCH_BENCHMARK_PATH, evaluate_search_benchmark


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
            "search_benchmark": {},
            "recommendation_smoke": {},
            "recommendation_benchmark": {},
            "semantic_benchmark": {},
        }
        try:
            report["health"] = _get_json(args.base_url, "/health", args.timeout)
            report["artifact_health"] = _get_json(args.base_url, "/v1/artifacts/health", args.timeout)
            search_params = urllib.parse.urlencode({"q": args.search_query, "limit": args.search_limit})
            report["search_smoke"] = _get_json(args.base_url, f"/v1/search?{search_params}", args.timeout)
            if args.skip_search_benchmark:
                report["search_benchmark"] = {"status": "skipped", "reason": "disabled by live gate"}
            elif args.search_benchmark_mode == "endpoint":
                benchmark_params = urllib.parse.urlencode({"k": args.search_benchmark_k})
                report["search_benchmark"] = _get_json(
                    args.base_url,
                    f"/v1/evaluation/search-benchmark?{benchmark_params}",
                    args.timeout,
                )
            else:
                report["search_benchmark"] = evaluate_search_benchmark(
                    lambda query, limit: _get_json(
                        args.base_url,
                        f"/v1/search?{urllib.parse.urlencode({'q': query, 'limit': limit})}",
                        args.timeout,
                    ),
                    benchmark_path=args.search_benchmark_path,
                    k=args.search_benchmark_k,
                )
            recommendation_params = urllib.parse.urlencode({"n": args.recommendation_smoke_k})
            report["recommendation_smoke"] = _get_json(
                args.base_url,
                f"/v1/recommendations/id/{args.recommendation_smoke_movie_id}?{recommendation_params}",
                args.timeout,
            )
            if args.skip_recommendation_benchmark:
                report["recommendation_benchmark"] = {"status": "skipped", "reason": "disabled by live gate"}
            else:
                report["recommendation_benchmark"] = _get_json(
                    args.base_url,
                    f"/v1/evaluation/recommendation-benchmark?k={args.k}",
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

        expected_commit = str(args.expected_app_commit or "").strip()
        actual_commit = str(report["health"].get("app_commit") or "").strip()
        if expected_commit:
            if not actual_commit:
                _threshold_failure("/health did not expose app_commit for revision verification", report)
            elif not expected_commit.startswith(actual_commit) and not actual_commit.startswith(expected_commit):
                _threshold_failure(
                    f"/health app_commit {actual_commit!r} does not match expected {expected_commit[:12]!r}",
                    report,
                )

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

        search_benchmark = report.get("search_benchmark") or {}
        if not args.skip_search_benchmark:
            if search_benchmark.get("status") not in {"ok", "needs_attention"}:
                _threshold_failure(
                    f"search benchmark unavailable: {search_benchmark.get('reason') or search_benchmark.get('status')}",
                    report,
                )
            search_metrics = search_benchmark.get("metrics") or {}
            search_checks = {
                "search_top1_hit_rate": (
                    float(search_metrics.get("top1_hit_rate") or 0.0),
                    ">=",
                    args.min_search_top1_hit_rate,
                ),
                "search_hit_rate_at_k": (
                    float(search_metrics.get("hit_rate_at_k") or 0.0),
                    ">=",
                    args.min_search_hit_rate,
                ),
                "search_blocked_hit_case_rate": (
                    float(search_metrics.get("blocked_hit_case_rate") or 0.0),
                    "<=",
                    args.max_search_blocked_hit_case_rate,
                ),
            }
            for name, (actual, op, expected) in search_checks.items():
                if op == "<=" and actual > expected:
                    _threshold_failure(f"{name} {actual} exceeds {expected}", report)
                if op == ">=" and actual < expected:
                    _threshold_failure(f"{name} {actual} below {expected}", report)

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

        recommendation_benchmark = report.get("recommendation_benchmark") or {}
        if not args.skip_recommendation_benchmark:
            if recommendation_benchmark.get("status") not in {"ok", "needs_attention"}:
                _threshold_failure(
                    "recommendation benchmark unavailable: "
                    f"{recommendation_benchmark.get('reason') or recommendation_benchmark.get('status')}",
                    report,
                )
            recommendation_metrics = recommendation_benchmark.get("metrics") or {}
            recommendation_checks = {
                "recommendation_benchmark_case_pass_rate": (
                    float(recommendation_metrics.get("case_pass_rate") or 0.0),
                    ">=",
                    args.min_recommendation_benchmark_pass_rate,
                ),
                "recommendation_benchmark_good_hit_case_rate": (
                    float(recommendation_metrics.get("good_hit_case_rate") or 0.0),
                    ">=",
                    args.min_recommendation_benchmark_hit_rate,
                ),
                "recommendation_benchmark_bad_case_rate_at_k": (
                    float(recommendation_metrics.get("bad_case_rate_at_k") or 0.0),
                    "<=",
                    args.max_recommendation_benchmark_bad_case_rate,
                ),
            }
            for name, (actual, op, expected) in recommendation_checks.items():
                if op == "<=" and actual > expected:
                    _threshold_failure(f"{name} {actual} exceeds {expected}", report)
                if op == ">=" and actual < expected:
                    _threshold_failure(f"{name} {actual} below {expected}", report)

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
    parser.add_argument("--expected-app-commit", default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--search-query", default="Avatar")
    parser.add_argument("--search-limit", type=int, default=5)
    parser.add_argument("--min-search-results", type=int, default=1)
    parser.add_argument("--expected-search-title", default="Avatar")
    parser.add_argument("--required-search-titles", default="Avatar: Fire and Ash,Avatar: The Way of Water")
    parser.add_argument("--min-required-search-hits", type=int, default=2)
    parser.add_argument("--search-benchmark-path", type=Path, default=DEFAULT_SEARCH_BENCHMARK_PATH)
    parser.add_argument("--search-benchmark-k", type=int, default=5)
    parser.add_argument("--search-benchmark-mode", choices=["endpoint", "client"], default="endpoint")
    parser.add_argument("--min-search-top1-hit-rate", type=float, default=0.98)
    parser.add_argument("--min-search-hit-rate", type=float, default=1.0)
    parser.add_argument("--max-search-blocked-hit-case-rate", type=float, default=0.0)
    parser.add_argument("--skip-search-benchmark", action="store_true")
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
    parser.add_argument("--min-recommendation-benchmark-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-recommendation-benchmark-hit-rate", type=float, default=0.90)
    parser.add_argument("--max-recommendation-benchmark-bad-case-rate", type=float, default=0.0)
    parser.add_argument("--skip-recommendation-benchmark", action="store_true")
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
    search_benchmark_metrics = (report.get("search_benchmark") or {}).get("metrics") or {}
    recommendation_benchmark_metrics = (report.get("recommendation_benchmark") or {}).get("metrics") or {}
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
        "search_benchmark_top1_hit_rate": search_benchmark_metrics.get("top1_hit_rate"),
        "search_benchmark_hit_rate_at_k": search_benchmark_metrics.get("hit_rate_at_k"),
        "search_benchmark_blocked_hit_case_rate": search_benchmark_metrics.get("blocked_hit_case_rate"),
        "search_benchmark_case_count": (report.get("search_benchmark") or {}).get("evaluated_case_count"),
        "recommendation_result_count": (report.get("recommendation_smoke_summary") or {}).get("result_count"),
        "recommendation_required_hit_count": (report.get("recommendation_smoke_summary") or {}).get("required_hit_count"),
        "recommendation_blocked_hits": (report.get("recommendation_smoke_summary") or {}).get("blocked_hits"),
        "recommendation_benchmark_case_count": (report.get("recommendation_benchmark") or {}).get("evaluated_case_count"),
        "recommendation_benchmark_case_pass_rate": recommendation_benchmark_metrics.get("case_pass_rate"),
        "recommendation_benchmark_good_hit_case_rate": recommendation_benchmark_metrics.get("good_hit_case_rate"),
        "recommendation_benchmark_bad_case_rate_at_k": recommendation_benchmark_metrics.get("bad_case_rate_at_k"),
        "recommendation_benchmark_mrr_at_k": recommendation_benchmark_metrics.get("mrr_at_k"),
        "recommendation_benchmark_ndcg_at_k": recommendation_benchmark_metrics.get("ndcg_at_k"),
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
