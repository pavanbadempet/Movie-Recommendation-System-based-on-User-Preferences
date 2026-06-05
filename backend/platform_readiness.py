"""
Platform readiness report helpers — extracted from backend/main.py.

Provides readiness component builders and the _platform_readiness_report function.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging

logger = logging.getLogger(__name__)


def readiness_component(
    *,
    name: str,
    status: str,
    summary: str,
    required: bool = True,
    details: dict | None = None,
) -> dict:
    return {
        "name": name,
        "status": status,
        "required": required,
        "summary": summary,
        "details": details or {},
    }


def benchmark_readiness_component(
    *,
    name: str,
    report: dict | None,
    required: bool,
    thresholds: dict[str, tuple[str, float]],
    safe_float_fn,
) -> dict:
    if not report:
        return readiness_component(
            name=name,
            status="warming",
            required=required,
            summary="Benchmark cache is not ready yet.",
        )

    report_status = str(report.get("status") or "unknown")
    if report_status == "warming":
        return readiness_component(
            name=name,
            status="warming",
            required=required,
            summary=str(report.get("reason") or "Benchmark is warming."),
        )
    if report_status not in {"ok", "needs_attention"}:
        return readiness_component(
            name=name,
            status="failed",
            required=True,
            summary=f"Benchmark status is {report_status}.",
            details={"reason": report.get("reason")},
        )

    metrics = report.get("metrics") or {}
    failures = []
    for metric, (op, expected) in thresholds.items():
        actual = safe_float_fn(metrics.get(metric)) or 0.0
        if op == ">=" and actual < expected:
            failures.append({"metric": metric, "actual": actual, "expected": expected, "operator": op})
        if op == "<=" and actual > expected:
            failures.append({"metric": metric, "actual": actual, "expected": expected, "operator": op})

    if failures:
        return readiness_component(
            name=name,
            status="failed",
            required=True,
            summary="Benchmark metrics are below readiness thresholds.",
            details={"failures": failures, "metrics": metrics},
        )

    return readiness_component(
        name=name,
        status="ok",
        required=required,
        summary="Benchmark metrics satisfy readiness thresholds.",
        details={
            "status": report_status,
            "evaluated_case_count": report.get("evaluated_case_count"),
            "metrics": metrics,
        },
    )


def combine_readiness_status(components: list[dict], strict: bool) -> str:
    required = [c for c in components if c.get("required")]
    bad = {"failed", "unavailable", "not_ready"}
    degraded = {"degraded", "warming", "missing"}
    if any(c.get("status") in bad for c in required):
        return "not_ready"
    if any(c.get("status") in degraded for c in required):
        return "degraded"
    return "ready"


def platform_readiness_report(
    *,
    context,
    rec,
    artifact_report: dict,
    behavior: dict,
    strict: bool,
    k: int,
    app_metadata_fn,
    safe_float_fn,
    serving_lineage_fn,
    get_cached_semantic_benchmark_fn,
    get_cached_recommendation_benchmark_fn,
    start_background_semantic_benchmark_fn,
    start_background_recommendation_benchmark_fn,
    env_truthy_fn,
) -> dict:
    lineage = serving_lineage_fn(rec)
    movie_count = len(rec.movies)
    components = []

    components.append(
        readiness_component(
            name="catalog",
            status="ok" if movie_count > 0 else "failed",
            summary=f"{movie_count:,} catalog items loaded." if movie_count > 0 else "No catalog items loaded.",
            details={"movie_count": movie_count},
        )
    )

    artifact_status = str(artifact_report.get("status") or "unknown")
    components.append(
        readiness_component(
            name="artifact_health",
            status="ok" if artifact_status == "ready" else "degraded" if artifact_status == "degraded" else "failed",
            summary=f"Artifact health is {artifact_status}.",
            details={
                "status": artifact_status,
                "checks": artifact_report.get("checks") or {},
                "recommendations": artifact_report.get("recommendations") or [],
            },
        )
    )

    vector_ready = lineage.get("vector_artifacts_ready") is True
    components.append(
        readiness_component(
            name="vector_serving",
            status="ok" if vector_ready else "degraded",
            summary="Vector artifacts are aligned and serving."
            if vector_ready
            else "Vector artifacts are not fully available.",
            details=lineage,
        )
    )

    search_status = "failed"
    search_details: dict = {}
    try:
        sample_movie = rec.get_movie_by_index(0)
        sample_results = rec.search_movies(str(sample_movie.get("title") or ""), limit=1)
        first_result = sample_results[0] if sample_results else {}
        search_status = "ok" if first_result.get("id") == sample_movie.get("id") else "degraded"
        search_details = {
            "query": sample_movie.get("title"),
            "expected_id": sample_movie.get("id"),
            "first_result_id": first_result.get("id"),
            "first_result_title": first_result.get("title"),
        }
    except Exception as exc:
        search_details = {"error": str(exc)}
    components.append(
        readiness_component(
            name="search_smoke",
            status=search_status,
            summary="Canonical title search returns the expected first item."
            if search_status == "ok"
            else "Canonical title search did not return the expected first item.",
            details=search_details,
        )
    )

    recommendation_status = "failed"
    recommendation_details: dict = {}
    try:
        sample_movie = rec.get_movie_by_index(0)
        recommendations = rec.recommend_by_id(int(sample_movie["id"]), n=min(5, max(1, k)))
        recommendation_status = "ok" if recommendations else "failed"
        recommendation_details = {
            "seed_id": sample_movie.get("id"),
            "seed_title": sample_movie.get("title"),
            "result_count": len(recommendations),
            "first_result_title": recommendations[0].get("title") if recommendations else None,
        }
    except Exception as exc:
        recommendation_details = {"error": str(exc)}
    components.append(
        readiness_component(
            name="recommendation_smoke",
            status=recommendation_status,
            summary="Item-to-item recommendations are returning results."
            if recommendation_status == "ok"
            else "Item-to-item recommendations are not returning results.",
            details=recommendation_details,
        )
    )

    sem_report = get_cached_semantic_benchmark_fn(k)
    rec_report = get_cached_recommendation_benchmark_fn(k)
    if env_truthy_fn("NOVA_ASYNC_EVALUATION_CACHE"):
        if sem_report is None:
            start_background_semantic_benchmark_fn(k)
        if rec_report is None:
            start_background_recommendation_benchmark_fn(k)

    components.append(
        benchmark_readiness_component(
            name="semantic_benchmark_cache",
            report=sem_report,
            required=strict,
            safe_float_fn=safe_float_fn,
            thresholds={
                "bad_match_rate_at_k": ("<=", 0.05),
                "hit_rate_at_k": (">=", 0.95),
                "mrr_at_k": (">=", 0.35),
                "ndcg_at_k": (">=", 0.25),
            },
        )
    )
    components.append(
        benchmark_readiness_component(
            name="recommendation_benchmark_cache",
            report=rec_report,
            required=strict,
            safe_float_fn=safe_float_fn,
            thresholds={
                "case_pass_rate": (">=", 0.80),
                "good_hit_case_rate": (">=", 0.90),
                "bad_case_rate_at_k": ("<=", 0.0),
            },
        )
    )

    ranker = getattr(rec, "_learned_ranker", None)
    components.append(
        readiness_component(
            name="learned_ranker",
            status="ok" if ranker is not None else "missing",
            required=False,
            summary="Learned ranker is loaded." if ranker is not None else "Learned ranker is optional and not loaded.",
            details=(getattr(ranker, "metadata", {}) or {}) if ranker is not None else {},
        )
    )
    components.append(
        readiness_component(
            name="event_store",
            status="ok",
            required=False,
            summary="Behavior event store is available for product analytics.",
            details={
                "mode": behavior.get("event_store"),
                "durable": behavior.get("durable"),
                "event_table": behavior.get("event_table"),
                "total_events": behavior.get("total_events"),
            },
        )
    )

    readiness_status = combine_readiness_status(components, strict=strict)
    return {
        "status": readiness_status,
        "strict": strict,
        "app": app_metadata_fn(),
        "tenant_id": context.tenant_id,
        "catalog_id": context.catalog_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "k": k,
        "lineage": lineage,
        "summary": {
            "component_count": len(components),
            "ok_count": sum(1 for c in components if c.get("status") == "ok"),
            "required_count": sum(1 for c in components if c.get("required")),
            "failed_required_count": sum(
                1 for c in components if c.get("required") and c.get("status") in {"failed", "unavailable", "not_ready"}
            ),
        },
        "components": components,
    }


# ---------------------------------------------------------------------------
# Private-named aliases used by backend/main.py (kept here after extraction)
# ---------------------------------------------------------------------------
from backend.recommender_helpers import safe_float as _safe_float


def _readiness_component(
    *,
    name: str,
    status: str,
    summary: str,
    required: bool = True,
    details: dict | None = None,
) -> dict:
    return readiness_component(
        name=name,
        status=status,
        summary=summary,
        required=required,
        details=details,
    )


def _benchmark_readiness_component(
    *,
    name: str,
    report: dict | None,
    required: bool,
    thresholds: dict[str, tuple[str, float]],
) -> dict:
    return benchmark_readiness_component(
        name=name,
        report=report,
        required=required,
        thresholds=thresholds,
        safe_float_fn=_safe_float,
    )


# ---------------------------------------------------------------------------
# Private-named wrappers used by backend/main.py (moved here from main.py)
# These preserve the exact call signatures that main.py and recommendation_routes.py expect.
# ---------------------------------------------------------------------------


def _combine_readiness_status(components: list[dict], strict: bool) -> str:
    """Private alias for combine_readiness_status — matches the signature used in main.py."""
    required_components = [component for component in components if component.get("required")]
    bad_statuses = {"failed", "unavailable", "not_ready"}
    degraded_statuses = {"degraded", "warming", "missing"}

    if any(component.get("status") in bad_statuses for component in required_components):
        return "not_ready"
    if strict and any(component.get("status") in degraded_statuses for component in required_components):
        return "degraded"
    if any(component.get("status") in degraded_statuses for component in required_components):
        return "degraded"
    return "ready"


def _platform_readiness_report(
    *,
    context,
    rec,
    artifact_report: dict,
    behavior: dict,
    strict: bool,
    k: int,
) -> dict:
    """Private alias for the inline readiness report builder — moved from backend/main.py."""
    from backend.app_info import app_metadata
    from backend.benchmark_cache import (
        get_cached_recommendation_benchmark,
        get_cached_semantic_benchmark,
        start_background_recommendation_benchmark,
        start_background_semantic_benchmark,
    )
    from backend.recommendation_events import _serving_lineage

    import os

    def _env_truthy(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    lineage = _serving_lineage(rec)
    movie_count = len(rec.movies)
    components = []

    components.append(
        _readiness_component(
            name="catalog",
            status="ok" if movie_count > 0 else "failed",
            summary=f"{movie_count:,} catalog items loaded." if movie_count > 0 else "No catalog items loaded.",
            details={"movie_count": movie_count},
        )
    )

    artifact_status = str(artifact_report.get("status") or "unknown")
    components.append(
        _readiness_component(
            name="artifact_health",
            status="ok" if artifact_status == "ready" else "degraded" if artifact_status == "degraded" else "failed",
            summary=f"Artifact health is {artifact_status}.",
            details={
                "status": artifact_status,
                "checks": artifact_report.get("checks") or {},
                "recommendations": artifact_report.get("recommendations") or [],
            },
        )
    )

    vector_ready = lineage.get("vector_artifacts_ready") is True
    components.append(
        _readiness_component(
            name="vector_serving",
            status="ok" if vector_ready else "degraded",
            summary="Vector artifacts are aligned and serving."
            if vector_ready
            else "Vector artifacts are not fully available.",
            details=lineage,
        )
    )

    search_status = "failed"
    search_details: dict = {}
    try:
        sample_movie = rec.get_movie_by_index(0)
        sample_results = rec.search_movies(str(sample_movie.get("title") or ""), limit=1)
        first_result = sample_results[0] if sample_results else {}
        search_status = "ok" if first_result.get("id") == sample_movie.get("id") else "degraded"
        search_details = {
            "query": sample_movie.get("title"),
            "expected_id": sample_movie.get("id"),
            "first_result_id": first_result.get("id"),
            "first_result_title": first_result.get("title"),
        }
    except Exception as exc:
        search_details = {"error": str(exc)}
    components.append(
        _readiness_component(
            name="search_smoke",
            status=search_status,
            summary="Canonical title search returns the expected first item."
            if search_status == "ok"
            else "Canonical title search did not return the expected first item.",
            details=search_details,
        )
    )

    recommendation_status = "failed"
    recommendation_details: dict = {}
    try:
        sample_movie = rec.get_movie_by_index(0)
        recommendations = rec.recommend_by_id(int(sample_movie["id"]), n=min(5, max(1, k)))
        recommendation_status = "ok" if recommendations else "failed"
        recommendation_details = {
            "seed_id": sample_movie.get("id"),
            "seed_title": sample_movie.get("title"),
            "result_count": len(recommendations),
            "first_result_title": recommendations[0].get("title") if recommendations else None,
        }
    except Exception as exc:
        recommendation_details = {"error": str(exc)}
    components.append(
        _readiness_component(
            name="recommendation_smoke",
            status=recommendation_status,
            summary="Item-to-item recommendations are returning results."
            if recommendation_status == "ok"
            else "Item-to-item recommendations are not returning results.",
            details=recommendation_details,
        )
    )

    semantic_benchmark_report = get_cached_semantic_benchmark(k)
    recommendation_benchmark_report = get_cached_recommendation_benchmark(k)
    if _env_truthy("NOVA_ASYNC_EVALUATION_CACHE"):
        if semantic_benchmark_report is None:
            start_background_semantic_benchmark(k)
        if recommendation_benchmark_report is None:
            start_background_recommendation_benchmark(k)

    components.append(
        _benchmark_readiness_component(
            name="semantic_benchmark_cache",
            report=semantic_benchmark_report,
            required=strict,
            thresholds={
                "bad_match_rate_at_k": ("<=", 0.05),
                "hit_rate_at_k": (">=", 0.95),
                "mrr_at_k": (">=", 0.35),
                "ndcg_at_k": (">=", 0.25),
            },
        )
    )
    components.append(
        _benchmark_readiness_component(
            name="recommendation_benchmark_cache",
            report=recommendation_benchmark_report,
            required=strict,
            thresholds={
                "case_pass_rate": (">=", 0.80),
                "good_hit_case_rate": (">=", 0.90),
                "bad_case_rate_at_k": ("<=", 0.0),
            },
        )
    )

    ranker = getattr(rec, "_learned_ranker", None)
    components.append(
        _readiness_component(
            name="learned_ranker",
            status="ok" if ranker is not None else "missing",
            required=False,
            summary="Learned ranker is loaded." if ranker is not None else "Learned ranker is optional and not loaded.",
            details=(getattr(ranker, "metadata", {}) or {}) if ranker is not None else {},
        )
    )
    components.append(
        _readiness_component(
            name="event_store",
            status="ok",
            required=False,
            summary="Behavior event store is available for product analytics.",
            details={
                "mode": behavior.get("event_store"),
                "durable": behavior.get("durable"),
                "event_table": behavior.get("event_table"),
                "total_events": behavior.get("total_events"),
            },
        )
    )

    readiness_status = _combine_readiness_status(components, strict=strict)
    return {
        "status": readiness_status,
        "strict": strict,
        "app": app_metadata(),
        "tenant_id": context.tenant_id,
        "catalog_id": context.catalog_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "k": k,
        "lineage": lineage,
        "summary": {
            "component_count": len(components),
            "ok_count": sum(1 for component in components if component.get("status") == "ok"),
            "required_count": sum(1 for component in components if component.get("required")),
            "failed_required_count": sum(
                1
                for component in components
                if component.get("required") and component.get("status") in {"failed", "unavailable", "not_ready"}
            ),
        },
        "components": components,
    }
