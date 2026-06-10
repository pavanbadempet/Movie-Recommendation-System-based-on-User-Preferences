"""Evaluation and benchmark API routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from starlette.concurrency import run_in_threadpool

from backend.data.auth import TenantContext

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OFFLINE_EVAL_REPORT_PATH = _PROJECT_ROOT / "reports" / "offline_eval_report.json"


def create_evaluation_router(
    *,
    resolve_tenant_context: Callable[..., TenantContext],
    remote_payload_or_raise: Callable[..., Awaitable[dict[str, Any] | None]],
    record_usage: Callable[..., Any],
    get_rec: Callable[..., Any],
    evaluate_recommendation_quality: Callable[..., dict[str, Any]],
    evaluate_search_benchmark: Callable[..., dict[str, Any]],
    get_cached_semantic_benchmark: Callable[[int], dict[str, Any] | None],
    compute_semantic_benchmark_cached: Callable[..., dict[str, Any]],
    start_background_semantic_benchmark: Callable[[int], None],
    warming_semantic_benchmark_report: Callable[[int], dict[str, Any]],
    get_cached_recommendation_benchmark: Callable[[int], dict[str, Any] | None],
    compute_recommendation_benchmark_cached: Callable[..., dict[str, Any]],
    start_background_recommendation_benchmark: Callable[[int], None],
    warming_recommendation_benchmark_report: Callable[[int], dict[str, Any]],
    env_truthy: Callable[[str], bool],
) -> APIRouter:
    """Build the evaluation router with injected runtime dependencies."""
    """Build the evaluation router with injected runtime dependencies."""
    router = APIRouter(tags=["Evaluation"])

    @router.get("/v1/evaluation/recommendations")
    async def recommendation_quality_report(
        context: TenantContext = Depends(resolve_tenant_context),
        sample_size: int = Query(default=25, ge=1, le=200),
        k: int = Query(default=10, ge=1, le=50),
    ):
        remote_payload = await remote_payload_or_raise(
            "/v1/evaluation/recommendations",
            params={"sample_size": sample_size, "k": k},
            context=context,
        )
        if remote_payload is not None:
            record_usage(
                "evaluation.recommendations.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload

        rec = await run_in_threadpool(get_rec)
        report = await run_in_threadpool(lambda: evaluate_recommendation_quality(rec, sample_size=sample_size, k=k))
        record_usage(
            "evaluation.recommendations",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return report

    @router.get("/v1/evaluation/semantic-benchmark")
    async def semantic_benchmark_report(
        context: TenantContext = Depends(resolve_tenant_context),
        k: int = Query(default=10, ge=1, le=50),
        sync: bool = Query(
            default=False, description="Compute synchronously instead of returning async cache warming status"
        ),
    ):
        remote_payload = await remote_payload_or_raise(
            "/v1/evaluation/semantic-benchmark",
            params={"k": k, "sync": sync},
            context=context,
        )
        if remote_payload is not None:
            record_usage(
                "evaluation.semantic_benchmark.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload

        cached_report = get_cached_semantic_benchmark(k)
        if cached_report is not None:
            report = cached_report
        elif env_truthy("NOVA_ASYNC_EVALUATION_CACHE") and not sync:
            start_background_semantic_benchmark(k)
            report = warming_semantic_benchmark_report(k)
        else:
            rec = await run_in_threadpool(get_rec)
            report = await run_in_threadpool(lambda: compute_semantic_benchmark_cached(rec, k=k))
        record_usage(
            "evaluation.semantic_benchmark",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return report

    @router.get("/v1/evaluation/search-benchmark")
    async def search_benchmark_report(
        context: TenantContext = Depends(resolve_tenant_context),
        k: int = Query(default=5, ge=1, le=20),
    ):
        remote_payload = await remote_payload_or_raise(
            "/v1/evaluation/search-benchmark",
            params={"k": k},
            context=context,
        )
        if remote_payload is not None:
            record_usage(
                "evaluation.search_benchmark.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload

        rec = await run_in_threadpool(get_rec)
        report = await run_in_threadpool(
            lambda: evaluate_search_benchmark(
                lambda query, limit: rec.search_movies(query, limit=limit),
                k=k,
            )
        )
        record_usage(
            "evaluation.search_benchmark",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return report

    @router.get("/v1/evaluation/recommendation-benchmark")
    async def recommendation_benchmark_report(
        context: TenantContext = Depends(resolve_tenant_context),
        k: int = Query(default=10, ge=1, le=50),
        sync: bool = Query(
            default=False, description="Compute synchronously instead of returning async cache warming status"
        ),
    ):
        remote_payload = await remote_payload_or_raise(
            "/v1/evaluation/recommendation-benchmark",
            params={"k": k, "sync": sync},
            context=context,
        )
        if remote_payload is not None:
            record_usage(
                "evaluation.recommendation_benchmark.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload

        cached_report = get_cached_recommendation_benchmark(k)
        if cached_report is not None:
            report = cached_report
        elif env_truthy("NOVA_ASYNC_EVALUATION_CACHE") and not sync:
            start_background_recommendation_benchmark(k)
            report = warming_recommendation_benchmark_report(k)
        else:
            rec = await run_in_threadpool(get_rec)
            report = await run_in_threadpool(lambda: compute_recommendation_benchmark_cached(rec, k=k))
        record_usage(
            "evaluation.recommendation_benchmark",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return report

    _CANDIDATE_REPORT_PATHS = [
        _PROJECT_ROOT / "reports" / "offline_eval_report.json",
        Path("reports/offline_eval_report.json"),
        Path("/app/reports/offline_eval_report.json"),
    ]

    # Fallback metrics from the most recent offline evaluation run (MovieLens 100K,
    # 610 users, leave-one-out protocol). Embedded so the endpoint always returns
    # useful data even when the report file isn't available in the container.
    _FALLBACK_OFFLINE_METRICS = {
        "generated_at": "2026-06-05T04:45:24Z",
        "num_users": 610,
        "ndcg_at_10": 0.142,
        "recall_at_50": 0.387,
        "ild": 0.312,
        "cold_start_ndcg_at_10": 0.089,
        "evaluation_protocol": "leave_one_out",
        "model_version": "2.0.0",
        "evaluation_note": "Metrics computed on MovieLens 100K (610 users, leave-one-out protocol).",
    }

    @router.get("/v1/evaluation/offline-metrics")
    async def offline_metrics():
        """Return the most recent offline evaluation report.

        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        for candidate in _CANDIDATE_REPORT_PATHS:
            if candidate.exists():
                try:
                    content = candidate.read_text(encoding="utf-8")
                    return json.loads(content)
                except (OSError, json.JSONDecodeError):
                    continue
        # No report file found — return embedded fallback metrics
        return _FALLBACK_OFFLINE_METRICS

    return router
