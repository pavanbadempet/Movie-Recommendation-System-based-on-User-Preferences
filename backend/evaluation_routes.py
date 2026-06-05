"""Evaluation and benchmark API routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from backend.auth import TenantContext

_OFFLINE_EVAL_REPORT_PATH = Path("reports/offline_eval_report.json")


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

    @router.get("/v1/evaluation/offline-metrics")
    async def offline_metrics():
        """Return the most recent offline evaluation report.

        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        if not _OFFLINE_EVAL_REPORT_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=("Offline evaluation has not been run yet. Execute scripts/run_offline_evaluation.py first."),
            )
        try:
            content = _OFFLINE_EVAL_REPORT_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read offline eval report: {exc}",
            ) from exc
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Offline eval report contains malformed JSON: {exc}",
            ) from exc

    return router
