"""Evaluation and benchmark API routes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from backend.data.auth import TenantContext
from backend.router_deps import RouterDeps

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OFFLINE_EVAL_REPORT_PATH = _PROJECT_ROOT / "reports" / "offline_eval_report.json"


def load_offline_metrics(candidate_paths: list[Path]) -> dict:
    """Load a persisted offline evaluation report with explicit provenance."""
    errors: list[str] = []
    for candidate in candidate_paths:
        if not candidate.is_file():
            continue
        try:
            report = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if not isinstance(report, dict):
            errors.append(f"{candidate}: report root must be a JSON object")
            continue
        result = dict(report)
        result["provenance"] = {
            "source": "offline_evaluation_report",
            "report_path": str(candidate.resolve()),
        }
        return result
    detail = f" ({'; '.join(errors)})" if errors else ""
    raise FileNotFoundError(f"No valid offline evaluation report is available{detail}")


def create_evaluation_router(deps: RouterDeps) -> APIRouter:
    """Build the evaluation router with injected runtime dependencies."""
    resolve_tenant_context = deps.resolve_tenant_context
    remote_payload_or_raise = deps.remote_payload_or_raise
    record_usage = deps.record_usage
    get_rec = deps.get_rec
    evaluate_recommendation_quality = deps.evaluate_recommendation_quality
    evaluate_search_benchmark = deps.evaluate_search_benchmark
    get_cached_semantic_benchmark = deps.get_cached_semantic_benchmark
    compute_semantic_benchmark_cached = deps.compute_semantic_benchmark_cached
    start_background_semantic_benchmark = deps.start_background_semantic_benchmark
    warming_semantic_benchmark_report = deps.warming_semantic_benchmark_report
    get_cached_recommendation_benchmark = deps.get_cached_recommendation_benchmark
    compute_recommendation_benchmark_cached = deps.compute_recommendation_benchmark_cached
    start_background_recommendation_benchmark = deps.start_background_recommendation_benchmark
    warming_recommendation_benchmark_report = deps.warming_recommendation_benchmark_report
    env_truthy = deps.env_truthy
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

    @router.get("/v1/evaluation/offline-metrics")
    async def offline_metrics():
        """Return the most recent offline evaluation report.

        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        try:
            return load_offline_metrics(_CANDIDATE_REPORT_PATHS)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unavailable",
                    "reason": "offline_evaluation_report_missing",
                    "message": str(exc),
                },
            ) from exc

    return router
