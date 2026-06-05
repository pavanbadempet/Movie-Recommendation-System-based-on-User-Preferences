"""Serving artifact diagnostics and reload API routes."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from backend.auth import TenantContext

logger = logging.getLogger(__name__)


def create_artifact_router(
    *,
    resolve_tenant_context: Callable[..., TenantContext],
    resolve_admin_token: Callable[..., None],
    evaluate_artifact_health: Callable[..., dict[str, Any]],
    record_usage: Callable[..., Any],
    reload_local_recommender: Callable[..., Any],
    refresh_artifact_files: Callable[..., Any],
    serving_lineage: Callable[..., dict[str, Any]],
    current_recommender: Callable[[], Any],
) -> APIRouter:
    """Build artifact routes with injected runtime dependencies."""
    router = APIRouter(tags=["Artifacts"])

    @router.get("/v1/artifacts/health")
    async def artifact_health_report(
        context: TenantContext = Depends(resolve_tenant_context),
    ):
        from backend import recommender as recommender_module

        report = evaluate_artifact_health(
            models_dir=recommender_module.MODELS_DIR,
            data_dir=recommender_module.DATA_DIR,
        )
        record_usage(
            "artifacts.health",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return report

    @router.post("/v1/artifacts/reload")
    async def artifact_reload(
        force_download: bool = Query(default=True),
        load: bool = Query(default=True),
        _admin_token: None = Depends(resolve_admin_token),
    ):
        from backend import recommender as recommender_module

        try:
            if load:
                rec = await run_in_threadpool(lambda: reload_local_recommender(force_download=force_download))
                download_results = None
                lineage = serving_lineage(rec)
            else:
                download_results = await run_in_threadpool(
                    lambda: refresh_artifact_files(force_download=force_download)
                )
                lineage = serving_lineage(current_recommender())

            report = await run_in_threadpool(
                lambda: evaluate_artifact_health(
                    models_dir=recommender_module.MODELS_DIR,
                    data_dir=recommender_module.DATA_DIR,
                )
            )
            record_usage(
                "artifacts.reload",
                tenant_id="admin",
                catalog_id="serving",
                plan="internal",
                authenticated=True,
                status=str(report.get("status") or "unknown"),
            )
            return {
                "status": "reloaded" if load else "refreshed",
                "force_download": force_download,
                "loaded": load,
                "download_results": download_results,
                "artifact_health": report,
                "lineage": lineage,
            }
        except Exception as exc:
            logger.exception("Artifact reload failed")
            record_usage(
                "artifacts.reload",
                tenant_id="admin",
                catalog_id="serving",
                plan="internal",
                authenticated=True,
                status="error",
            )
            raise HTTPException(status_code=503, detail=f"Artifact reload failed: {exc}") from exc

    return router
