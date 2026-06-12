"""Experiment assignment and metrics API routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, Query

from backend.data.auth import TenantContext


from backend.router_deps import RouterDeps


def create_experiment_router(deps: RouterDeps) -> APIRouter:
    """Build experiment routes with injected runtime dependencies."""
    resolve_tenant_context = deps.resolve_tenant_context
    assign_experiment = deps.assign_experiment
    summarize_experiment_metrics = deps.summarize_experiment_metrics
    record_usage = deps.record_usage
    router = APIRouter(tags=["Experiments"])

    @router.get("/v1/experiments/assignment")
    async def experiment_assignment(
        user_id: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
        experiment: str | None = Query(default=None),
        context: TenantContext = Depends(resolve_tenant_context),
    ):
        subject_id = user_id or session_id or f"{context.tenant_id}:{context.catalog_id}:anonymous"
        assignment = assign_experiment(subject_id=subject_id, experiment_name=experiment)
        record_usage(
            "experiments.assignment",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return assignment

    @router.get("/v1/experiments/metrics")
    async def experiment_metrics(
        context: TenantContext = Depends(resolve_tenant_context),
    ):
        record_usage(
            "experiments.metrics",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return summarize_experiment_metrics()

    return router
