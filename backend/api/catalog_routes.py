"""Catalog onboarding API routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.data.auth import TenantContext


class CatalogPreviewRequest(BaseModel):
    """CSV catalog preview request from the APEX Console."""

    filename: str = "catalog.csv"
    csv_text: str
    column_mapping: dict[str, str] = Field(default_factory=dict)
    sample_size: int = 20


class CatalogUploadRequest(CatalogPreviewRequest):
    """Persisted catalog upload request."""


def create_catalog_router(
    *,
    resolve_tenant_context: Callable[..., TenantContext],
    profile_catalog_csv: Callable[..., dict[str, Any]],
    persist_catalog_upload: Callable[..., dict[str, Any]],
    record_usage: Callable[..., Any],
) -> APIRouter:
    """Build catalog onboarding routes with injected runtime dependencies."""
    router = APIRouter(tags=["Catalog"])

    @router.post("/v1/catalog/preview")
    async def preview_catalog(
        payload: CatalogPreviewRequest,
        context: TenantContext = Depends(resolve_tenant_context),
    ):
        try:
            profile = profile_catalog_csv(
                payload.csv_text,
                tenant_id=context.tenant_id,
                catalog_id=context.catalog_id,
                column_mapping=payload.column_mapping,
                sample_size=payload.sample_size,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        record_usage(
            "catalog.preview",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return profile

    @router.post("/v1/catalog/upload")
    async def upload_catalog(
        payload: CatalogUploadRequest,
        context: TenantContext = Depends(resolve_tenant_context),
    ):
        try:
            manifest = persist_catalog_upload(
                payload.csv_text,
                tenant_id=context.tenant_id,
                catalog_id=context.catalog_id,
                filename=payload.filename,
                column_mapping=payload.column_mapping,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        record_usage(
            "catalog.upload",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return {
            "status": "stored",
            "upload_id": manifest["upload_id"],
            "raw_path": manifest["raw_path"],
            "manifest_path": manifest["manifest_path"],
            "profile": manifest["profile"],
        }

    return router
