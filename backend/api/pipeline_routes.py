"""Data Engineering Medallion Lakehouse and Schema Contracts API routes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from backend.data.auth import TenantContext
from backend.router_deps import RouterDeps
from scripts.inspect_lakehouse import inspect_lakehouse

logger = logging.getLogger(__name__)

# Locate contracts directory in the project root
CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"


def get_all_contracts() -> dict[str, Any]:
    """Load all defined dataset contracts from the contracts/ directory."""
    contracts = {}
    if CONTRACTS_DIR.exists():
        for path in CONTRACTS_DIR.glob("*.schema.json"):
            try:
                name = path.name.replace(".schema.json", "")
                contracts[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Failed to load contract {path}: {exc}")
    return contracts


def create_pipeline_router(deps: RouterDeps) -> APIRouter:
    """Build the pipeline router with injected dependencies."""
    resolve_tenant_context = deps.resolve_tenant_context
    record_usage = deps.record_usage
    event_storage_status = deps.event_storage_status
    router = APIRouter(tags=["Data Pipelines"])

    @router.get("/v1/platform/pipelines")
    async def get_pipelines_status(
        context: TenantContext = Depends(resolve_tenant_context),
    ):
        """Retrieve the operational status of Medallion Lakehouse tables, data contracts, and event streams."""
        try:
            # 1. Query table snapshots metadata via inspect_lakehouse helper
            lakehouse_report = inspect_lakehouse()
            
            # 2. Get event storage status
            stream_status = event_storage_status()
            
            # 3. Load defined contracts schemas
            contracts_schema = get_all_contracts()
            
            # Record usage
            record_usage(
                "platform.pipelines",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            
            return {
                "status": "ok",
                "lakehouse": lakehouse_report,
                "contracts": contracts_schema,
                "streaming": stream_status,
            }
        except Exception as exc:
            logger.exception("Failed to retrieve pipeline diagnostics")
            return {
                "status": "error",
                "message": str(exc)
            }

    return router
