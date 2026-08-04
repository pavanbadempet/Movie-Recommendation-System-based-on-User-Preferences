"""FastAPI Governance Router for Unity Catalog & Data Lineage."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from etl.data_lineage import get_lineage_tracker
from etl.unity_catalog import get_unity_catalog

router = APIRouter(prefix="/v1/governance", tags=["Data Governance & Unity Catalog"])


@router.get("/catalogs", response_model=dict[str, Any])
def get_catalogs():
    """Retrieve full Unity Catalog metastore hierarchy (3-level namespaces)."""
    uc = get_unity_catalog()
    return {
        "status": "online",
        "default_catalog": uc.default_catalog,
        "catalogs": uc.to_dict(),
    }


@router.get("/tables", response_model=list[dict[str, Any]])
def list_catalog_tables(
    catalog: str = Query("main", description="Catalog name"),
    schema: str = Query("recommendations", description="Schema name"),
):
    """List Unity Catalog tables for a catalog.schema namespace."""
    uc = get_unity_catalog()
    tables = uc.list_tables(catalog, schema)
    return [t.to_dict() for t in tables]


@router.get("/lineage", response_model=dict[str, Any])
def get_data_lineage():
    """Retrieve OpenLineage interactive DAG graph for Medallion pipeline."""
    tracker = get_lineage_tracker()
    return tracker.to_graph_dict()


@router.get("/lineage/openlineage-spec", response_model=dict[str, Any])
def get_openlineage_spec(job_name: str = Query("pyspark_medallion_daily_run")):
    """Get OpenLineage 1.0 specification event payload."""
    tracker = get_lineage_tracker()
    return tracker.get_openlineage_event(job_name)
