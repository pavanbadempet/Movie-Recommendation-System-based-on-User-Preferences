"""Unit & Integration Tests for Unity Catalog & OpenLineage Governance Engine."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from etl.data_lineage import get_lineage_tracker
from etl.unity_catalog import get_unity_catalog


def test_unity_catalog_3_level_namespaces_and_rbac():
    uc = get_unity_catalog()
    table = uc.get_table("main", "recommendations", "movies_curated")

    assert table is not None
    assert table.full_name == "main.recommendations.movies_curated"
    assert table.check_privilege("account_admin", "ALL_PRIVILEGES") is True
    assert table.check_privilege("data_scientists", "SELECT") is True
    assert table.check_privilege("unauthorized_user", "SELECT") is False


def test_unity_catalog_pii_masking():
    uc = get_unity_catalog()
    masked = uc.apply_pii_masking("user_email", "user@example.com")
    unmasked = uc.apply_pii_masking("movie_title", "Inception")

    assert masked != "user@example.com"
    assert len(masked) == 16
    assert unmasked == "Inception"


def test_openlineage_tracker_dag_graph():
    tracker = get_lineage_tracker()
    graph = tracker.to_graph_dict()

    assert graph["node_count"] >= 3
    assert graph["edge_count"] >= 2

    openlineage_spec = tracker.get_openlineage_event("test_job")
    assert openlineage_spec["eventType"] == "COMPLETE"
    assert openlineage_spec["job"]["name"] == "test_job"


def test_fastapi_governance_api_endpoints():
    client = TestClient(app)

    res_cat = client.get("/v1/governance/catalogs")
    assert res_cat.status_code == 200
    assert res_cat.json()["status"] == "online"

    res_tables = client.get("/v1/governance/tables?catalog=main&schema=recommendations")
    assert res_tables.status_code == 200
    assert len(res_tables.json()) >= 3

    res_lineage = client.get("/v1/governance/lineage")
    assert res_lineage.status_code == 200
    assert "nodes" in res_lineage.json()
