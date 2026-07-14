"""Tests for the Data Engineering platform pipelines routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_pipeline_diagnostics_endpoint_returns_success(monkeypatch):
    """Assert that /v1/platform/pipelines returns a success status with lakehouse, contracts, and streaming logs."""
    monkeypatch.setenv("NOVA_EVENT_STORE", "jsonl")

    client = TestClient(app)
    response = client.get("/v1/platform/pipelines")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "lakehouse" in data
    assert "contracts" in data
    assert "streaming" in data

    # Verify contracts dictionary is loaded
    contracts = data["contracts"]
    assert isinstance(contracts, dict)

    # Verify streaming configurations are exposed
    streaming = data["streaming"]
    assert "event_store" in streaming
    assert "event_log_path" in streaming
