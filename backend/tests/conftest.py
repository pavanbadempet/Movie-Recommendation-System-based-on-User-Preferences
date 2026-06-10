"""Shared pytest configuration for backend/tests/."""

import pytest


@pytest.fixture(autouse=True)
def _backend_test_env(monkeypatch):
    """Ensure backend tests have consistent environment variables."""
    monkeypatch.setenv("NOVA_DISABLE_MODEL_DOWNLOADS", "1")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-key-for-ci-only")
