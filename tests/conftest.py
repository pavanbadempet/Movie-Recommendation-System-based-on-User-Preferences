"""Shared pytest configuration."""

import pytest


@pytest.fixture(autouse=True)
def disable_external_model_downloads(monkeypatch):
    """Unit tests must use their fixtures, not live Hugging Face artifacts."""
    monkeypatch.setenv("NOVA_DISABLE_MODEL_DOWNLOADS", "1")

