"""
App metadata helpers for the Movie Recommendation System.

Provides :func:`app_metadata` and :func:`public_base_url` — lightweight
utilities that return deploy-lineage information and the externally visible
API base URL without requiring the recommender model to be loaded.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Request

APP_VERSION = "2.0.0"
REVISION_FILE = Path(__file__).resolve().parent.parent.parent / "REVISION"


def app_metadata() -> dict[str, str | None]:
    """Return deploy lineage without loading the recommender."""
    import sys

    revision_file = REVISION_FILE
    if "backend.main" in sys.modules:
        main_mod = sys.modules["backend.main"]
        if hasattr(main_mod, "REVISION_FILE"):
            revision_file = main_mod.REVISION_FILE

    commit = None
    source = None
    for env_name in (
        "NOVA_APP_COMMIT",
        "RENDER_GIT_COMMIT",
        "GITHUB_SHA",
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            commit = value
            source = env_name
            break
    if not commit and revision_file.exists():
        try:
            value = revision_file.read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
        else:
            if value:
                commit = value
                source = "REVISION"
    if not commit:
        for env_name in ("SOURCE_VERSION", "COMMIT_SHA"):
            value = os.getenv(env_name, "").strip()
            if value:
                commit = value
                source = env_name
                break
    return {
        "version": APP_VERSION,
        "commit": commit[:12] if commit else None,
        "commit_full": commit if commit else None,
        "source": source,
    }


def public_base_url(request: Request) -> str:
    """Return the configured public API base URL for absolute local links.

    Host and forwarded-host headers are intentionally ignored here. When no
    canonical public base URL is configured, same-origin frontend redirects
    remain relative (for example `/ui/`) instead of being resolved against an
    attacker-controlled Host value.
    """
    configured = os.getenv("NOVA_PUBLIC_BASE_URL", "").strip()
    if not configured:
        return ""
    parsed = urlparse(configured)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return configured.rstrip("/") + "/"
