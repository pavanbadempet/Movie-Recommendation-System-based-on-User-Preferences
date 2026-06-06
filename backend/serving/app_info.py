"""
App metadata helpers for the Movie Recommendation System.

Provides :func:`app_metadata` and :func:`public_base_url` — lightweight
utilities that return deploy-lineage information and the externally visible
API base URL without requiring the recommender model to be loaded.
"""

import os
from pathlib import Path

from fastapi import Request

APP_VERSION = "2.0.0"
REVISION_FILE = Path(__file__).resolve().parent.parent / "REVISION"


def app_metadata() -> dict[str, str | None]:
    """Return deploy lineage without loading the recommender."""
    import sys
    revision_file = REVISION_FILE
    if "backend.main" in sys.modules:
        main_mod = sys.modules["backend.main"]
        if hasattr(main_mod, "REVISION_FILE"):
            revision_file = getattr(main_mod, "REVISION_FILE")

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
    """Return the externally visible API base URL behind hosted proxies."""
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    forwarded_host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
    proto = forwarded_proto or request.url.scheme
    host = forwarded_host or request.headers.get("host") or request.url.netloc
    if proto == "http" and host.endswith((".hf.space", ".onrender.com", ".streamlit.app")):
        proto = "https"
    return f"{proto}://{host.strip('/')}/"
