"""
Export the APEX OpenAPI specification to static JSON and YAML files.

Generates:
  docs/openapi.json  — machine-readable OpenAPI 3.1 spec
  docs/openapi.yaml  — human-readable YAML version

These files are committed to the repository so the full API spec is
available without running the server. They are also used by:
  - GitHub Pages / Swagger UI for hosted interactive docs
  - API client code generation (openapi-generator, etc.)
  - CI contract testing

Usage:
    python scripts/export_openapi_spec.py

The script imports the FastAPI app in a lightweight mode (no model loading,
no database connections) by setting NOVA_DISABLE_MODEL_DOWNLOADS=1 and
NOVA_HEALTH_LOAD_RECOMMENDER=false before importing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Set env vars BEFORE importing the app to prevent model loading
# ---------------------------------------------------------------------------
os.environ.setdefault("NOVA_DISABLE_MODEL_DOWNLOADS", "1")
os.environ.setdefault("NOVA_HEALTH_LOAD_RECOMMENDER", "false")
os.environ.setdefault("NOVA_BACKGROUND_RECOMMENDER_WARMUP", "false")
os.environ.setdefault("JWT_SECRET_KEY", "openapi-export-placeholder-not-for-auth")

logging.basicConfig(level=logging.WARNING)  # suppress startup noise during export

OUTPUT_DIR = REPO_ROOT / "docs"
JSON_PATH = OUTPUT_DIR / "openapi.json"
YAML_PATH = OUTPUT_DIR / "openapi.yaml"


def _rebuild_forward_refs(app) -> None:
    """
    Force Pydantic to resolve all ForwardRef annotations on models registered
    with the app. This is needed because the router factory pattern passes
    Pydantic models as local variables, which Pydantic sees as ForwardRefs
    when generating the OpenAPI schema.
    """
    from backend.main import (
        EnrichedMovie,
        EnrichedRecommendationResponse,
        EventRequest,
        EventResponse,
        HealthResponse,
        Movie,
        PlatformContextResponse,
        RecommendationResponse,
        UsageResponse,
    )

    models = [
        Movie,
        EnrichedMovie,
        HealthResponse,
        RecommendationResponse,
        EnrichedRecommendationResponse,
        EventRequest,
        EventResponse,
        PlatformContextResponse,
        UsageResponse,
    ]

    # Build a namespace mapping all model names to their classes
    namespace = {m.__name__: m for m in models}

    for model in models:
        try:
            model.model_rebuild(_types_namespace=namespace)
        except Exception as exc:
            logging.debug("model_rebuild skipped for %s: %s", model.__name__, exc)


def export_json(spec: dict) -> None:
    """Write the OpenAPI spec as formatted JSON."""
    JSON_PATH.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON -> {JSON_PATH.relative_to(REPO_ROOT)}")


def export_yaml(spec: dict) -> None:
    """Write the OpenAPI spec as YAML (requires PyYAML)."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        print("  YAML skipped -- install PyYAML: pip install pyyaml")
        return

    YAML_PATH.write_text(
        yaml.dump(spec, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"  YAML -> {YAML_PATH.relative_to(REPO_ROOT)}")


def main() -> None:
    print("Exporting APEX OpenAPI specification...")

    # Import the app — triggers FastAPI route registration but NOT model loading
    from backend.main import app

    # Resolve all Pydantic ForwardRefs before schema generation
    _rebuild_forward_refs(app)

    # Clear any cached schema so it's regenerated with resolved refs
    app.openapi_schema = None

    spec = app.openapi()

    # Patch the servers block
    spec["servers"] = [
        {"url": "http://localhost:8000", "description": "Local development"},
        {
            "url": "https://your-api.onrender.com",
            "description": "Production (Render free tier — Tier 3)",
        },
    ]

    endpoint_count = sum(len(methods) for methods in spec.get("paths", {}).values())
    schema_count = len(spec.get("components", {}).get("schemas", {}))
    print(f"  Endpoints: {endpoint_count} | Schemas: {schema_count}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_json(spec)
    export_yaml(spec)

    print("Done.")


if __name__ == "__main__":
    main()
