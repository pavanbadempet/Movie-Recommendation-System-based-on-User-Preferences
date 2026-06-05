"""
Admin API routes — extracted from backend/main.py.

Provides admin-only endpoints protected by resolve_admin_token.
Use create_admin_router() to build the router with injected dependencies.

Requirements: 9.1, 9.2, 9.4
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Depends


def create_admin_router(
    *,
    resolve_admin_token: Callable,
    get_apex_engine: Callable,
) -> APIRouter:
    """Build the admin router with injected runtime dependencies.

    Parameters
    ----------
    resolve_admin_token:
        FastAPI dependency that validates the admin token and returns it.
    get_apex_engine:
        Callable that returns the singleton ApexEnsembleEngine instance.
    """
    router = APIRouter(tags=["Admin"])

    @router.post("/v1/admin/reload-ensemble-weights")
    async def reload_ensemble_weights(
        admin_token: str = Depends(resolve_admin_token),
    ):
        """Reload ensemble blend weights from models/ensemble_weights.json without restarting."""
        engine = get_apex_engine()
        new_weights = engine.reload_weights()
        weights_file = Path("models/ensemble_weights.json")
        source = "file" if weights_file.exists() else "defaults"
        return {
            "status": "ok",
            "weights": new_weights,
            "source": source,
        }

    return router
