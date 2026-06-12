"""
Admin API routes — extracted from backend/main.py.

Provides admin-only endpoints protected by resolve_admin_token.
Use create_admin_router() to build the router with injected dependencies.

Requirements: 9.1, 9.2, 9.4
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from backend.router_deps import RouterDeps


def create_admin_router(deps: RouterDeps) -> APIRouter:
    """Build the admin router with injected runtime dependencies."""
    resolve_admin_token = deps.resolve_admin_token
    get_apex_engine = deps.get_apex_engine
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

    @router.post("/v1/demo/reset")
    async def reset_demo(
        admin_token: str = Depends(resolve_admin_token),
    ):
        """Reload baseline demo recommendation artifacts from disk."""
        rec = deps.get_rec()
        rec.load()
        return {
            "status": "ok",
            "message": "Demo recommendation artifacts successfully reloaded from disk.",
            "movie_count": len(rec._movies) if rec._movies is not None else 0,
            "vector_count": len(rec._vectors) if rec._vectors is not None else 0,
        }

    return router
