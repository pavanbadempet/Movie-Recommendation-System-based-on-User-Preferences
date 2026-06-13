"""Catalog browse and semantic twin API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.data.auth import TenantContext
from backend.router_deps import RouterDeps


def create_browse_router(deps: RouterDeps) -> APIRouter:
    """Build browse routes with injected runtime dependencies."""
    resolve_tenant_context = deps.resolve_tenant_context
    remote_payload_or_raise = deps.remote_payload_or_raise
    get_rec = deps.get_rec
    record_usage = deps.record_usage
    router = APIRouter(tags=["Browse"])

    @router.get("/movies")
    async def list_movies(
        limit: int = Query(default=100, le=1000, description="Maximum movies to return"),
        offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    ):
        remote_payload = await remote_payload_or_raise(
            "/movies",
            params={"limit": limit, "offset": offset},
        )
        if remote_payload is not None:
            return remote_payload

        rec = get_rec()
        movies = rec.movies.iloc[offset : offset + limit]
        return movies.to_dict(orient="records")

    @router.get("/movies/titles")
    async def get_all_titles(
        limit: int = Query(default=100000, ge=1, le=100000, description="Maximum number of titles to return"),
    ):
        remote_payload = await remote_payload_or_raise("/movies/titles", params={"limit": limit})
        if remote_payload is not None:
            return remote_payload

        rec = get_rec()
        return rec.get_all_titles(limit=limit)

    @router.get("/v1/semantic-twins/id/{movie_id}")
    async def semantic_twin_by_id(
        movie_id: int,
        context: TenantContext = Depends(resolve_tenant_context),
    ):
        remote_payload = await remote_payload_or_raise(
            f"/v1/semantic-twins/id/{movie_id}",
            context=context,
        )
        if remote_payload is not None:
            record_usage(
                "semantic_twins.id.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload

        rec = get_rec()
        twin = rec.get_semantic_twin_by_id(movie_id)
        if twin is None:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        record_usage(
            "semantic_twins.id",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return twin

    return router
