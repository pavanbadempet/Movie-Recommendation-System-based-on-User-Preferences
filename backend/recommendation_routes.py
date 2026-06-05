"""
Recommendation, search, movie, events, and chat API routes.

Extracted from backend/main.py to keep main.py under 1500 lines.
Requirements: 11.3
"""

from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import logging
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import RedirectResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level Pydantic models for chat endpoint
# These must be at module level (not inside factory functions) so that
# FastAPI's TypeAdapter can resolve them for OpenAPI schema generation.
# ---------------------------------------------------------------------------
from pydantic import BaseModel as _BaseModel  # noqa: E402


class ChatMessage(_BaseModel):
    role: str
    content: str


class ChatRequest(_BaseModel):
    messages: list[ChatMessage]


class ChatResponse(_BaseModel):
    role: str
    content: str


# ---------------------------------------------------------------------------
# Module-level config (resolved at import time from environment / main.py)
# ---------------------------------------------------------------------------
_TMDB_KEY: str | None = None
_TMDB_BASE: str = "https://api.themoviedb.org/3"
_FRONTEND_DIST_DIR = None
_http_client_getter = None  # callable → httpx.AsyncClient | None
_online_learner_getter = None  # callable → OnlineLearner | None
_recommender_getter = None  # callable → Recommender | None
_slo_tracker_getter = None  # callable → RequestSloTracker
_tier_detector_getter = None  # callable → TierDetector | None
_limiter = None
_app_metadata_fn = None
_public_base_url_fn = None
_platform_readiness_report_fn = None
_recommendation_diagnostic_report_fn = None
_trigger_active_inference_fn = None
_serving_lineage_fn = None
_candidate_event_summary_fn = None
_event_logging_enabled_fn = None
_safe_float_fn = None


def configure(
    *,
    tmdb_key,
    tmdb_base,
    frontend_dist_dir,
    http_client_getter,
    online_learner_getter,
    recommender_getter,
    slo_tracker_getter,
    tier_detector_getter,
    limiter,
    app_metadata_fn,
    public_base_url_fn,
    platform_readiness_report_fn,
    recommendation_diagnostic_report_fn,
    trigger_active_inference_fn,
    serving_lineage_fn,
    candidate_event_summary_fn,
    event_logging_enabled_fn,
    safe_float_fn,
):
    """Wire module-level singletons from main.py at startup."""
    global _TMDB_KEY, _TMDB_BASE, _FRONTEND_DIST_DIR
    global _http_client_getter, _online_learner_getter, _recommender_getter
    global _slo_tracker_getter, _tier_detector_getter, _limiter
    global _app_metadata_fn, _public_base_url_fn
    global _platform_readiness_report_fn, _recommendation_diagnostic_report_fn
    global _trigger_active_inference_fn, _serving_lineage_fn
    global _candidate_event_summary_fn, _event_logging_enabled_fn, _safe_float_fn
    _TMDB_KEY = tmdb_key
    _TMDB_BASE = tmdb_base
    _FRONTEND_DIST_DIR = frontend_dist_dir
    _http_client_getter = http_client_getter
    _online_learner_getter = online_learner_getter
    _recommender_getter = recommender_getter
    _slo_tracker_getter = slo_tracker_getter
    _tier_detector_getter = tier_detector_getter
    _limiter = limiter
    _app_metadata_fn = app_metadata_fn
    _public_base_url_fn = public_base_url_fn
    _platform_readiness_report_fn = platform_readiness_report_fn
    _recommendation_diagnostic_report_fn = recommendation_diagnostic_report_fn
    _trigger_active_inference_fn = trigger_active_inference_fn
    _serving_lineage_fn = serving_lineage_fn
    _candidate_event_summary_fn = candidate_event_summary_fn
    _event_logging_enabled_fn = event_logging_enabled_fn
    _safe_float_fn = safe_float_fn


# ---------------------------------------------------------------------------
# Async LRU cache for TMDB calls
# ---------------------------------------------------------------------------


class _AsyncLRUCache:
    def __init__(self, maxsize=1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            result = await func(*args, **kwargs)
            if result is not None:
                self.cache[key] = result
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)
            return result

        return wrapper


@_AsyncLRUCache(maxsize=1000)
async def fetch_trailer(movie_id: int):
    http_client = _http_client_getter() if _http_client_getter else None
    if not _TMDB_KEY or not http_client:
        return None
    try:
        r = await http_client.get(
            f"{_TMDB_BASE}/movie/{movie_id}/videos",
            params={"api_key": _TMDB_KEY, "language": "en-US"},
        )
        data = r.json()
        for v in data.get("results", []):
            if v.get("type") == "Trailer":
                return v.get("key")
        if data.get("results"):
            return data["results"][0].get("key")
    except Exception as e:
        logger.warning("Trailer fetch failed for %s: %s", movie_id, e)
    return None


@_AsyncLRUCache(maxsize=1000)
async def fetch_details(movie_id: int):
    http_client = _http_client_getter() if _http_client_getter else None
    if not _TMDB_KEY or not http_client:
        return {}
    try:
        r = await http_client.get(
            f"{_TMDB_BASE}/movie/{movie_id}",
            params={"api_key": _TMDB_KEY},
        )
        return r.json()
    except Exception as e:
        logger.warning("Details fetch failed for %s: %s", movie_id, e)
    return {}


@_AsyncLRUCache(maxsize=1000)
async def fetch_credits(movie_id: int):
    http_client = _http_client_getter() if _http_client_getter else None
    if not _TMDB_KEY or not http_client:
        return {"cast": "N/A", "director": "N/A"}
    try:
        r = await http_client.get(
            f"{_TMDB_BASE}/movie/{movie_id}/credits",
            params={"api_key": _TMDB_KEY},
        )
        data = r.json()
        cast = [c["name"] for c in data.get("cast", [])[:3]]
        director = next(
            (c["name"] for c in data.get("crew", []) if c.get("job") == "Director"),
            "Unknown",
        )
        return {"cast": ", ".join(cast), "director": director}
    except Exception as e:
        logger.warning("Credits fetch failed for %s: %s", movie_id, e)
    return {"cast": "N/A", "director": "N/A"}


async def enrich_movie(movie: dict) -> dict:
    movie_id = movie["id"]
    trailer, details, credits = await asyncio.gather(
        fetch_trailer(movie_id),
        fetch_details(movie_id),
        fetch_credits(movie_id),
    )
    return {
        **movie,
        "trailer_key": trailer,
        "runtime": details.get("runtime"),
        "director": credits.get("director"),
        "cast": credits.get("cast"),
    }


async def _apply_llm_explanations(recommendations, user_id, user_context=None):
    from backend.llm_explanations import generate_explanation

    def process_movie(m):
        m["explanation_text"] = generate_explanation(user_id, m, user_context)
        return m

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=10) as pool:
        tasks = [loop.run_in_executor(pool, process_movie, m) for m in recommendations]
        await asyncio.gather(*tasks)
    return recommendations


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------
# Pydantic models used as route parameter type annotations must be present in
# this module's global namespace. Because `from __future__ import annotations`
# makes all annotations lazy strings (PEP 563), FastAPI's TypeAdapter resolves
# them against __globals__ of the defining module. The models are defined in
# backend/main.py and injected here at import time to avoid circular imports.
# This is a one-time deferred import that runs when the first factory is called.
def _inject_pydantic_models_into_globals() -> None:
    """Inject Pydantic response models into this module's global namespace."""
    import sys

    _mod = sys.modules[__name__]
    if getattr(_mod, "_PYDANTIC_MODELS_INJECTED", False):
        return
    try:
        # Import from backend.main — safe because main.py is already loaded
        # by the time any factory function is called.
        import backend.main as _main  # noqa: PLC0415

        for _name in [
            "Movie",
            "EnrichedMovie",
            "HealthResponse",
            "RecommendationResponse",
            "EnrichedRecommendationResponse",
            "EventRequest",
            "EventResponse",
            "PlatformContextResponse",
            "UsageResponse",
        ]:
            if hasattr(_main, _name):
                setattr(_mod, _name, getattr(_main, _name))
        _mod._PYDANTIC_MODELS_INJECTED = True  # type: ignore[attr-defined]
    except Exception as exc:
        logger.debug("Could not inject Pydantic models into recommendation_routes globals: %s", exc)


def create_recommendation_router(
    *,
    get_rec,
    record_usage,
    remote_payload_or_raise,
    record_recommendation_events,
    resolve_tenant_context,
    build_user_behavior_profile,
    assign_experiment,
    attach_experiment,
    aggregate_behavior_features,
    append_event,
    summarize_recommendation_events,
    evaluate_artifact_health,
    build_slo_report,
    frontend_status_report,
    configured_frontends,
    remote_recommender_status,
    load_ranker,
    enforce_payload_context,
    get_db,
    generate_chat_response,
    summarize_usage,
    event_storage_status,
    get_events_path,
    limiter,
    Movie,
    EnrichedMovie,
    HealthResponse,
    RecommendationResponse,
    EnrichedRecommendationResponse,
    EventRequest,
    EventResponse,
    PlatformContextResponse,
    UsageResponse,
):
    _inject_pydantic_models_into_globals()
    router = APIRouter()

    # ── /movies/latest ──────────────────────────────────────────────────────
    @router.get("/movies/latest")
    async def get_latest_movies(limit: int = Query(default=8, le=20)):
        rec = get_rec()
        import math

        def _sanitize_float(v):
            if isinstance(v, float) and math.isnan(v):
                return None
            return v

        if not _TMDB_KEY or not (_http_client_getter and _http_client_getter()):
            all_movies = rec.get_all_movies() if hasattr(rec, "get_all_movies") else []
            sorted_movies = sorted(
                [m for m in all_movies if m.get("poster_path") and m.get("release_date")],
                key=lambda m: m.get("release_date", ""),
                reverse=True,
            )
            return [{k: _sanitize_float(v) for k, v in m.items()} for m in sorted_movies[:limit]]

        seen_ids: set[int] = set()
        catalog_matches: list[dict] = []
        endpoints = [
            f"{_TMDB_BASE}/trending/movie/week",
            f"{_TMDB_BASE}/movie/now_playing",
            f"{_TMDB_BASE}/movie/popular",
        ]
        http_client = _http_client_getter()

        tasks = []
        for url in endpoints:
            for page in range(1, 4):
                tasks.append(http_client.get(url, params={"api_key": _TMDB_KEY, "language": "en-US", "page": page}))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.warning("TMDB latest fetch failed: %s", r)
                continue

            try:
                data = r.json()
                for movie in data.get("results", []):
                    mid = movie.get("id")
                    if not mid or mid in seen_ids or not movie.get("poster_path"):
                        continue
                    seen_ids.add(mid)
                    if rec.get_movie_by_id(mid) is not None:
                        catalog_matches.append(movie)
            except Exception as e:
                logger.warning("Failed to process TMDB latest fetch result: %s", e)

            if len(catalog_matches) >= limit:
                break

        genre_map = {
            28: "Action",
            12: "Adventure",
            16: "Animation",
            35: "Comedy",
            80: "Crime",
            99: "Documentary",
            18: "Drama",
            10751: "Family",
            14: "Fantasy",
            36: "History",
            27: "Horror",
            10402: "Music",
            9648: "Mystery",
            10749: "Romance",
            878: "Science Fiction",
            10770: "TV Movie",
            53: "Thriller",
            10752: "War",
            37: "Western",
        }

        async def enrich_tmdb(m: dict):
            try:
                trailer, credits_data = await asyncio.gather(fetch_trailer(m["id"]), fetch_credits(m["id"]))
                gids = m.get("genre_ids", [])
                genres = ", ".join(genre_map.get(g, "") for g in gids if g in genre_map)
                return {
                    "id": m["id"],
                    "title": m.get("title", ""),
                    "overview": m.get("overview"),
                    "genres": genres or None,
                    "vote_average": m.get("vote_average"),
                    "vote_count": m.get("vote_count"),
                    "popularity": m.get("popularity"),
                    "release_date": m.get("release_date"),
                    "poster_path": m.get("poster_path"),
                    "trailer_key": trailer,
                    "runtime": None,
                    "director": credits_data.get("director"),
                    "cast": credits_data.get("cast"),
                }
            except Exception:
                return None

        enriched = await asyncio.gather(*(enrich_tmdb(m) for m in catalog_matches[:limit]))
        return [e for e in enriched if e]

    # ── /v1/frontends/status ─────────────────────────────────────────────────
    @router.get("/v1/frontends/status")
    async def frontends_status(
        request: Request,
        include_remote: bool = Query(default=True),
        preferred: str | None = Query(default=None),
    ):
        return await frontend_status_report(
            frontend_dist_dir=_FRONTEND_DIST_DIR,
            base_url=_public_base_url_fn(request),
            include_remote=include_remote,
            preferred=preferred,
            app=_app_metadata_fn(),
        )

    # ── /v1/platform/slo ─────────────────────────────────────────────────────
    @router.get("/v1/platform/slo")
    async def platform_slo(
        request: Request,
        include_frontends: bool = Query(default=False),
        include_remote_frontends: bool = Query(default=False),
        preferred_frontend: str | None = Query(default=None),
    ):
        from backend import recommender as recommender_module

        artifact_report = await run_in_threadpool(
            lambda: evaluate_artifact_health(
                models_dir=recommender_module.MODELS_DIR,
                data_dir=recommender_module.DATA_DIR,
            )
        )
        dependencies = {
            "artifacts": {
                "status": artifact_report.get("status"),
                "row_counts": artifact_report.get("row_counts"),
                "alignment": artifact_report.get("alignment"),
            },
            "remote_recommender": remote_recommender_status(),
        }
        if include_frontends:
            dependencies["frontends"] = await frontend_status_report(
                frontend_dist_dir=_FRONTEND_DIST_DIR,
                base_url=_public_base_url_fn(request),
                include_remote=include_remote_frontends,
                preferred=preferred_frontend,
                app=_app_metadata_fn(),
            )
        else:
            dependencies["frontends"] = {
                "status": "skipped",
                "reason": "Set include_frontends=true to attach frontend failover health.",
            }
        return build_slo_report(tracker=_slo_tracker_getter(), app=_app_metadata_fn(), dependencies=dependencies)

    # ── /go and /v1/frontends/launch ─────────────────────────────────────────
    @router.get("/go")
    @router.get("/v1/frontends/launch")
    async def launch_frontend(
        request: Request,
        include_remote: bool = Query(default=True),
        preferred: str | None = Query(default=None),
    ):
        report = await frontend_status_report(
            frontend_dist_dir=_FRONTEND_DIST_DIR,
            base_url=_public_base_url_fn(request),
            include_remote=include_remote,
            preferred=preferred,
            app=_app_metadata_fn(),
        )
        selected = report.get("selected") or {}
        launch_url = selected.get("absolute_url")
        if not launch_url or selected.get("status") == "unavailable":
            raise HTTPException(
                status_code=503, detail={"message": "No frontend is currently available", "report": report}
            )
        return RedirectResponse(str(launch_url), status_code=302)

    return router


def create_core_router(
    *,
    get_rec,
    record_usage,
    remote_payload_or_raise,
    record_recommendation_events,
    resolve_tenant_context,
    build_user_behavior_profile,
    assign_experiment,
    attach_experiment,
    aggregate_behavior_features,
    append_event,
    summarize_recommendation_events,
    evaluate_artifact_health,
    load_ranker,
    enforce_payload_context,
    get_db,
    generate_chat_response,
    summarize_usage,
    event_storage_status,
    get_events_path,
    limiter,
    Movie,
    EnrichedMovie,
    HealthResponse,
    RecommendationResponse,
    EnrichedRecommendationResponse,
    EventRequest,
    EventResponse,
    PlatformContextResponse,
    UsageResponse,
):
    """Core movie/search/recommendation/events/chat router."""
    _inject_pydantic_models_into_globals()
    router = APIRouter()

    # ── /health ───────────────────────────────────────────────────────────────
    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        metadata = _app_metadata_fn()
        td = _tier_detector_getter() if _tier_detector_getter else None
        if td is not None and td._detected:
            p = td._profile
            serving_tier = td._tier
            hardware_profile = {
                "gpu_available": p.gpu_available,
                "ram_gb": round(p.ram_gb, 2),
                "cpu_cores": p.cpu_cores,
            }
            tier_selection_reason = td._reason
        else:
            serving_tier = None
            hardware_profile = None
            tier_selection_reason = "detection_pending"

        load_rec_env = os.getenv("NOVA_HEALTH_LOAD_RECOMMENDER", "true").strip().lower()
        if load_rec_env in {"0", "false", "no", "off"}:
            from backend import recommender as recommender_module

            report = evaluate_artifact_health(
                models_dir=recommender_module.MODELS_DIR, data_dir=recommender_module.DATA_DIR
            )
            return HealthResponse(
                status="healthy" if report.get("files", {}).get("movies", {}).get("exists") else "degraded",
                movie_count=int((report.get("row_counts") or {}).get("movies") or 0),
                app_version=metadata["version"],
                app_commit=metadata["commit"],
                serving_tier=serving_tier,
                hardware_profile=hardware_profile,
                tier_selection_reason=tier_selection_reason,
            )
        try:
            rec = get_rec()
            return HealthResponse(
                status="healthy",
                movie_count=len(rec.movies),
                app_version=metadata["version"],
                app_commit=metadata["commit"],
                serving_tier=serving_tier,
                hardware_profile=hardware_profile,
                tier_selection_reason=tier_selection_reason,
            )
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return HealthResponse(
                status="unhealthy",
                movie_count=0,
                app_version=metadata["version"],
                app_commit=metadata["commit"],
                serving_tier=serving_tier,
                hardware_profile=hardware_profile,
                tier_selection_reason=tier_selection_reason,
            )

    # ── /v1/platform/context ─────────────────────────────────────────────────
    @router.get("/v1/platform/context", response_model=PlatformContextResponse)
    async def platform_context(context=Depends(resolve_tenant_context)):
        return PlatformContextResponse(
            tenant_id=context.tenant_id,
            catalog_id=context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
            mode="authenticated" if context.authenticated else "public-demo",
        )

    # ── /v1/platform/status ──────────────────────────────────────────────────
    @router.get("/v1/platform/status")
    async def platform_status(context=Depends(resolve_tenant_context)):
        remote_payload = await remote_payload_or_raise("/v1/platform/status", context=context)
        if remote_payload is not None:
            behavior = await run_in_threadpool(lambda: aggregate_behavior_features(limit=5))
            assignment = assign_experiment(subject_id=f"{context.tenant_id}:{context.catalog_id}:status")
            record_usage(
                "platform.status.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            if isinstance(remote_payload, dict):
                payload = dict(remote_payload)
                payload["gateway"] = {
                    "status": "ready",
                    "app": _app_metadata_fn(),
                    "tenant_id": context.tenant_id,
                    "catalog_id": context.catalog_id,
                    "event_store": {
                        "mode": behavior.get("event_store"),
                        "durable": behavior.get("durable"),
                        "event_table": behavior.get("event_table"),
                        "total_events": behavior.get("total_events"),
                    },
                    "remote_recommender": {},
                    "experimentation": {"enabled": True, "default_assignment": assignment},
                }
                return payload
            return remote_payload
        rec = get_rec()
        ranker = getattr(rec, "_learned_ranker", None)
        behavior = aggregate_behavior_features(limit=5)
        assignment = assign_experiment(subject_id=f"{context.tenant_id}:{context.catalog_id}:status")
        record_usage(
            "platform.status",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return {
            "status": "ready",
            "app": _app_metadata_fn(),
            "tenant_id": context.tenant_id,
            "catalog_id": context.catalog_id,
            "movie_count": len(rec.movies),
            "event_store": {
                "mode": behavior.get("event_store"),
                "durable": behavior.get("durable"),
                "event_table": behavior.get("event_table"),
                "total_events": behavior.get("total_events"),
            },
            "ranker": {
                "available": ranker is not None,
                "training_mode": (ranker.metadata.get("training_mode") if ranker else None),
                "promotion": (ranker.metadata.get("promotion") if ranker else None),
            },
            "experimentation": {"enabled": True, "default_assignment": assignment},
        }

    # ── /v1/platform/readiness ───────────────────────────────────────────────
    @router.get("/v1/platform/readiness")
    async def platform_readiness(
        strict: bool = Query(default=False),
        k: int = Query(default=10, ge=1, le=50),
        context=Depends(resolve_tenant_context),
    ):
        remote_payload = await remote_payload_or_raise(
            "/v1/platform/readiness", params={"strict": strict, "k": k}, context=context
        )
        if remote_payload is not None:
            record_usage(
                "platform.readiness.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload
        from backend import recommender as recommender_module

        rec = await run_in_threadpool(get_rec)
        artifact_report = await run_in_threadpool(
            lambda: evaluate_artifact_health(
                models_dir=recommender_module.MODELS_DIR, data_dir=recommender_module.DATA_DIR
            )
        )
        behavior = await run_in_threadpool(lambda: aggregate_behavior_features(limit=5))
        report = await run_in_threadpool(
            lambda: _platform_readiness_report_fn(
                context=context, rec=rec, artifact_report=artifact_report, behavior=behavior, strict=strict, k=k
            )
        )
        record_usage(
            "platform.readiness",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
            status=report.get("status"),
        )
        return report

    # ── /v1/usage ────────────────────────────────────────────────────────────
    @router.get("/v1/usage", response_model=UsageResponse)
    async def usage_summary(context=Depends(resolve_tenant_context), limit: int = Query(default=20, ge=1, le=100)):
        record_usage(
            "usage.summary",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return summarize_usage(limit=limit)

    # ── /v1/ranker/status ────────────────────────────────────────────────────
    @router.get("/v1/ranker/status")
    async def ranker_status(context=Depends(resolve_tenant_context)):
        remote_payload = await remote_payload_or_raise("/v1/ranker/status", context=context)
        if remote_payload is not None:
            record_usage(
                "ranker.status.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload
        from backend import recommender as recommender_module

        ranker = await run_in_threadpool(lambda: load_ranker(models_dir=recommender_module.MODELS_DIR))
        record_usage(
            "ranker.status",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        if ranker is None:
            return {
                "available": False,
                "message": "No learned ranker artifact loaded. Run scripts/train_ranker.py to create one.",
            }
        return {"available": True, "feature_columns": ranker.feature_columns, "metadata": ranker.metadata}

    return router


def create_search_movie_router(
    *,
    get_rec,
    record_usage,
    remote_payload_or_raise,
    record_recommendation_events,
    resolve_tenant_context,
    build_user_behavior_profile,
    assign_experiment,
    attach_experiment,
    aggregate_behavior_features,
    append_event,
    summarize_recommendation_events,
    enforce_payload_context,
    get_db,
    generate_chat_response,
    summarize_usage,
    event_storage_status,
    get_events_path,
    limiter,
    Movie,
    EnrichedMovie,
    RecommendationResponse,
    EnrichedRecommendationResponse,
    EventRequest,
    EventResponse,
):
    """Search, movie detail, recommendation, events, and chat router."""
    _inject_pydantic_models_into_globals()
    from sqlalchemy.orm import Session

    router = APIRouter()

    # ── /v1/search ───────────────────────────────────────────────────────────
    @router.get("/v1/search", response_model=list[Movie])
    @router.get("/search", response_model=list[Movie])
    @limiter.limit("30/minute")
    async def search_movies(
        request: Request,
        q: str = Query(..., min_length=1),
        limit: int = Query(default=20, le=100),
        context=Depends(resolve_tenant_context),
    ):
        remote_payload = await remote_payload_or_raise("/v1/search", params={"q": q, "limit": limit}, context=context)
        if remote_payload is not None:
            record_usage(
                "search.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload
        rec = get_rec()
        results = rec.search_movies(q, limit=limit)
        record_usage(
            "search", context.tenant_id, context.catalog_id, plan=context.plan, authenticated=context.authenticated
        )
        return results

    # ── /v1/search/ai ────────────────────────────────────────────────────────
    @router.get("/v1/search/ai", response_model=list[Movie])
    async def ai_search_movies(
        q: str = Query(..., min_length=1),
        limit: int = Query(default=20, le=100),
        top_k: int | None = Query(default=None, ge=1, le=100),
        context=Depends(resolve_tenant_context),
    ):
        result_limit = top_k or limit
        remote_payload = await remote_payload_or_raise(
            "/v1/search/ai", params={"q": q, "limit": limit, "top_k": result_limit}, context=context
        )
        if remote_payload is not None:
            record_usage(
                "search.ai.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload
        rec = get_rec()
        results = rec.ai_search(q, n=result_limit)
        record_usage(
            "search.ai", context.tenant_id, context.catalog_id, plan=context.plan, authenticated=context.authenticated
        )
        return results

    # ── /movie/{movie_id} ────────────────────────────────────────────────────
    @router.get("/movie/{movie_id}", response_model=Movie)
    async def get_movie(movie_id: int):
        remote_payload = await remote_payload_or_raise(f"/movie/{movie_id}")
        if remote_payload is not None:
            return remote_payload
        rec = get_rec()
        movie = rec.get_movie_by_id(movie_id)
        if movie is None:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        return movie

    @router.get("/movie/{movie_id}/enriched", response_model=EnrichedMovie)
    async def get_movie_enriched(movie_id: int):
        rec = get_rec()
        movie = rec.get_movie_by_id(movie_id)
        if movie is None:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        if not _TMDB_KEY:
            return {**movie, "trailer_key": None, "runtime": None, "director": None, "cast": None}
        return await enrich_movie(movie)

    @router.get("/movie/{movie_id}/trailer")
    async def get_movie_trailer(movie_id: int):
        if not _TMDB_KEY:
            return {"trailer_key": None}
        return {"trailer_key": await fetch_trailer(movie_id)}

    # ── /v1/events ───────────────────────────────────────────────────────────
    @router.post("/v1/events", response_model=EventResponse)
    @router.post("/events", response_model=EventResponse)
    async def record_event(
        payload: EventRequest,
        background_tasks: BackgroundTasks,
        context=Depends(resolve_tenant_context),
        db: Session = Depends(get_db),
    ):
        enforce_payload_context(payload, context)
        if payload.event_type == "search":
            if not payload.query_text or not payload.query_text.strip():
                raise HTTPException(status_code=400, detail="query_text is required for search events")
        elif payload.movie_id is None and payload.content_id is None and payload.source_content_id is None:
            raise HTTPException(status_code=400, detail="movie_id or content_id is required for content events")
        if payload.event_type == "rating":
            if payload.rating is None:
                raise HTTPException(status_code=400, detail="rating is required for rating events")
            if not 1 <= payload.rating <= 5:
                raise HTTPException(status_code=400, detail="rating must be between 1 and 5")

        event_payload = payload.model_dump(exclude_none=True)
        event_payload["tenant_id"] = event_payload.get("tenant_id") or context.tenant_id
        event_payload["catalog_id"] = event_payload.get("catalog_id") or context.catalog_id
        event_payload["source"] = "api"

        try:
            event = append_event(event_payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        from backend.database import UserEvent

        try:
            pg_event = UserEvent(
                tenant_id=context.tenant_id,
                event_type=payload.event_type,
                event_value=payload.rating,
                query_text=payload.query_text,
            )
            db.add(pg_event)
            db.commit()
        except Exception as e:
            logger.error("Failed to persist event to PostgreSQL: %s", e)
            db.rollback()

        ol = _online_learner_getter() if _online_learner_getter else None
        if payload.event_type in {"click", "rating"} and ol is not None:
            try:
                ol.enqueue(event_payload)
            except Exception as exc:
                logger.warning("OnlineLearner.enqueue failed: %s", exc)

        try:
            from backend.realtime_feature_updater import update_user_index

            update_user_index(event_payload)
        except Exception as exc:
            logger.warning("Real-time index update failed: %s", exc)

        if payload.event_type == "rating" and payload.movie_id and payload.rating is not None:
            if payload.rating >= 4.0:
                background_tasks.add_task(_trigger_active_inference_fn, payload.movie_id, 1.0)
            elif payload.rating <= 2.0:
                background_tasks.add_task(_trigger_active_inference_fn, payload.movie_id, -1.0)

        if payload.movie_id and payload.event_type in ["click", "rating"]:
            try:
                from backend.contextual_bandit import get_bandit_engine

                bandit = get_bandit_engine()
                is_success = payload.event_type == "click" or (payload.event_type == "rating" and payload.rating >= 4.0)
                bandit.update_reward(payload.movie_id, clicked=is_success)
            except Exception as e:
                logger.error("Bandit Engine failed to update reward: %s", e)

        rec_ref = _recommender_getter() if _recommender_getter else None
        if rec_ref is not None:
            rec_ref.refresh_behavior_features(force=True)

        record_usage(
            "events.write",
            event_payload["tenant_id"],
            event_payload["catalog_id"],
            plan=context.plan,
            authenticated=context.authenticated,
        )
        storage = event_storage_status()
        return EventResponse(
            status="accepted",
            event_id=event["event_id"],
            event_path=str(event.get("event_log_path") or get_events_path()),
            event_store=str(event.get("event_store") or storage["event_store"]),
            durable=bool(event.get("durable") or storage["durable"]),
        )

    @router.get("/v1/events/features")
    @router.get("/events/features")
    async def get_behavior_features(
        limit: int = Query(default=20, ge=1, le=100), context=Depends(resolve_tenant_context)
    ):
        record_usage(
            "events.features",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return aggregate_behavior_features(limit=limit)

    @router.get("/v1/events/recommendation-analytics")
    async def recommendation_event_analytics(
        limit: int = Query(default=20, ge=1, le=100), context=Depends(resolve_tenant_context)
    ):
        record_usage(
            "events.recommendation_analytics",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return summarize_recommendation_events(limit=limit)

    return router


def create_rec_engine_router(
    *,
    get_rec,
    record_usage,
    remote_payload_or_raise,
    record_recommendation_events,
    resolve_tenant_context,
    build_user_behavior_profile,
    assign_experiment,
    attach_experiment,
    generate_chat_response,
    limiter,
    Movie,
    RecommendationResponse,
    EnrichedRecommendationResponse,
    EnrichedMovie,
):
    """Recommendation engine endpoints (by-id, by-title, by-user, visual, KG, diagnostics, chat)."""
    _inject_pydantic_models_into_globals()
    from urllib.parse import quote

    router = APIRouter()

    # ── /v1/diagnostics/recommendations/{movie_id} ───────────────────────────
    @router.get("/v1/diagnostics/recommendations/{movie_id}")
    async def recommendation_diagnostics(
        movie_id: int,
        n: int = Query(default=10, ge=1, le=50),
        context=Depends(resolve_tenant_context),
    ):
        remote_payload = await remote_payload_or_raise(
            f"/v1/diagnostics/recommendations/{movie_id}", params={"n": n}, context=context
        )
        if remote_payload is not None:
            record_usage(
                "diagnostics.recommendations.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload
        rec = await run_in_threadpool(get_rec)
        query_movie = rec.get_movie_by_id(movie_id)
        if query_movie is None:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        recommendations = await run_in_threadpool(lambda: rec.recommend_by_id(movie_id, n=n))
        report = _recommendation_diagnostic_report_fn(
            context=context, rec=rec, query_movie=query_movie, recommendations=recommendations, k=n
        )
        record_usage(
            "diagnostics.recommendations",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return report

    # ── /v1/recommendations/visually-similar/{movie_id} ─────────────────────
    @router.get("/v1/recommendations/visually-similar/{movie_id}", response_model=RecommendationResponse)
    async def visual_recommendation_by_id(
        movie_id: int,
        background_tasks: BackgroundTasks,
        request: Request,
        context=Depends(resolve_tenant_context),
        n: int = Query(default=10, ge=1, le=100),
        explain: bool = Query(default=False),
    ):
        request_id = str(uuid.uuid4())
        rec = await run_in_threadpool(get_rec)
        query_movie = rec.get_movie_by_id(movie_id)
        if query_movie is None:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        if getattr(rec, "multimodal_index", None) is None:
            raise HTTPException(status_code=503, detail="Visual Search is currently disabled due to missing artifacts.")
        recommendations = await run_in_threadpool(lambda: rec.visual_search(movie_id, n=n))
        record_usage(
            "recommendations.visual",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return RecommendationResponse(
            request_id=request_id, query_movie=query_movie, recommendations=recommendations or []
        )

    # ── /v1/recommendations/knowledge-graph/{movie_id} ──────────────────────
    @router.get("/v1/recommendations/knowledge-graph/{movie_id}", response_model=RecommendationResponse)
    async def kg_recommendation_by_id(
        movie_id: int,
        background_tasks: BackgroundTasks,
        request: Request,
        context=Depends(resolve_tenant_context),
        n: int = Query(default=10, ge=1, le=100),
    ):
        request_id = str(uuid.uuid4())
        rec = await run_in_threadpool(get_rec)
        query_movie = rec.get_movie_by_id(movie_id)
        if query_movie is None:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        if getattr(rec, "kg_engine", None) is None or not getattr(rec.kg_engine, "graph", None):
            raise HTTPException(
                status_code=503, detail="Knowledge Graph is currently disabled due to missing artifacts."
            )
        recommendations = await run_in_threadpool(lambda: rec.kg_recommend(movie_id, n=n))
        record_usage(
            "recommendations.knowledge_graph",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return RecommendationResponse(
            request_id=request_id, query_movie=query_movie, recommendations=recommendations or []
        )

    # ── /v1/recommendations/id/{movie_id} ────────────────────────────────────
    @router.get("/v1/recommendations/id/{movie_id}", response_model=RecommendationResponse)
    @router.get("/recommend/id/{movie_id}", response_model=RecommendationResponse)
    async def recommend_by_id(
        movie_id: int,
        background_tasks: BackgroundTasks,
        n: int = Query(default=10, le=50),
        request_id: str | None = Query(default=None),
        user_id: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
        explain: bool = Query(default=False),
        context=Depends(resolve_tenant_context),
    ):
        resolved_request_id = request_id or str(uuid.uuid4())
        remote_payload = await remote_payload_or_raise(
            f"/v1/recommendations/id/{movie_id}",
            params={
                "n": n,
                "request_id": resolved_request_id,
                "user_id": user_id,
                "session_id": session_id,
                "explain": explain,
            },
            context=context,
        )
        if remote_payload is not None:
            if isinstance(remote_payload, dict):
                background_tasks.add_task(
                    record_recommendation_events,
                    endpoint="recommendations.id.remote",
                    context=context,
                    query_movie=remote_payload.get("query_movie") or {"id": movie_id},
                    recommendations=list(remote_payload.get("recommendations") or []),
                    rec=None,
                    request_id=resolved_request_id,
                    user_id=user_id,
                    session_id=session_id,
                )
                remote_payload.setdefault("request_id", resolved_request_id)
            record_usage(
                "recommendations.id.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload
        rec = get_rec()
        query_movie = rec.get_movie_by_id(movie_id)
        if query_movie is None:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        recommendations = await run_in_threadpool(lambda: rec.recommend_by_id(movie_id, n=n))
        if explain:
            await _apply_llm_explanations(recommendations, user_id or "anonymous")
        background_tasks.add_task(
            record_recommendation_events,
            endpoint="recommendations.id",
            context=context,
            query_movie=query_movie,
            recommendations=recommendations,
            rec=rec,
            request_id=resolved_request_id,
            user_id=user_id,
            session_id=session_id,
        )
        record_usage(
            "recommendations.id",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return RecommendationResponse(
            request_id=resolved_request_id, query_movie=query_movie, recommendations=recommendations
        )

    # ── /v1/recommendations/id/{movie_id}/enriched ───────────────────────────
    @router.get("/v1/recommendations/id/{movie_id}/enriched", response_model=EnrichedRecommendationResponse)
    @router.get("/recommend/id/{movie_id}/enriched", response_model=EnrichedRecommendationResponse)
    async def recommend_by_id_enriched(
        movie_id: int,
        background_tasks: BackgroundTasks,
        n: int = Query(default=10, le=50),
        request_id: str | None = Query(default=None),
        user_id: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
        context=Depends(resolve_tenant_context),
    ):
        resolved_request_id = request_id or str(uuid.uuid4())
        remote_payload = await remote_payload_or_raise(
            f"/v1/recommendations/id/{movie_id}/enriched",
            params={"n": n, "request_id": resolved_request_id, "user_id": user_id, "session_id": session_id},
            context=context,
        )
        if remote_payload is not None:
            if isinstance(remote_payload, dict):
                background_tasks.add_task(
                    record_recommendation_events,
                    endpoint="recommendations.id.enriched.remote",
                    context=context,
                    query_movie=remote_payload.get("query_movie") or {"id": movie_id},
                    recommendations=list(remote_payload.get("recommendations") or []),
                    rec=None,
                    request_id=resolved_request_id,
                    user_id=user_id,
                    session_id=session_id,
                )
                remote_payload.setdefault("request_id", resolved_request_id)
            record_usage(
                "recommendations.id.enriched.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload
        rec = get_rec()
        query_movie = rec.get_movie_by_id(movie_id)
        if query_movie is None:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        recommendations = await run_in_threadpool(lambda: rec.recommend_by_id(movie_id, n=n))
        background_tasks.add_task(
            record_recommendation_events,
            endpoint="recommendations.id.enriched",
            context=context,
            query_movie=query_movie,
            recommendations=recommendations,
            rec=rec,
            request_id=resolved_request_id,
            user_id=user_id,
            session_id=session_id,
        )
        record_usage(
            "recommendations.id.enriched",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        enriched = await asyncio.gather(*[enrich_movie(m) for m in recommendations])
        return EnrichedRecommendationResponse(
            request_id=resolved_request_id, query_movie=query_movie, recommendations=enriched
        )

    # ── /v1/recommendations/title/{title} ────────────────────────────────────
    @router.get("/v1/recommendations/title/{title}", response_model=RecommendationResponse)
    @router.get("/recommend/title/{title}", response_model=RecommendationResponse)
    async def recommend_by_title(
        title: str,
        background_tasks: BackgroundTasks,
        n: int = Query(default=10, le=50),
        request_id: str | None = Query(default=None),
        user_id: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
        context=Depends(resolve_tenant_context),
    ):
        resolved_request_id = request_id or str(uuid.uuid4())
        remote_payload = await remote_payload_or_raise(
            f"/v1/recommendations/title/{quote(title, safe='')}",
            params={"n": n, "request_id": resolved_request_id, "user_id": user_id, "session_id": session_id},
            context=context,
        )
        if remote_payload is not None:
            if isinstance(remote_payload, dict):
                background_tasks.add_task(
                    record_recommendation_events,
                    endpoint="recommendations.title.remote",
                    context=context,
                    query_movie=remote_payload.get("query_movie") or {"title": title},
                    recommendations=list(remote_payload.get("recommendations") or []),
                    rec=None,
                    request_id=resolved_request_id,
                    user_id=user_id,
                    session_id=session_id,
                )
                remote_payload.setdefault("request_id", resolved_request_id)
            record_usage(
                "recommendations.title.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload
        rec = get_rec()
        matches = await run_in_threadpool(lambda: rec.search_movies(title, limit=1))
        if not matches:
            raise HTTPException(status_code=404, detail=f"Movie '{title}' not found")
        query_movie = matches[0]
        recommendations = await run_in_threadpool(lambda: rec.recommend_by_title(title, n=n))
        background_tasks.add_task(
            record_recommendation_events,
            endpoint="recommendations.title",
            context=context,
            query_movie=query_movie,
            recommendations=recommendations,
            rec=rec,
            request_id=resolved_request_id,
            user_id=user_id,
            session_id=session_id,
        )
        record_usage(
            "recommendations.title",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return RecommendationResponse(
            request_id=resolved_request_id, query_movie=query_movie, recommendations=recommendations
        )

    # ── /v1/recommendations/user/{user_id} ───────────────────────────────────
    @router.get("/v1/recommendations/user/{user_id}", response_model=list[Movie])
    async def recommend_for_user(
        user_id: str,
        background_tasks: BackgroundTasks,
        n: int = Query(default=10, le=50),
        limit: int | None = Query(default=None, ge=1, le=50),
        top_k: int | None = Query(default=None, ge=1, le=50),
        request_id: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
        context=Depends(resolve_tenant_context),
    ):
        result_limit = top_k or limit or n
        resolved_request_id = request_id or str(uuid.uuid4())
        remote_payload = await remote_payload_or_raise(
            f"/v1/recommendations/user/{quote(user_id, safe='')}",
            params={
                "n": n,
                "limit": limit,
                "top_k": top_k,
                "request_id": resolved_request_id,
                "session_id": session_id,
            },
            context=context,
        )
        if remote_payload is not None:
            if isinstance(remote_payload, list):
                background_tasks.add_task(
                    record_recommendation_events,
                    endpoint="recommendations.user.remote",
                    context=context,
                    query_movie={"id": None, "title": f"user:{user_id}"},
                    recommendations=remote_payload,
                    rec=None,
                    request_id=resolved_request_id,
                    user_id=user_id,
                    session_id=session_id,
                )
            record_usage(
                "recommendations.user.remote",
                context.tenant_id,
                context.catalog_id,
                plan=context.plan,
                authenticated=context.authenticated,
            )
            return remote_payload
        rec = get_rec()
        profile = await run_in_threadpool(lambda: build_user_behavior_profile(user_id, limit=12))
        assignment = assign_experiment(subject_id=user_id)
        results = await run_in_threadpool(lambda: rec.recommend_for_user_profile(profile, n=result_limit))
        results = attach_experiment(results, assignment)
        background_tasks.add_task(
            record_recommendation_events,
            endpoint="recommendations.user",
            context=context,
            query_movie={
                "id": profile["seed_movie_ids"][0] if profile.get("seed_movie_ids") else None,
                "title": f"user:{user_id}",
            },
            recommendations=results,
            rec=rec,
            request_id=resolved_request_id,
            user_id=user_id,
            session_id=session_id,
        )
        record_usage(
            "recommendations.user",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return results

    # ── /chat ─────────────────────────────────────────────────────────────────
    @router.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        try:
            msgs = [m.model_dump() for m in request.messages]
            response = generate_chat_response(msgs)
            return ChatResponse(**response)
        except Exception as e:
            logger.error("Chat endpoint failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    return router


# ---------------------------------------------------------------------------
# Diagnostic helpers — moved from backend/main.py (task 6.3)
# ---------------------------------------------------------------------------
from backend.app_info import app_metadata
from backend.auth import TenantContext
from backend.recommendation_benchmark import (
    evaluate_recommendation_case,
    find_recommendation_benchmark_case,
    load_recommendation_benchmark,
)
from backend.recommendation_events import _serving_lineage
from backend.recommender import Recommender
from backend.recommender_helpers import safe_float as _safe_float


def _candidate_diagnostic_summary(candidate: dict, rank: int) -> dict:
    """Return recommendation evidence that is safe to expose to product/debug clients."""
    return {
        "rank": rank,
        "id": candidate.get("id"),
        "title": candidate.get("title"),
        "score": _safe_float(candidate.get("similarity_score")),
        "ranker_score": _safe_float(candidate.get("ranker_score")),
        "retrieval_stage": candidate.get("retrieval_stage") or "unknown",
        "explanation": candidate.get("explanation") or [],
        "explanation_text": candidate.get("explanation_text"),
        "retrieval_signals": candidate.get("retrieval_signals") or {},
    }


def _recommendation_diagnostic_report(
    *,
    context: TenantContext,
    rec: Recommender,
    query_movie: dict,
    recommendations: list[dict],
    k: int,
) -> dict:
    """Build a compact per-seed report for ranking explainability and support."""
    diagnostic_items = [
        _candidate_diagnostic_summary(candidate, rank) for rank, candidate in enumerate(recommendations[:k], start=1)
    ]
    stage_counts = Counter(item["retrieval_stage"] for item in diagnostic_items)
    explained_count = sum(1 for item in diagnostic_items if item.get("explanation") or item.get("explanation_text"))
    scores = [item["score"] for item in diagnostic_items if item.get("score") is not None]

    benchmark_case = find_recommendation_benchmark_case(
        query_movie,
        cases=load_recommendation_benchmark(),
    )
    benchmark_summary = None
    if benchmark_case is not None:
        benchmark_summary = evaluate_recommendation_case(
            recommendations,
            benchmark_case,
            k=k,
            seed_movie=query_movie,
        )
        benchmark_summary.pop("_aggregate", None)

    return {
        "status": "ok",
        "app": app_metadata(),
        "tenant_id": context.tenant_id,
        "catalog_id": context.catalog_id,
        "query_movie": {
            "id": query_movie.get("id"),
            "title": query_movie.get("title"),
            "genres": query_movie.get("genres"),
            "release_date": query_movie.get("release_date"),
            "vote_average": query_movie.get("vote_average"),
            "vote_count": query_movie.get("vote_count"),
        },
        "lineage": _serving_lineage(rec),
        "diagnostics": {
            "requested_k": k,
            "result_count": len(diagnostic_items),
            "stage_distribution": dict(sorted(stage_counts.items())),
            "explanation_coverage": round(explained_count / max(len(diagnostic_items), 1), 4),
            "average_similarity_score": (round(sum(scores) / len(scores), 6) if scores else None),
            "benchmark_case_available": benchmark_summary is not None,
            "benchmark_case_passed": (benchmark_summary.get("passed") if benchmark_summary is not None else None),
        },
        "benchmark_case": benchmark_summary,
        "recommendations": diagnostic_items,
    }
