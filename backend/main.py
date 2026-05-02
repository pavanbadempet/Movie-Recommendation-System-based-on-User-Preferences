"""
FastAPI backend for the Movie Recommendation System.
Provides REST API endpoints for movie search and recommendations.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional
from urllib.parse import quote

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.events import append_event, aggregate_behavior_features, build_user_behavior_profile, event_storage_status, get_events_path
from backend.evaluation import evaluate_recommendation_quality
from backend.experiments import assign_experiment, attach_experiment, summarize_experiment_metrics
from backend.auth import TenantContext, enforce_payload_context, resolve_tenant_context
from backend.catalogs import profile_catalog_csv, persist_catalog_upload
from backend.recommender import get_recommender, Recommender
from backend.remote_recommender import remote_get_json
from backend.usage import record_usage, summarize_usage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sentry (Error Monitoring)
import sentry_sdk
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
        )
    except Exception as e:
        logger.warning("SENTRY_DSN is invalid. Error monitoring disabled: %s", e)
    else:
        logger.info("Sentry monitoring enabled.")
else:
    logger.warning("SENTRY_DSN not set. Error monitoring disabled.")

# TMDB API config
TMDB_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"

# Async HTTP client (initialized via lifespan)
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage async resources for app lifetime."""
    global http_client
    http_client = httpx.AsyncClient(timeout=10.0)
    yield
    await http_client.aclose()


# Create FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommendation engine using FAISS",
    version="2.0.0",
    lifespan=lifespan,
)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Welcome to the Movie Recommendation API. Head over to /docs to explore the endpoints!",
        "version": "2.0.0"
    }

# Rate limiting (30 requests/minute per IP)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration for Streamlit frontend
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == [""]:
    ALLOWED_ORIGINS = [
        "https://movie-recommendation-system.streamlit.app",
        "http://localhost:8501",
        "http://localhost:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class Movie(BaseModel):
    """Movie response model."""
    id: int
    title: str
    overview: Optional[str] = None
    genres: Optional[str] = None
    vote_average: Optional[float] = None
    vote_count: Optional[float] = None
    popularity: Optional[float] = None
    release_date: Optional[str] = None
    poster_path: Optional[str] = None
    metadata_completeness: Optional[float] = None
    content_quality_score: Optional[float] = None
    quality_bucket: Optional[str] = None
    recommendable: Optional[bool] = None
    similarity_score: Optional[float] = None
    ranker_score: Optional[float] = None
    retrieval_stage: Optional[str] = None
    retrieval_signals: Optional[dict] = None
    explanation_text: Optional[str] = None
    explanation: Optional[list[str]] = None


class MovieTitle(BaseModel):
    """Lightweight movie title response for autocomplete."""
    id: int
    title: str


class EnrichedMovie(BaseModel):
    """Movie with TMDB enrichment data."""
    id: int
    title: str
    overview: Optional[str] = None
    genres: Optional[str] = None
    vote_average: Optional[float] = None
    vote_count: Optional[float] = None
    popularity: Optional[float] = None
    release_date: Optional[str] = None
    poster_path: Optional[str] = None
    metadata_completeness: Optional[float] = None
    content_quality_score: Optional[float] = None
    quality_bucket: Optional[str] = None
    recommendable: Optional[bool] = None
    similarity_score: Optional[float] = None
    explanation_text: Optional[str] = None
    explanation: Optional[list[str]] = None
    # Enriched fields
    trailer_key: Optional[str] = None
    runtime: Optional[int] = None
    director: Optional[str] = None
    cast: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    movie_count: int


class RecommendationResponse(BaseModel):
    """Recommendation response."""
    query_movie: Movie
    recommendations: list[Movie]


class EnrichedRecommendationResponse(BaseModel):
    """Enriched recommendation response with TMDB data."""
    query_movie: Movie
    recommendations: list[EnrichedMovie]


class EventRequest(BaseModel):
    """Behavior event request model."""
    event_type: Literal[
        "view",
        "search",
        "click",
        "rating",
        "recommendation_impression",
    ]
    tenant_id: Optional[str] = None
    catalog_id: Optional[str] = None
    content_id: Optional[str] = None
    source_content_id: Optional[str] = None
    movie_id: Optional[int] = None
    query_text: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    rating: Optional[float] = None
    request_id: Optional[str] = None
    metadata: Optional[dict] = None


class EventResponse(BaseModel):
    """Behavior event write response."""
    status: str
    event_id: str
    event_path: str
    event_store: str
    durable: bool


class PlatformContextResponse(BaseModel):
    """Resolved product API context."""
    tenant_id: str
    catalog_id: str
    plan: str
    authenticated: bool
    mode: str


class UsageResponse(BaseModel):
    """Lightweight API usage summary."""
    generated_at: str
    usage_log_path: str
    total_requests: int
    last_seen: Optional[str] = None
    operation_counts: dict[str, int]
    tenant_counts: dict[str, int]


class CatalogPreviewRequest(BaseModel):
    """CSV catalog preview request from the Nova Console."""
    filename: str = "catalog.csv"
    csv_text: str
    column_mapping: dict[str, str] = Field(default_factory=dict)
    sample_size: int = 20


class CatalogUploadRequest(CatalogPreviewRequest):
    """Persisted catalog upload request."""
    pass


# Lazy-load recommender on first request
_recommender: Recommender | None = None


def get_rec() -> Recommender:
    """Get recommender instance, loading on first call."""
    global _recommender
    if _recommender is None:
        logger.info("Loading recommender on first request...")
        _recommender = get_recommender()
    return _recommender


async def remote_payload_or_raise(
    path: str,
    params: dict | None = None,
    context: TenantContext | None = None,
) -> object | None:
    """Return remote recommender payload when configured, otherwise None."""
    remote_response = await remote_get_json(path, params=params, context=context)
    if remote_response is None:
        return None
    if remote_response.status_code >= 400:
        detail = remote_response.payload
        if isinstance(remote_response.payload, dict) and "detail" in remote_response.payload:
            detail = remote_response.payload["detail"]
        raise HTTPException(status_code=remote_response.status_code, detail=detail)
    return remote_response.payload


# ===== ASYNC TMDB FETCH FUNCTIONS =====

async def fetch_trailer(movie_id: int) -> str | None:
    """Fetch trailer key from TMDB."""
    try:
        r = await http_client.get(
            f"{TMDB_BASE}/movie/{movie_id}/videos",
            params={"api_key": TMDB_KEY, "language": "en-US"}
        )
        data = r.json()
        for v in data.get("results", []):
            if v.get("type") == "Trailer":
                return v.get("key")
        if data.get("results"):
            return data["results"][0].get("key")
    except Exception as e:
        logger.warning(f"Trailer fetch failed for {movie_id}: {e}")
    return None


async def fetch_details(movie_id: int) -> dict:
    """Fetch movie details from TMDB."""
    try:
        r = await http_client.get(
            f"{TMDB_BASE}/movie/{movie_id}",
            params={"api_key": TMDB_KEY}
        )
        return r.json()
    except Exception as e:
        logger.warning(f"Details fetch failed for {movie_id}: {e}")
    return {}


async def fetch_credits(movie_id: int) -> dict:
    """Fetch cast and crew from TMDB."""
    try:
        r = await http_client.get(
            f"{TMDB_BASE}/movie/{movie_id}/credits",
            params={"api_key": TMDB_KEY}
        )
        data = r.json()
        cast = [c["name"] for c in data.get("cast", [])[:3]]
        director = next(
            (c["name"] for c in data.get("crew", []) if c.get("job") == "Director"),
            "Unknown"
        )
        return {"cast": ", ".join(cast), "director": director}
    except Exception as e:
        logger.warning(f"Credits fetch failed for {movie_id}: {e}")
    return {"cast": "N/A", "director": "N/A"}


async def enrich_movie(movie: dict) -> dict:
    """Enrich a single movie with all TMDB data in parallel."""
    movie_id = movie["id"]
    
    # Fetch all 3 APIs in parallel
    trailer, details, credits = await asyncio.gather(
        fetch_trailer(movie_id),
        fetch_details(movie_id),
        fetch_credits(movie_id)
    )
    
    return {
        **movie,
        "trailer_key": trailer,
        "runtime": details.get("runtime"),
        "director": credits.get("director"),
        "cast": credits.get("cast"),
    }


# ===== API ENDPOINTS =====

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    load_recommender = os.getenv("NOVA_HEALTH_LOAD_RECOMMENDER", "true").strip().lower()
    if load_recommender in {"0", "false", "no", "off"}:
        return HealthResponse(status="healthy", movie_count=0)

    try:
        rec = get_rec()
        return HealthResponse(status="healthy", movie_count=len(rec.movies))
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="unhealthy", movie_count=0)


@app.get("/v1/platform/context", response_model=PlatformContextResponse)
async def platform_context(
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Return the resolved tenant/catalog context for SDK and console clients."""
    return PlatformContextResponse(
        tenant_id=context.tenant_id,
        catalog_id=context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
        mode="authenticated" if context.authenticated else "public-demo",
    )


@app.get("/v1/platform/status")
async def platform_status(
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Return product-readiness status across serving, data, AI, and experimentation."""
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
        "experimentation": {
            "enabled": True,
            "default_assignment": assignment,
        },
        "capabilities": [
            "hybrid_ai_search",
            "learned_ranker",
            "personalization_v2",
            "experiment_metrics",
            "durable_event_store",
            "daily_artifact_refresh",
        ],
    }


@app.get("/v1/usage", response_model=UsageResponse)
async def usage_summary(
    context: TenantContext = Depends(resolve_tenant_context),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Return free-tier-safe API usage metrics."""
    record_usage(
        "usage.summary",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return summarize_usage(limit=limit)


@app.get("/v1/evaluation/recommendations")
async def recommendation_quality_report(
    context: TenantContext = Depends(resolve_tenant_context),
    sample_size: int = Query(default=25, ge=1, le=200),
    k: int = Query(default=10, ge=1, le=50),
):
    """Return label-free recommendation quality metrics for the current artifacts."""
    rec = get_rec()
    report = evaluate_recommendation_quality(rec, sample_size=sample_size, k=k)
    record_usage(
        "evaluation.recommendations",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return report


@app.get("/v1/ranker/status")
async def ranker_status(
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Return learned ranker artifact status and metadata."""
    rec = get_rec()
    ranker = getattr(rec, "_learned_ranker", None)
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
    return {
        "available": True,
        "feature_columns": ranker.feature_columns,
        "metadata": ranker.metadata,
    }


@app.get("/v1/experiments/assignment")
async def experiment_assignment(
    user_id: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
    experiment: Optional[str] = Query(default=None),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Return a deterministic experiment assignment for a user or session."""
    subject_id = user_id or session_id or f"{context.tenant_id}:{context.catalog_id}:anonymous"
    assignment = assign_experiment(subject_id=subject_id, experiment_name=experiment)
    record_usage(
        "experiments.assignment",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return assignment


@app.get("/v1/experiments/metrics")
async def experiment_metrics(
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Return experiment outcome metrics derived from behavior events."""
    record_usage(
        "experiments.metrics",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return summarize_experiment_metrics()


@app.post("/v1/catalog/preview")
async def preview_catalog(
    payload: CatalogPreviewRequest,
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Profile a customer CSV catalog before ingestion."""
    try:
        profile = profile_catalog_csv(
            payload.csv_text,
            tenant_id=context.tenant_id,
            catalog_id=context.catalog_id,
            column_mapping=payload.column_mapping,
            sample_size=payload.sample_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record_usage(
        "catalog.preview",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return profile


@app.post("/v1/catalog/upload")
async def upload_catalog(
    payload: CatalogUploadRequest,
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Persist a raw customer catalog upload plus manifest on local storage."""
    try:
        manifest = persist_catalog_upload(
            payload.csv_text,
            tenant_id=context.tenant_id,
            catalog_id=context.catalog_id,
            filename=payload.filename,
            column_mapping=payload.column_mapping,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record_usage(
        "catalog.upload",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return {
        "status": "stored",
        "upload_id": manifest["upload_id"],
        "raw_path": manifest["raw_path"],
        "manifest_path": manifest["manifest_path"],
        "profile": manifest["profile"],
    }


@app.get("/movies", response_model=list[Movie])
async def list_movies(
    limit: int = Query(default=100, le=1000, description="Maximum movies to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    """List movies with pagination."""
    remote_payload = await remote_payload_or_raise(
        "/movies",
        params={"limit": limit, "offset": offset},
    )
    if remote_payload is not None:
        return remote_payload

    rec = get_rec()
    movies = rec.movies.iloc[offset:offset + limit]
    return movies.to_dict(orient="records")


@app.get("/movies/titles", response_model=list[MovieTitle])
async def get_all_titles():
    """
    Get a lightweight list of all movie titles and IDs.
    Perfect for populating the Streamlit autocomplete dropdown.
    """
    remote_payload = await remote_payload_or_raise("/movies/titles")
    if remote_payload is not None:
        return remote_payload

    rec = get_rec()
    return rec.get_all_titles()


@app.get("/v1/search", response_model=list[Movie])
@app.get("/search", response_model=list[Movie])
@limiter.limit("30/minute")
async def search_movies(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=20, le=100, description="Maximum results"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Search movies by title."""
    remote_payload = await remote_payload_or_raise(
        "/v1/search",
        params={"q": q, "limit": limit},
        context=context,
    )
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
        "search",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return results


@app.get("/v1/search/ai", response_model=list[Movie])
async def ai_search_movies(
    q: str = Query(..., min_length=1, description="Natural language search query"),
    limit: int = Query(default=20, le=100, description="Maximum results"),
    top_k: Optional[int] = Query(default=None, ge=1, le=100, description="Alias for maximum results"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Hybrid AI search using sparse recall, optional dense recall, reranking, and diversity."""
    result_limit = top_k or limit
    remote_payload = await remote_payload_or_raise(
        "/v1/search/ai",
        params={"q": q, "limit": limit, "top_k": result_limit},
        context=context,
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
        "search.ai",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return results


@app.get("/movie/{movie_id}", response_model=Movie)
async def get_movie(movie_id: int):
    """Get a movie by TMDB ID."""
    remote_payload = await remote_payload_or_raise(f"/movie/{movie_id}")
    if remote_payload is not None:
        return remote_payload

    rec = get_rec()
    movie = rec.get_movie_by_id(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
    return movie


@app.post("/v1/events", response_model=EventResponse)
@app.post("/events", response_model=EventResponse)
async def record_event(
    payload: EventRequest,
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Record one product behavior event for personalization and analytics."""
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

    if _recommender is not None:
        _recommender.refresh_behavior_features(force=True)

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


@app.get("/v1/events/features")
@app.get("/events/features")
async def get_behavior_features(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum feature rows"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Return aggregated behavior features used by the recommender."""
    record_usage(
        "events.features",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return aggregate_behavior_features(limit=limit)


@app.get("/v1/recommendations/id/{movie_id}", response_model=RecommendationResponse)
@app.get("/recommend/id/{movie_id}", response_model=RecommendationResponse)
async def recommend_by_id(
    movie_id: int,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Get recommendations for a movie by TMDB ID."""
    remote_payload = await remote_payload_or_raise(
        f"/v1/recommendations/id/{movie_id}",
        params={"n": n},
        context=context,
    )
    if remote_payload is not None:
        record_usage(
            "recommendations.id.remote",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return remote_payload

    rec = get_rec()
    
    # Get query movie
    query_movie = rec.get_movie_by_id(movie_id)
    if query_movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
    
    # Get recommendations
    recommendations = rec.recommend_by_id(movie_id, n=n)
    record_usage(
        "recommendations.id",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    
    return RecommendationResponse(
        query_movie=query_movie,
        recommendations=recommendations,
    )


@app.get("/v1/recommendations/id/{movie_id}/enriched", response_model=EnrichedRecommendationResponse)
@app.get("/recommend/id/{movie_id}/enriched", response_model=EnrichedRecommendationResponse)
async def recommend_by_id_enriched(
    movie_id: int,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Get recommendations with FULL TMDB data (trailers, cast, etc) - PARALLEL FETCH."""
    remote_payload = await remote_payload_or_raise(
        f"/v1/recommendations/id/{movie_id}/enriched",
        params={"n": n},
        context=context,
    )
    if remote_payload is not None:
        record_usage(
            "recommendations.id.enriched.remote",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return remote_payload

    rec = get_rec()
    
    # Get query movie
    query_movie = rec.get_movie_by_id(movie_id)
    if query_movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
    
    # Get recommendations
    recommendations = rec.recommend_by_id(movie_id, n=n)
    record_usage(
        "recommendations.id.enriched",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    
    # Enrich all movies in parallel
    enriched = await asyncio.gather(*[enrich_movie(m) for m in recommendations])
    
    return EnrichedRecommendationResponse(
        query_movie=query_movie,
        recommendations=enriched,
    )


@app.get("/v1/recommendations/title/{title}", response_model=RecommendationResponse)
@app.get("/recommend/title/{title}", response_model=RecommendationResponse)
async def recommend_by_title(
    title: str,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Get recommendations for a movie by title."""
    remote_payload = await remote_payload_or_raise(
        f"/v1/recommendations/title/{quote(title, safe='')}",
        params={"n": n},
        context=context,
    )
    if remote_payload is not None:
        record_usage(
            "recommendations.title.remote",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return remote_payload

    rec = get_rec()
    
    # Search for the movie
    matches = rec.search_movies(title, limit=1)
    if not matches:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found")
    
    query_movie = matches[0]
    
    # Get recommendations
    recommendations = rec.recommend_by_title(title, n=n)
    record_usage(
        "recommendations.title",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    
    return RecommendationResponse(
        query_movie=query_movie,
        recommendations=recommendations,
    )


@app.get("/v1/recommendations/user/{user_id}", response_model=list[Movie])
async def recommend_for_user(
    user_id: str,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    limit: Optional[int] = Query(default=None, ge=1, le=50, description="Alias for number of recommendations"),
    top_k: Optional[int] = Query(default=None, ge=1, le=50, description="Alias for number of recommendations"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Personalize recommendations from a user's recent implicit feedback events."""
    rec = get_rec()
    result_limit = top_k or limit or n
    profile = build_user_behavior_profile(user_id, limit=12)
    assignment = assign_experiment(subject_id=user_id)
    results = rec.recommend_for_user_profile(profile, n=result_limit)
    results = attach_experiment(results, assignment)
    record_usage(
        "recommendations.user",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return results


# ===== CHATBOT (RAG) =====
from backend.chat import generate_chat_response

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class ChatResponse(BaseModel):
    role: str
    content: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    RAG Chatbot Endpoint.
    Consumes user messages, retrieves movie context, and generates AI response.
    """
    try:
        # Convert Pydantic models to dicts for internal function
        msgs = [m.model_dump() for m in request.messages]
        response = generate_chat_response(msgs)
        return ChatResponse(**response)
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

