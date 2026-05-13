"""
FastAPI backend for the Movie Recommendation System.
Provides REST API endpoints for movie search and recommendations.
"""
import asyncio
import gc
import logging
import os
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Literal, Optional
from urllib.parse import quote

import httpx
import sentry_sdk
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool

from backend.events import append_event, aggregate_behavior_features, build_user_behavior_profile, event_storage_status, get_events_path, summarize_recommendation_events
from backend.evaluation import evaluate_recommendation_quality
from backend.experiments import assign_experiment, attach_experiment, summarize_experiment_metrics
from backend.auth import TenantContext, enforce_payload_context, resolve_admin_token, resolve_tenant_context
from backend.artifact_health import evaluate_artifact_health
from backend.catalogs import profile_catalog_csv, persist_catalog_upload
from backend.chat import generate_chat_response
from backend.recommender import get_recommender, Recommender
from backend.remote_recommender import remote_get_json
from backend.semantic_benchmark import evaluate_semantic_benchmark
from backend.usage import record_usage, summarize_usage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
APP_VERSION = "2.0.0"
REVISION_FILE = Path(__file__).resolve().parent.parent / "REVISION"

# Async HTTP client (initialized via lifespan)
http_client: httpx.AsyncClient | None = None
_warmup_thread: Thread | None = None
_warmup_thread_lock = Lock()
_semantic_benchmark_cache: dict[int, tuple[float, dict]] = {}
_semantic_benchmark_threads: dict[int, Thread] = {}
_semantic_benchmark_cache_lock = Lock()
_semantic_benchmark_compute_lock = Lock()


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage async resources for app lifetime."""
    global http_client
    http_client = httpx.AsyncClient(timeout=10.0)
    if _env_truthy("NOVA_BACKGROUND_RECOMMENDER_WARMUP"):
        _start_background_recommender_warmup()
    yield
    await http_client.aclose()


# Create FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommendation engine using FAISS",
    version=APP_VERSION,
    lifespan=lifespan,
)


def app_metadata() -> dict[str, str | None]:
    """Return deploy lineage without loading the recommender."""
    commit = None
    source = None
    for env_name in (
        "NOVA_APP_COMMIT",
        "RENDER_GIT_COMMIT",
        "SOURCE_VERSION",
        "GITHUB_SHA",
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            commit = value
            source = env_name
            break
    if not commit and REVISION_FILE.exists():
        try:
            value = REVISION_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
        else:
            if value:
                commit = value
                source = "REVISION"
    if not commit:
        value = os.getenv("COMMIT_SHA", "").strip()
        if value:
            commit = value
            source = "COMMIT_SHA"
    return {
        "version": APP_VERSION,
        "commit": commit[:12] if commit else None,
        "commit_full": commit if commit else None,
        "source": source,
    }


@app.get("/")
async def root():
    metadata = app_metadata()
    return {
        "status": "online",
        "message": "Welcome to the Movie Recommendation API. Head over to /docs to explore the endpoints!",
        "version": metadata["version"],
        "app": metadata,
    }

# Rate limiting (30 requests/minute per IP)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration for hosted frontends.
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = [
        "https://movie-recommendation-system.streamlit.app",
        "http://localhost:8501",
        "http://localhost:5173",
        "http://localhost:3000",
    ]
ALLOWED_ORIGIN_REGEX = os.getenv(
    "ALLOWED_ORIGIN_REGEX",
    r"https://([a-zA-Z0-9-]+\.)+(vercel\.app|pages\.dev|netlify\.app|github\.io)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
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
    semantic_twin: Optional[dict] = None
    semantic_signals: Optional[dict] = None
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
    retrieval_stage: Optional[str] = None
    retrieval_signals: Optional[dict] = None
    semantic_twin: Optional[dict] = None
    semantic_signals: Optional[dict] = None
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
    app_version: Optional[str] = None
    app_commit: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Recommendation response."""
    request_id: Optional[str] = None
    query_movie: Movie
    recommendations: list[Movie]


class EnrichedRecommendationResponse(BaseModel):
    """Enriched recommendation response with TMDB data."""
    request_id: Optional[str] = None
    query_movie: Movie
    recommendations: list[EnrichedMovie]


class EventRequest(BaseModel):
    """Behavior event request model."""
    event_type: Literal[
        "view",
        "search",
        "click",
        "rating",
        "recommendation_request",
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


def _background_recommender_warmup() -> None:
    """Warm the recommender after startup without blocking health probes."""
    try:
        logger.info("Starting background recommender warmup...")
        rec = get_rec()
        if _env_truthy("NOVA_PRECOMPUTE_SEMANTIC_BENCHMARK"):
            k = int(os.getenv("NOVA_SEMANTIC_BENCHMARK_K", "10"))
            _compute_semantic_benchmark_cached(rec, k=k)
        logger.info("Background recommender warmup completed.")
    except Exception as exc:
        logger.exception("Background recommender warmup failed: %s", exc)


def _start_background_recommender_warmup() -> None:
    """Start one daemon warmup thread per process."""
    global _warmup_thread
    with _warmup_thread_lock:
        if _warmup_thread is not None and _warmup_thread.is_alive():
            return
        _warmup_thread = Thread(
            target=_background_recommender_warmup,
            name="recommender-warmup",
            daemon=True,
        )
        _warmup_thread.start()


def _semantic_benchmark_ttl_seconds() -> int:
    return max(60, int(os.getenv("NOVA_SEMANTIC_BENCHMARK_CACHE_TTL_SECONDS", "3600")))


def _warming_semantic_benchmark_report(k: int) -> dict:
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "warming",
        "reason": "Semantic benchmark is warming in the background. Retry shortly.",
        "case_count": 0,
        "evaluated_case_count": 0,
        "skipped_case_count": 0,
        "k": k,
        "metrics": {},
        "cases": [],
        "skipped": [],
    }


def _get_cached_semantic_benchmark(k: int) -> dict | None:
    with _semantic_benchmark_cache_lock:
        cached = _semantic_benchmark_cache.get(k)
    if cached is None:
        return None
    cached_at, report = cached
    if time.time() - cached_at > _semantic_benchmark_ttl_seconds():
        return None
    return report


def _compute_semantic_benchmark_cached(rec: Recommender, k: int) -> dict:
    cached = _get_cached_semantic_benchmark(k)
    if cached is not None:
        return cached

    with _semantic_benchmark_compute_lock:
        cached = _get_cached_semantic_benchmark(k)
        if cached is not None:
            return cached
        report = evaluate_semantic_benchmark(rec, k=k)
        with _semantic_benchmark_cache_lock:
            _semantic_benchmark_cache[k] = (time.time(), report)
        return report


def _background_semantic_benchmark(k: int) -> None:
    try:
        rec = get_rec()
        _compute_semantic_benchmark_cached(rec, k=k)
    except Exception as exc:
        logger.exception("Background semantic benchmark failed: %s", exc)


def _start_background_semantic_benchmark(k: int) -> None:
    with _semantic_benchmark_cache_lock:
        thread = _semantic_benchmark_threads.get(k)
        if thread is not None and thread.is_alive():
            return
        thread = Thread(
            target=_background_semantic_benchmark,
            args=(k,),
            name=f"semantic-benchmark-{k}",
            daemon=True,
        )
        _semantic_benchmark_threads[k] = thread
        thread.start()


@contextmanager
def _temporary_env(overrides: dict[str, str | None]):
    """Temporarily override environment variables for one operation."""
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _artifact_refresh_env(force_download: bool) -> dict[str, str]:
    """Return model-loader env overrides for artifact reload operations."""
    overrides = {"NOVA_REFRESH_PIPELINE_MANIFEST": "1"}
    if force_download:
        overrides["FORCE_MODEL_REFRESH"] = "1"
    return overrides


def _reload_local_recommender(force_download: bool) -> Recommender:
    """Load a fresh recommender and atomically publish it to both singletons."""
    global _recommender

    from backend import recommender as recommender_module

    previous_main_recommender = _recommender
    previous_module_recommender = recommender_module._recommender
    try:
        with _temporary_env(_artifact_refresh_env(force_download)):
            fresh_recommender = recommender_module.Recommender().load()
    except Exception:
        _recommender = previous_main_recommender
        recommender_module._recommender = previous_module_recommender
        raise

    _recommender = fresh_recommender
    recommender_module._recommender = fresh_recommender
    gc.collect()
    return fresh_recommender


def _refresh_artifact_files(force_download: bool) -> dict[str, bool]:
    """Refresh serving artifact files without rebuilding the in-memory recommender."""
    from backend import recommender as recommender_module
    from backend.model_loader import default_artifacts_for_serving_profile, ensure_model_files

    with _temporary_env(_artifact_refresh_env(force_download)):
        return ensure_model_files(
            recommender_module.MODELS_DIR,
            selected_files=default_artifacts_for_serving_profile(),
        )


def _event_logging_enabled() -> bool:
    """Return whether recommendation serving should emit analytics events."""
    value = os.getenv("NOVA_RECOMMENDATION_EVENT_LOGGING", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return round(number, 6)


def _serving_lineage(rec: Recommender | None) -> dict:
    """Return compact model/artifact lineage for recommendation events."""
    if rec is None:
        return {"serving_path": "remote_gateway"}

    artifact_status = dict(getattr(rec, "_artifact_status", {}) or {})
    manifest = dict(getattr(rec, "_artifact_manifest", {}) or {})
    ranker = getattr(rec, "_learned_ranker", None)
    ranker_metadata = dict(getattr(ranker, "metadata", {}) or {}) if ranker is not None else {}
    return {
        "serving_path": "local",
        "manifest_run_id": artifact_status.get("manifest_run_id") or manifest.get("run_id"),
        "manifest_run_date": artifact_status.get("manifest_run_date") or manifest.get("run_date"),
        "vector_artifacts_ready": artifact_status.get("vector_artifacts_ready"),
        "movie_count": artifact_status.get("movie_count"),
        "vector_count": artifact_status.get("vector_count"),
        "faiss_index_count": artifact_status.get("faiss_index_count"),
        "ranker_available": ranker is not None,
        "ranker_training_mode": ranker_metadata.get("training_mode"),
        "ranker_promoted_at": ranker_metadata.get("promoted_at"),
    }


def _candidate_event_summary(candidate: dict, rank: int) -> dict:
    """Return the event-safe summary for one ranked recommendation."""
    return {
        "rank": rank,
        "movie_id": candidate.get("id"),
        "title": candidate.get("title"),
        "retrieval_stage": candidate.get("retrieval_stage"),
        "similarity_score": _safe_float(candidate.get("similarity_score")),
        "ranker_score": _safe_float(candidate.get("ranker_score")),
        "retrieval_signals": candidate.get("retrieval_signals") or {},
    }


def record_recommendation_events(
    *,
    endpoint: str,
    context: TenantContext,
    query_movie: dict,
    recommendations: list[dict],
    rec: Recommender | None,
    request_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> str:
    """Persist request and impression events for offline analysis and training labels."""
    resolved_request_id = request_id or str(uuid.uuid4())
    if not _event_logging_enabled():
        return resolved_request_id

    try:
        lineage = _serving_lineage(rec)
        ranked_candidates = [
            _candidate_event_summary(candidate, rank)
            for rank, candidate in enumerate(recommendations, start=1)
        ]
        stage_counts = Counter(
            str(candidate.get("retrieval_stage") or "unknown")
            for candidate in recommendations
        )
        common_payload = {
            "tenant_id": context.tenant_id,
            "catalog_id": context.catalog_id,
            "user_id": user_id,
            "session_id": session_id,
            "request_id": resolved_request_id,
            "source": "recommendation_api",
        }
        append_event(
            {
                **common_payload,
                "event_type": "recommendation_request",
                "movie_id": query_movie.get("id"),
                "metadata": {
                    "endpoint": endpoint,
                    "query_movie": {
                        "id": query_movie.get("id"),
                        "title": query_movie.get("title"),
                    },
                    "requested_count": len(recommendations),
                    "candidate_ids": [candidate.get("movie_id") for candidate in ranked_candidates],
                    "retrieval_stage_counts": dict(stage_counts),
                    "lineage": lineage,
                },
            }
        )
        for candidate in ranked_candidates:
            append_event(
                {
                    **common_payload,
                    "event_type": "recommendation_impression",
                    "movie_id": candidate.get("movie_id"),
                    "metadata": {
                        "endpoint": endpoint,
                        "seed_movie_id": query_movie.get("id"),
                        "seed_title": query_movie.get("title"),
                        "lineage": lineage,
                        **candidate,
                    },
                }
            )
    except Exception as exc:
        logger.warning("Recommendation event logging skipped: %s", exc)

    return resolved_request_id


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
    metadata = app_metadata()
    load_recommender = os.getenv("NOVA_HEALTH_LOAD_RECOMMENDER", "true").strip().lower()
    if load_recommender in {"0", "false", "no", "off"}:
        from backend import recommender as recommender_module

        report = evaluate_artifact_health(
            models_dir=recommender_module.MODELS_DIR,
            data_dir=recommender_module.DATA_DIR,
        )
        return HealthResponse(
            status="healthy" if report.get("files", {}).get("movies", {}).get("exists") else "degraded",
            movie_count=int((report.get("row_counts") or {}).get("movies") or 0),
            app_version=metadata["version"],
            app_commit=metadata["commit"],
        )

    try:
        rec = get_rec()
        return HealthResponse(
            status="healthy",
            movie_count=len(rec.movies),
            app_version=metadata["version"],
            app_commit=metadata["commit"],
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            movie_count=0,
            app_version=metadata["version"],
            app_commit=metadata["commit"],
        )


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
        "app": app_metadata(),
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
            "semantic_item_twins",
            "semantic_benchmark",
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
    rec = await run_in_threadpool(get_rec)
    report = await run_in_threadpool(
        lambda: evaluate_recommendation_quality(rec, sample_size=sample_size, k=k)
    )
    record_usage(
        "evaluation.recommendations",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return report


@app.get("/v1/evaluation/semantic-benchmark")
async def semantic_benchmark_report(
    context: TenantContext = Depends(resolve_tenant_context),
    k: int = Query(default=10, ge=1, le=50),
):
    """Return human-labeled semantic benchmark metrics for obvious bad-match detection."""
    cached_report = _get_cached_semantic_benchmark(k)
    if cached_report is not None:
        report = cached_report
    elif _env_truthy("NOVA_ASYNC_EVALUATION_CACHE"):
        _start_background_semantic_benchmark(k)
        report = _warming_semantic_benchmark_report(k)
    else:
        rec = await run_in_threadpool(get_rec)
        report = await run_in_threadpool(lambda: _compute_semantic_benchmark_cached(rec, k=k))
    record_usage(
        "evaluation.semantic_benchmark",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return report


@app.get("/v1/artifacts/health")
async def artifact_health_report(
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Return serving artifact availability and alignment diagnostics."""
    from backend import recommender as recommender_module

    report = evaluate_artifact_health(
        models_dir=recommender_module.MODELS_DIR,
        data_dir=recommender_module.DATA_DIR,
    )
    record_usage(
        "artifacts.health",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return report


@app.post("/v1/artifacts/reload")
async def artifact_reload(
    force_download: bool = Query(default=True),
    load: bool = Query(default=True),
    _admin_token: None = Depends(resolve_admin_token),
):
    """
    Refresh local serving artifacts and optionally rebuild the loaded recommender.

    This is for deployment automation after the daily artifact pipeline uploads a
    new manifest. It is deliberately admin-token protected and separate from
    customer-facing API keys.
    """
    from backend import recommender as recommender_module

    try:
        if load:
            rec = await run_in_threadpool(
                lambda: _reload_local_recommender(force_download=force_download)
            )
            download_results = None
            lineage = _serving_lineage(rec)
        else:
            download_results = await run_in_threadpool(
                lambda: _refresh_artifact_files(force_download=force_download)
            )
            lineage = _serving_lineage(_recommender)

        report = await run_in_threadpool(
            lambda: evaluate_artifact_health(
                models_dir=recommender_module.MODELS_DIR,
                data_dir=recommender_module.DATA_DIR,
            )
        )
        record_usage(
            "artifacts.reload",
            tenant_id="admin",
            catalog_id="serving",
            plan="internal",
            authenticated=True,
            status=str(report.get("status") or "unknown"),
        )
        return {
            "status": "reloaded" if load else "refreshed",
            "force_download": force_download,
            "loaded": load,
            "download_results": download_results,
            "artifact_health": report,
            "lineage": lineage,
        }
    except Exception as exc:
        logger.exception("Artifact reload failed")
        record_usage(
            "artifacts.reload",
            tenant_id="admin",
            catalog_id="serving",
            plan="internal",
            authenticated=True,
            status="error",
        )
        raise HTTPException(status_code=503, detail=f"Artifact reload failed: {exc}") from exc


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
async def get_all_titles(
    limit: int = Query(default=5000, ge=1, le=20000, description="Maximum number of titles to return"),
):
    """
    Get a lightweight list of all movie titles and IDs.
    Perfect for populating the Streamlit autocomplete dropdown.
    """
    remote_payload = await remote_payload_or_raise("/movies/titles", params={"limit": limit})
    if remote_payload is not None:
        return remote_payload

    rec = get_rec()
    return rec.get_all_titles(limit=limit)


@app.get("/v1/semantic-twins/id/{movie_id}")
async def semantic_twin_by_id(
    movie_id: int,
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Return the deterministic semantic item twin used for explainable scoring."""
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


@app.get("/v1/events/recommendation-analytics")
async def recommendation_event_analytics(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum rows per analytics section"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Return request/impression analytics from the recommendation event ledger."""
    record_usage(
        "events.recommendation_analytics",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return summarize_recommendation_events(limit=limit)


@app.get("/v1/recommendations/id/{movie_id}", response_model=RecommendationResponse)
@app.get("/recommend/id/{movie_id}", response_model=RecommendationResponse)
async def recommend_by_id(
    movie_id: int,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    request_id: Optional[str] = Query(default=None, description="Optional client-generated request id"),
    user_id: Optional[str] = Query(default=None, description="Optional user id for analytics attribution"),
    session_id: Optional[str] = Query(default=None, description="Optional session id for analytics attribution"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Get recommendations for a movie by TMDB ID."""
    remote_payload = await remote_payload_or_raise(
        f"/v1/recommendations/id/{movie_id}",
        params={"n": n, "request_id": request_id, "user_id": user_id, "session_id": session_id},
        context=context,
    )
    if remote_payload is not None:
        if isinstance(remote_payload, dict):
            request_id = record_recommendation_events(
                endpoint="recommendations.id.remote",
                context=context,
                query_movie=remote_payload.get("query_movie") or {"id": movie_id},
                recommendations=list(remote_payload.get("recommendations") or []),
                rec=None,
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
            )
            remote_payload.setdefault("request_id", request_id)
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
    request_id = record_recommendation_events(
        endpoint="recommendations.id",
        context=context,
        query_movie=query_movie,
        recommendations=recommendations,
        rec=rec,
        request_id=request_id,
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
        request_id=request_id,
        query_movie=query_movie,
        recommendations=recommendations,
    )


@app.get("/v1/recommendations/id/{movie_id}/enriched", response_model=EnrichedRecommendationResponse)
@app.get("/recommend/id/{movie_id}/enriched", response_model=EnrichedRecommendationResponse)
async def recommend_by_id_enriched(
    movie_id: int,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    request_id: Optional[str] = Query(default=None, description="Optional client-generated request id"),
    user_id: Optional[str] = Query(default=None, description="Optional user id for analytics attribution"),
    session_id: Optional[str] = Query(default=None, description="Optional session id for analytics attribution"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Get recommendations with FULL TMDB data (trailers, cast, etc) - PARALLEL FETCH."""
    remote_payload = await remote_payload_or_raise(
        f"/v1/recommendations/id/{movie_id}/enriched",
        params={"n": n, "request_id": request_id, "user_id": user_id, "session_id": session_id},
        context=context,
    )
    if remote_payload is not None:
        if isinstance(remote_payload, dict):
            request_id = record_recommendation_events(
                endpoint="recommendations.id.enriched.remote",
                context=context,
                query_movie=remote_payload.get("query_movie") or {"id": movie_id},
                recommendations=list(remote_payload.get("recommendations") or []),
                rec=None,
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
            )
            remote_payload.setdefault("request_id", request_id)
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
    request_id = record_recommendation_events(
        endpoint="recommendations.id.enriched",
        context=context,
        query_movie=query_movie,
        recommendations=recommendations,
        rec=rec,
        request_id=request_id,
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
    
    # Enrich all movies in parallel
    enriched = await asyncio.gather(*[enrich_movie(m) for m in recommendations])
    
    return EnrichedRecommendationResponse(
        request_id=request_id,
        query_movie=query_movie,
        recommendations=enriched,
    )


@app.get("/v1/recommendations/title/{title}", response_model=RecommendationResponse)
@app.get("/recommend/title/{title}", response_model=RecommendationResponse)
async def recommend_by_title(
    title: str,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    request_id: Optional[str] = Query(default=None, description="Optional client-generated request id"),
    user_id: Optional[str] = Query(default=None, description="Optional user id for analytics attribution"),
    session_id: Optional[str] = Query(default=None, description="Optional session id for analytics attribution"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Get recommendations for a movie by title."""
    remote_payload = await remote_payload_or_raise(
        f"/v1/recommendations/title/{quote(title, safe='')}",
        params={"n": n, "request_id": request_id, "user_id": user_id, "session_id": session_id},
        context=context,
    )
    if remote_payload is not None:
        if isinstance(remote_payload, dict):
            request_id = record_recommendation_events(
                endpoint="recommendations.title.remote",
                context=context,
                query_movie=remote_payload.get("query_movie") or {"title": title},
                recommendations=list(remote_payload.get("recommendations") or []),
                rec=None,
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
            )
            remote_payload.setdefault("request_id", request_id)
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
    request_id = record_recommendation_events(
        endpoint="recommendations.title",
        context=context,
        query_movie=query_movie,
        recommendations=recommendations,
        rec=rec,
        request_id=request_id,
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
        request_id=request_id,
        query_movie=query_movie,
        recommendations=recommendations,
    )


@app.get("/v1/recommendations/user/{user_id}", response_model=list[Movie])
async def recommend_for_user(
    user_id: str,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    limit: Optional[int] = Query(default=None, ge=1, le=50, description="Alias for number of recommendations"),
    top_k: Optional[int] = Query(default=None, ge=1, le=50, description="Alias for number of recommendations"),
    request_id: Optional[str] = Query(default=None, description="Optional client-generated request id"),
    session_id: Optional[str] = Query(default=None, description="Optional session id for analytics attribution"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Personalize recommendations from a user's recent implicit feedback events."""
    rec = get_rec()
    result_limit = top_k or limit or n
    profile = build_user_behavior_profile(user_id, limit=12)
    assignment = assign_experiment(subject_id=user_id)
    results = rec.recommend_for_user_profile(profile, n=result_limit)
    results = attach_experiment(results, assignment)
    record_recommendation_events(
        endpoint="recommendations.user",
        context=context,
        query_movie={
            "id": profile["seed_movie_ids"][0] if profile.get("seed_movie_ids") else None,
            "title": f"user:{user_id}",
        },
        recommendations=results,
        rec=rec,
        request_id=request_id,
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

