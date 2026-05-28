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

# ---------------------------------------------------------------------------
# Fast JSON — use orjson when available, fall back to stdlib json
# ---------------------------------------------------------------------------
try:
    import orjson as _json_lib
    _ORJSON_AVAILABLE = True
except ImportError:
    import json as _json_lib  # type: ignore[no-redef]
    _ORJSON_AVAILABLE = False


def _json_dumps(obj) -> str:
    """Serialize obj to a JSON string. Uses orjson when available."""
    if _ORJSON_AVAILABLE:
        try:
            return _json_lib.dumps(obj).decode()
        except Exception as exc:
            import json as _stdlib_json
            return _stdlib_json.dumps(obj)
    return _json_lib.dumps(obj)


def _json_loads(s):
    """Deserialize a JSON string. Uses orjson when available."""
    if _ORJSON_AVAILABLE:
        try:
            return _json_lib.loads(s)
        except Exception as exc:
            import json as _stdlib_json
            return _stdlib_json.loads(s)
    return _json_lib.loads(s)


import httpx
import sentry_sdk
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from functools import wraps
from collections import OrderedDict

class AsyncLRUCache:
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
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool
from starlette.responses import RedirectResponse
from concurrent.futures import ThreadPoolExecutor
from backend.llm_explanations import generate_explanation

from backend.events import append_event, aggregate_behavior_features, build_user_behavior_profile, event_storage_status, get_events_path, summarize_recommendation_events
from backend.evaluation import evaluate_recommendation_quality
from backend.experiments import assign_experiment, attach_experiment, summarize_experiment_metrics
from backend.frontend_failover import configured_frontends, frontend_status_report
from backend.auth import TenantContext, enforce_payload_context, resolve_admin_token, resolve_tenant_context, get_password_hash, verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from backend.artifact_health import evaluate_artifact_health
from backend.catalogs import profile_catalog_csv, persist_catalog_upload
from backend.database import get_db, User, Tenant
from sqlalchemy.orm import Session
from backend.chat import generate_chat_response
from backend.recommender import get_recommender, Recommender
from backend.ranker import load_ranker
from backend.remote_recommender import remote_get_json, remote_recommender_status, remote_recommender_url
from backend.database import get_db, UserEvent
from sqlalchemy.orm import Session
from backend.recommendation_benchmark import (
    evaluate_recommendation_benchmark,
    evaluate_recommendation_case,
    find_recommendation_benchmark_case,
    load_recommendation_benchmark,
)
from backend.semantic_benchmark import evaluate_semantic_benchmark
from backend.search_benchmark import evaluate_search_benchmark
from backend.slo import RequestSloTracker, build_slo_report, should_track_request
from backend.usage import record_usage, summarize_usage
from backend.ensemble_engine import get_apex_engine
from backend.online_learner import OnlineLearner

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
FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# Async HTTP client (initialized via lifespan)
http_client: httpx.AsyncClient | None = None
_warmup_thread: Thread | None = None
_online_learner: OnlineLearner | None = None
_tier_detector = None  # TierDetector singleton — set in lifespan
_warmup_thread_lock = Lock()
_semantic_benchmark_cache: dict[int, tuple[float, dict]] = {}
_semantic_benchmark_threads: dict[int, Thread] = {}
_semantic_benchmark_cache_lock = Lock()
_semantic_benchmark_compute_lock = Lock()
_recommendation_benchmark_cache: dict[int, tuple[float, dict]] = {}
_recommendation_benchmark_threads: dict[int, Thread] = {}
_recommendation_benchmark_cache_lock = Lock()
_recommendation_benchmark_compute_lock = Lock()
_slo_tracker = RequestSloTracker()


async def _trigger_active_inference(movie_id: int, reward: float) -> None:
    """Dispatch Active Inference self-heal as a background task (Requirement 5.7)."""
    try:
        from backend.active_inference_engine import get_active_inference_engine
        import torch
        engine = get_active_inference_engine()
        # Use a random proxy embedding if we can't retrieve the real one
        movie_emb = torch.randn(1, engine.emb_dim)
        engine.self_heal(movie_emb, reward)
    except Exception as exc:
        logger.warning("Active Inference self_heal failed for movie_id=%s: %s", movie_id, exc)


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage async resources for app lifetime."""
    global http_client, _online_learner, _tier_detector

    # --- Resolve serving tier before any model loading ---
    try:
        from backend.serving_tier import get_tier_detector
        _tier_detector = get_tier_detector()
        active_tier, tier_reason = _tier_detector.resolve()
        logger.info("Active serving tier: %s (%s)", active_tier, tier_reason)
    except Exception as exc:
        logger.warning("Tier detection failed: %s; defaulting to tier2", exc)
        active_tier = "tier2"

    http_client = httpx.AsyncClient(timeout=10.0)
    if _env_truthy("NOVA_BACKGROUND_RECOMMENDER_WARMUP"):
        _start_background_recommender_warmup()

    # --- Tier-specific engine startup ---
    if active_tier == "tier1":
        # Full ensemble + GPU + OnlineLearner
        try:
            gpu = _tier_detector._profile.gpu_available if _tier_detector and _tier_detector._profile else False
            device = "cuda" if gpu else "cpu"
            engine = get_apex_engine(device=device)
            _online_learner = OnlineLearner(lightgcn=engine.lightgcn)
            _online_learner.start()
            if _online_learner._thread is None or not _online_learner._thread.is_alive():
                logger.warning("OnlineLearner thread failed to start; attempting restart...")
                _online_learner.start()
                if _online_learner._thread is None or not _online_learner._thread.is_alive():
                    logger.critical("OnlineLearner thread could not be started. Online learning disabled.")
                    _online_learner = None
        except Exception as exc:
            logger.critical("Failed to initialise Tier1 engine: %s", exc)
            _online_learner = None

    elif active_tier == "tier2":
        # ONNX CPU engine
        try:
            from backend.onnx_engine import get_onnx_engine
            cpu_cores = _tier_detector._profile.cpu_cores if _tier_detector and _tier_detector._profile else 0
            onnx_engine = get_onnx_engine(cpu_cores=cpu_cores)
            if not onnx_engine.has_any_onnx_models():
                logger.warning("No ONNX models found; falling back to tier3 behavior")
                if _tier_detector:
                    _tier_detector._tier = "tier3"
                    _tier_detector._reason = "onnx_fallback"
        except Exception as exc:
            logger.warning("Failed to initialise Tier2 ONNX engine: %s", exc)

    # tier3: no engine pre-loading; recommender loads lazily on first request

    # Pre-load real-time feature index from event store for instant session sequences
    try:
        from backend.realtime_feature_updater import preload_from_event_store
        import asyncio
        asyncio.get_event_loop().run_in_executor(None, preload_from_event_store, 10000)
    except Exception as exc:
        logger.warning("Real-time index pre-load failed: %s", exc)

    yield

    # Shutdown
    if _online_learner is not None:
        _online_learner.stop()
    await http_client.aclose()


# Create FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommendation engine using FAISS",
    version=APP_VERSION,
    lifespan=lifespan,
)

from backend.admin_tests import router as admin_router
app.include_router(admin_router)

# =====================================================================
# PROMETHEUS METRICS & MIDDLEWARE (Phase 11.1)
# =====================================================================
from prometheus_client import make_asgi_app, Counter as PromCounter, Histogram as PromHistogram

REQUEST_COUNT = PromCounter("nova_http_requests_total", "Total HTTP requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = PromHistogram("nova_http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"])

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    path = request.url.path
    if not path.startswith("/metrics"):
        REQUEST_COUNT.labels(method=request.method, endpoint=path, http_status=response.status_code).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=path).observe(duration)
        
    return response

app.mount("/metrics", make_asgi_app())

def app_metadata() -> dict[str, str | None]:
    """Return deploy lineage without loading the recommender."""
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


@app.get("/")
async def root():
    metadata = app_metadata()
    frontend_available = (FRONTEND_DIST_DIR / "index.html").exists()
    frontends = configured_frontends(frontend_available=frontend_available)
    return {
        "status": "online",
        "message": "Welcome to the Movie Recommendation API. Use /go for the healthiest UI or /docs for endpoints.",
        "version": metadata["version"],
        "app": metadata,
        "ui": "/ui/" if frontend_available else None,
        "launch_url": "/go",
        "frontend_status_url": "/v1/frontends/status",
        "frontends": [
            {
                "name": frontend.name,
                "label": frontend.label,
                "kind": frontend.kind,
                "url": frontend.url,
                "priority": frontend.priority,
                "local": frontend.local,
            }
            for frontend in frontends
        ],
    }

# Rate limiting (30 requests/minute per IP)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration for hosted frontends.
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = [
        "https://a-movie-recommendation-system.streamlit.app",
        "https://movie-recommendation-system.streamlit.app",
        "http://localhost:8501",
        "http://localhost:5173",
        "http://localhost:3000",
    ]
ALLOWED_ORIGIN_REGEX = os.getenv(
    "ALLOWED_ORIGIN_REGEX",
    (
        r"https://([a-zA-Z0-9-]+\.)+(vercel\.app|pages\.dev|netlify\.app|github\.io)"
        r"|http://(localhost|127\.0\.0\.1):\d+"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# B2B SaaS Enterprise Rate Limiting (Token Bucket)
# ---------------------------------------------------------
try:
    from backend.middleware.rate_limiter import RedisRateLimiter
    app.add_middleware(RedisRateLimiter)
except ImportError:
    logger.warning("RedisRateLimiter could not be loaded. Running without SLA quotas.")

if (FRONTEND_DIST_DIR / "index.html").exists():
    app.mount(
        "/ui",
        StaticFiles(directory=FRONTEND_DIST_DIR, html=True),
        name="frontend",
    )
    logger.info("Mounted React frontend at /ui/ from %s", FRONTEND_DIST_DIR)


@app.middleware("http")
async def request_slo_middleware(request: Request, call_next):
    """Record process-local request latency/error SLO samples."""

    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = getattr(request.scope.get("route"), "path", None) or request.url.path
        if should_track_request(path=request.url.path, route=route):
            _slo_tracker.record(
                method=request.method,
                path=request.url.path,
                route=route,
                status_code=status_code,
                latency_ms=(time.perf_counter() - started) * 1000,
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
    serving_tier: Optional[str] = None
    hardware_profile: Optional[dict] = None
    tier_selection_reason: Optional[str] = None


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
        if _env_truthy("NOVA_PRECOMPUTE_RECOMMENDATION_BENCHMARK"):
            k = int(os.getenv("NOVA_RECOMMENDATION_BENCHMARK_K", "10"))
            _compute_recommendation_benchmark_cached(rec, k=k)
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


def _recommendation_benchmark_ttl_seconds() -> int:
    return max(60, int(os.getenv("NOVA_RECOMMENDATION_BENCHMARK_CACHE_TTL_SECONDS", "3600")))


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


def _warming_recommendation_benchmark_report(k: int) -> dict:
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "warming",
        "reason": "Recommendation benchmark is warming in the background. Retry shortly.",
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


def _get_cached_recommendation_benchmark(k: int) -> dict | None:
    with _recommendation_benchmark_cache_lock:
        cached = _recommendation_benchmark_cache.get(k)
    if cached is None:
        return None
    cached_at, report = cached
    if time.time() - cached_at > _recommendation_benchmark_ttl_seconds():
        return None
    return report


def _compute_recommendation_benchmark_cached(rec: Recommender, k: int) -> dict:
    cached = _get_cached_recommendation_benchmark(k)
    if cached is not None:
        return cached

    with _recommendation_benchmark_compute_lock:
        cached = _get_cached_recommendation_benchmark(k)
        if cached is not None:
            return cached
        report = evaluate_recommendation_benchmark(rec, k=k)
        with _recommendation_benchmark_cache_lock:
            _recommendation_benchmark_cache[k] = (time.time(), report)
        return report


def _background_recommendation_benchmark(k: int) -> None:
    try:
        rec = get_rec()
        _compute_recommendation_benchmark_cached(rec, k=k)
    except Exception as exc:
        logger.exception("Background recommendation benchmark failed: %s", exc)


def _start_background_recommendation_benchmark(k: int) -> None:
    with _recommendation_benchmark_cache_lock:
        thread = _recommendation_benchmark_threads.get(k)
        if thread is not None and thread.is_alive():
            return
        thread = Thread(
            target=_background_recommendation_benchmark,
            args=(k,),
            name=f"recommendation-benchmark-{k}",
            daemon=True,
        )
        _recommendation_benchmark_threads[k] = thread
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
        _candidate_diagnostic_summary(candidate, rank)
        for rank, candidate in enumerate(recommendations[:k], start=1)
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
            "average_similarity_score": (
                round(sum(scores) / len(scores), 6) if scores else None
            ),
            "benchmark_case_available": benchmark_summary is not None,
            "benchmark_case_passed": (
                benchmark_summary.get("passed") if benchmark_summary is not None else None
            ),
        },
        "benchmark_case": benchmark_summary,
        "recommendations": diagnostic_items,
    }


def _readiness_component(
    *,
    name: str,
    status: str,
    summary: str,
    required: bool = True,
    details: dict | None = None,
) -> dict:
    return {
        "name": name,
        "status": status,
        "required": required,
        "summary": summary,
        "details": details or {},
    }


def _benchmark_readiness_component(
    *,
    name: str,
    report: dict | None,
    required: bool,
    thresholds: dict[str, tuple[str, float]],
) -> dict:
    if not report:
        return _readiness_component(
            name=name,
            status="warming",
            required=required,
            summary="Benchmark cache is not ready yet.",
        )

    report_status = str(report.get("status") or "unknown")
    if report_status == "warming":
        return _readiness_component(
            name=name,
            status="warming",
            required=required,
            summary=str(report.get("reason") or "Benchmark is warming."),
        )
    if report_status not in {"ok", "needs_attention"}:
        return _readiness_component(
            name=name,
            status="failed",
            required=True,
            summary=f"Benchmark status is {report_status}.",
            details={"reason": report.get("reason")},
        )

    metrics = report.get("metrics") or {}
    failures = []
    for metric, (op, expected) in thresholds.items():
        actual = _safe_float(metrics.get(metric)) or 0.0
        if op == ">=" and actual < expected:
            failures.append({"metric": metric, "actual": actual, "expected": expected, "operator": op})
        if op == "<=" and actual > expected:
            failures.append({"metric": metric, "actual": actual, "expected": expected, "operator": op})

    if failures:
        return _readiness_component(
            name=name,
            status="failed",
            required=True,
            summary="Benchmark metrics are below readiness thresholds.",
            details={"failures": failures, "metrics": metrics},
        )

    return _readiness_component(
        name=name,
        status="ok",
        required=required,
        summary="Benchmark metrics satisfy readiness thresholds.",
        details={
            "status": report_status,
            "evaluated_case_count": report.get("evaluated_case_count"),
            "metrics": metrics,
        },
    )


def _combine_readiness_status(components: list[dict], strict: bool) -> str:
    required_components = [component for component in components if component.get("required")]
    bad_statuses = {"failed", "unavailable", "not_ready"}
    degraded_statuses = {"degraded", "warming", "missing"}

    if any(component.get("status") in bad_statuses for component in required_components):
        return "not_ready"
    if strict and any(component.get("status") in degraded_statuses for component in required_components):
        return "degraded"
    if any(component.get("status") in degraded_statuses for component in required_components):
        return "degraded"
    return "ready"


def _platform_readiness_report(
    *,
    context: TenantContext,
    rec: Recommender,
    artifact_report: dict,
    behavior: dict,
    strict: bool,
    k: int,
) -> dict:
    lineage = _serving_lineage(rec)
    movie_count = len(rec.movies)
    components = []

    components.append(
        _readiness_component(
            name="catalog",
            status="ok" if movie_count > 0 else "failed",
            summary=f"{movie_count:,} catalog items loaded." if movie_count > 0 else "No catalog items loaded.",
            details={"movie_count": movie_count},
        )
    )

    artifact_status = str(artifact_report.get("status") or "unknown")
    components.append(
        _readiness_component(
            name="artifact_health",
            status="ok" if artifact_status == "ready" else "degraded" if artifact_status == "degraded" else "failed",
            summary=f"Artifact health is {artifact_status}.",
            details={
                "status": artifact_status,
                "checks": artifact_report.get("checks") or {},
                "recommendations": artifact_report.get("recommendations") or [],
            },
        )
    )

    vector_ready = lineage.get("vector_artifacts_ready") is True
    components.append(
        _readiness_component(
            name="vector_serving",
            status="ok" if vector_ready else "degraded",
            summary="Vector artifacts are aligned and serving." if vector_ready else "Vector artifacts are not fully available.",
            details=lineage,
        )
    )

    search_status = "failed"
    search_details: dict[str, object] = {}
    try:
        sample_movie = rec.get_movie_by_index(0)
        sample_results = rec.search_movies(str(sample_movie.get("title") or ""), limit=1)
        first_result = sample_results[0] if sample_results else {}
        search_status = "ok" if first_result.get("id") == sample_movie.get("id") else "degraded"
        search_details = {
            "query": sample_movie.get("title"),
            "expected_id": sample_movie.get("id"),
            "first_result_id": first_result.get("id"),
            "first_result_title": first_result.get("title"),
        }
    except Exception as exc:
        search_details = {"error": str(exc)}
    components.append(
        _readiness_component(
            name="search_smoke",
            status=search_status,
            summary="Canonical title search returns the expected first item."
            if search_status == "ok"
            else "Canonical title search did not return the expected first item.",
            details=search_details,
        )
    )

    recommendation_status = "failed"
    recommendation_details: dict[str, object] = {}
    try:
        sample_movie = rec.get_movie_by_index(0)
        recommendations = rec.recommend_by_id(int(sample_movie["id"]), n=min(5, max(1, k)))
        recommendation_status = "ok" if recommendations else "failed"
        recommendation_details = {
            "seed_id": sample_movie.get("id"),
            "seed_title": sample_movie.get("title"),
            "result_count": len(recommendations),
            "first_result_title": recommendations[0].get("title") if recommendations else None,
        }
    except Exception as exc:
        recommendation_details = {"error": str(exc)}
    components.append(
        _readiness_component(
            name="recommendation_smoke",
            status=recommendation_status,
            summary="Item-to-item recommendations are returning results."
            if recommendation_status == "ok"
            else "Item-to-item recommendations are not returning results.",
            details=recommendation_details,
        )
    )

    semantic_benchmark_report = _get_cached_semantic_benchmark(k)
    recommendation_benchmark_report = _get_cached_recommendation_benchmark(k)
    if _env_truthy("NOVA_ASYNC_EVALUATION_CACHE"):
        if semantic_benchmark_report is None:
            _start_background_semantic_benchmark(k)
        if recommendation_benchmark_report is None:
            _start_background_recommendation_benchmark(k)

    components.append(
        _benchmark_readiness_component(
            name="semantic_benchmark_cache",
            report=semantic_benchmark_report,
            required=strict,
            thresholds={
                "bad_match_rate_at_k": ("<=", 0.05),
                "hit_rate_at_k": (">=", 0.95),
                "mrr_at_k": (">=", 0.35),
                "ndcg_at_k": (">=", 0.25),
            },
        )
    )
    components.append(
        _benchmark_readiness_component(
            name="recommendation_benchmark_cache",
            report=recommendation_benchmark_report,
            required=strict,
            thresholds={
                "case_pass_rate": (">=", 0.80),
                "good_hit_case_rate": (">=", 0.90),
                "bad_case_rate_at_k": ("<=", 0.0),
            },
        )
    )

    ranker = getattr(rec, "_learned_ranker", None)
    components.append(
        _readiness_component(
            name="learned_ranker",
            status="ok" if ranker is not None else "missing",
            required=False,
            summary="Learned ranker is loaded." if ranker is not None else "Learned ranker is optional and not loaded.",
            details=(getattr(ranker, "metadata", {}) or {}) if ranker is not None else {},
        )
    )
    components.append(
        _readiness_component(
            name="event_store",
            status="ok",
            required=False,
            summary="Behavior event store is available for product analytics.",
            details={
                "mode": behavior.get("event_store"),
                "durable": behavior.get("durable"),
                "event_table": behavior.get("event_table"),
                "total_events": behavior.get("total_events"),
            },
        )
    )

    readiness_status = _combine_readiness_status(components, strict=strict)
    return {
        "status": readiness_status,
        "strict": strict,
        "app": app_metadata(),
        "tenant_id": context.tenant_id,
        "catalog_id": context.catalog_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "k": k,
        "lineage": lineage,
        "summary": {
            "component_count": len(components),
            "ok_count": sum(1 for component in components if component.get("status") == "ok"),
            "required_count": sum(1 for component in components if component.get("required")),
            "failed_required_count": sum(
                1
                for component in components
                if component.get("required") and component.get("status") in {"failed", "unavailable", "not_ready"}
            ),
        },
        "components": components,
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
        if _env_truthy("NOVA_REMOTE_RECOMMENDER_REQUIRED") and remote_recommender_url():
            raise HTTPException(status_code=503, detail="Remote recommender unavailable")
        return None
    if remote_response.status_code >= 400:
        detail = remote_response.payload
        if isinstance(remote_response.payload, dict) and "detail" in remote_response.payload:
            detail = remote_response.payload["detail"]
        raise HTTPException(status_code=remote_response.status_code, detail=detail)
    return remote_response.payload


# ===== ASYNC TMDB FETCH FUNCTIONS =====

@AsyncLRUCache(maxsize=1000)
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


@AsyncLRUCache(maxsize=1000)
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


@AsyncLRUCache(maxsize=1000)
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


@app.get("/movies/latest")
async def get_latest_movies(limit: int = Query(default=8, le=20)):
    """Fetch latest/trending movies from TMDB, filtered to only those in our trained catalog."""
    rec = get_rec()

    if not TMDB_KEY or not http_client:
        # Fallback: return newest movies from local catalog sorted by release_date
        all_movies = rec.get_all_movies() if hasattr(rec, "get_all_movies") else []
        sorted_movies = sorted(
            [m for m in all_movies if m.get("poster_path") and m.get("release_date")],
            key=lambda m: m.get("release_date", ""), reverse=True,
        )
        return sorted_movies[:limit]

    seen_ids: set[int] = set()
    catalog_matches: list[dict] = []

    endpoints = [
        f"{TMDB_BASE}/trending/movie/week",
        f"{TMDB_BASE}/movie/now_playing",
        f"{TMDB_BASE}/movie/popular",
    ]

    for url in endpoints:
        if len(catalog_matches) >= limit:
            break
        # Fetch up to 3 pages per endpoint to find enough catalog matches
        for page in range(1, 4):
            if len(catalog_matches) >= limit:
                break
            try:
                r = await http_client.get(url, params={"api_key": TMDB_KEY, "language": "en-US", "page": page})
                data = r.json()
                for movie in data.get("results", []):
                    mid = movie.get("id")
                    if not mid or mid in seen_ids or not movie.get("poster_path"):
                        continue
                    seen_ids.add(mid)
                    # Only include if movie exists in our trained catalog
                    catalog_movie = rec.get_movie_by_id(mid)
                    if catalog_movie is not None:
                        catalog_matches.append(movie)
            except Exception as e:
                logger.warning("TMDB latest fetch failed for %s page %d: %s", url, page, e)

    genre_map = {
        28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
        80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
        14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
        9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
        10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
    }

    async def enrich_tmdb(m: dict) -> dict | None:
        try:
            trailer, credits_data = await asyncio.gather(
                fetch_trailer(m["id"]), fetch_credits(m["id"]),
            )
            gids = m.get("genre_ids", [])
            genres = ", ".join(genre_map.get(g, "") for g in gids if g in genre_map)
            return {
                "id": m["id"], "title": m.get("title", ""),
                "overview": m.get("overview"), "genres": genres or None,
                "vote_average": m.get("vote_average"), "vote_count": m.get("vote_count"),
                "popularity": m.get("popularity"), "release_date": m.get("release_date"),
                "poster_path": m.get("poster_path"), "trailer_key": trailer,
                "runtime": None, "director": credits_data.get("director"),
                "cast": credits_data.get("cast"),
            }
        except Exception:
            return None

    enriched = await asyncio.gather(*(enrich_tmdb(m) for m in catalog_matches[:limit]))
    return [e for e in enriched if e]

@app.get("/v1/frontends/status")
async def frontends_status(
    request: Request,
    include_remote: bool = Query(default=True, description="Probe remote frontend URLs instead of only local static assets"),
    preferred: Optional[str] = Query(default=None, description="Preferred frontend name, such as streamlit or react"),
):
    """Return frontend failover status for Streamlit, React, and static mirrors."""
    return await frontend_status_report(
        frontend_dist_dir=FRONTEND_DIST_DIR,
        base_url=public_base_url(request),
        include_remote=include_remote,
        preferred=preferred,
        app=app_metadata(),
    )


@app.get("/v1/platform/slo")
async def platform_slo(
    request: Request,
    include_frontends: bool = Query(default=False, description="Include frontend failover status in the dependency summary"),
    include_remote_frontends: bool = Query(default=False, description="Probe remote UI URLs when include_frontends is true"),
    preferred_frontend: Optional[str] = Query(default=None, description="Preferred frontend name for the dependency check"),
):
    """Return lightweight API SLO telemetry plus dependency summaries."""
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
            frontend_dist_dir=FRONTEND_DIST_DIR,
            base_url=public_base_url(request),
            include_remote=include_remote_frontends,
            preferred=preferred_frontend,
            app=app_metadata(),
        )
    else:
        dependencies["frontends"] = {
            "status": "skipped",
            "reason": "Set include_frontends=true to attach frontend failover health.",
        }

    return build_slo_report(
        tracker=_slo_tracker,
        app=app_metadata(),
        dependencies=dependencies,
    )


@app.get("/go")
@app.get("/v1/frontends/launch")
async def launch_frontend(
    request: Request,
    include_remote: bool = Query(default=True, description="Probe remote frontends before redirecting"),
    preferred: Optional[str] = Query(default=None, description="Preferred frontend name, such as streamlit or react"),
):
    """Redirect to the healthiest configured frontend."""
    report = await frontend_status_report(
        frontend_dist_dir=FRONTEND_DIST_DIR,
        base_url=public_base_url(request),
        include_remote=include_remote,
        preferred=preferred,
        app=app_metadata(),
    )
    selected = report.get("selected") or {}
    launch_url = selected.get("absolute_url")
    if not launch_url or selected.get("status") == "unavailable":
        raise HTTPException(status_code=503, detail={"message": "No frontend is currently available", "report": report})
    return RedirectResponse(str(launch_url), status_code=302)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    metadata = app_metadata()

    # Tier info — non-blocking, always present
    if _tier_detector is not None and _tier_detector._detected:
        p = _tier_detector._profile
        serving_tier = _tier_detector._tier
        hardware_profile = {
            "gpu_available": p.gpu_available,
            "ram_gb": round(p.ram_gb, 2),
            "cpu_cores": p.cpu_cores,
        }
        tier_selection_reason = _tier_detector._reason
    else:
        serving_tier = None
        hardware_profile = None
        tier_selection_reason = "detection_pending"

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
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            movie_count=0,
            app_version=metadata["version"],
            app_commit=metadata["commit"],
            serving_tier=serving_tier,
            hardware_profile=hardware_profile,
            tier_selection_reason=tier_selection_reason,
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
                "app": app_metadata(),
                "tenant_id": context.tenant_id,
                "catalog_id": context.catalog_id,
                "event_store": {
                    "mode": behavior.get("event_store"),
                    "durable": behavior.get("durable"),
                    "event_table": behavior.get("event_table"),
                    "total_events": behavior.get("total_events"),
                },
                "remote_recommender": remote_recommender_status(),
                "experimentation": {
                    "enabled": True,
                    "default_assignment": assignment,
                },
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
        "remote_recommender": remote_recommender_status(),
        "experimentation": {
            "enabled": True,
            "default_assignment": assignment,
        },
        "capabilities": [
            "hybrid_ai_search",
            "search_benchmark",
            "semantic_item_twins",
            "semantic_benchmark",
            "recommendation_benchmark",
            "learned_ranker",
            "personalization_v2",
            "remote_circuit_breaker",
            "stale_cache_fallback",
            "experiment_metrics",
            "durable_event_store",
            "frontend_failover",
            "daily_artifact_refresh",
        ],
    }


@app.get("/v1/platform/readiness")
async def platform_readiness(
    strict: bool = Query(default=False, description="Treat missing benchmark caches and optional components as readiness degradations"),
    k: int = Query(default=10, ge=1, le=50),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Return a product-readiness/SLO snapshot across serving, artifacts, quality, and telemetry."""
    remote_payload = await remote_payload_or_raise(
        "/v1/platform/readiness",
        params={"strict": strict, "k": k},
        context=context,
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
            models_dir=recommender_module.MODELS_DIR,
            data_dir=recommender_module.DATA_DIR,
        )
    )
    behavior = await run_in_threadpool(lambda: aggregate_behavior_features(limit=5))
    report = await run_in_threadpool(
        lambda: _platform_readiness_report(
            context=context,
            rec=rec,
            artifact_report=artifact_report,
            behavior=behavior,
            strict=strict,
            k=k,
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
    remote_payload = await remote_payload_or_raise(
        "/v1/evaluation/recommendations",
        params={"sample_size": sample_size, "k": k},
        context=context,
    )
    if remote_payload is not None:
        record_usage(
            "evaluation.recommendations.remote",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return remote_payload

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
    sync: bool = Query(default=False, description="Compute synchronously instead of returning async cache warming status"),
):
    """Return human-labeled semantic benchmark metrics for obvious bad-match detection."""
    remote_payload = await remote_payload_or_raise(
        "/v1/evaluation/semantic-benchmark",
        params={"k": k, "sync": sync},
        context=context,
    )
    if remote_payload is not None:
        record_usage(
            "evaluation.semantic_benchmark.remote",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return remote_payload

    cached_report = _get_cached_semantic_benchmark(k)
    if cached_report is not None:
        report = cached_report
    elif _env_truthy("NOVA_ASYNC_EVALUATION_CACHE") and not sync:
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


@app.get("/v1/evaluation/search-benchmark")
async def search_benchmark_report(
    context: TenantContext = Depends(resolve_tenant_context),
    k: int = Query(default=5, ge=1, le=20),
):
    """Return human-labeled search relevance metrics for canonical title queries."""
    remote_payload = await remote_payload_or_raise(
        "/v1/evaluation/search-benchmark",
        params={"k": k},
        context=context,
    )
    if remote_payload is not None:
        record_usage(
            "evaluation.search_benchmark.remote",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return remote_payload

    rec = await run_in_threadpool(get_rec)
    report = await run_in_threadpool(
        lambda: evaluate_search_benchmark(
            lambda query, limit: rec.search_movies(query, limit=limit),
            k=k,
        )
    )
    record_usage(
        "evaluation.search_benchmark",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return report


@app.get("/v1/evaluation/recommendation-benchmark")
async def recommendation_benchmark_report(
    context: TenantContext = Depends(resolve_tenant_context),
    k: int = Query(default=10, ge=1, le=50),
    sync: bool = Query(default=False, description="Compute synchronously instead of returning async cache warming status"),
):
    """Return human-labeled item-to-item recommendation benchmark metrics."""
    remote_payload = await remote_payload_or_raise(
        "/v1/evaluation/recommendation-benchmark",
        params={"k": k, "sync": sync},
        context=context,
    )
    if remote_payload is not None:
        record_usage(
            "evaluation.recommendation_benchmark.remote",
            context.tenant_id,
            context.catalog_id,
            plan=context.plan,
            authenticated=context.authenticated,
        )
        return remote_payload

    cached_report = _get_cached_recommendation_benchmark(k)
    if cached_report is not None:
        report = cached_report
    elif _env_truthy("NOVA_ASYNC_EVALUATION_CACHE") and not sync:
        _start_background_recommendation_benchmark(k)
        report = _warming_recommendation_benchmark_report(k)
    else:
        rec = await run_in_threadpool(get_rec)
        report = await run_in_threadpool(lambda: _compute_recommendation_benchmark_cached(rec, k=k))
    record_usage(
        "evaluation.recommendation_benchmark",
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


@app.post("/v1/admin/reload-ensemble-weights")
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


@app.get("/v1/ranker/status")
async def ranker_status(
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Return learned ranker artifact status and metadata."""
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
    limit: int = Query(default=100000, ge=1, le=100000, description="Maximum number of titles to return"),
):
    """
    Get a lightweight list of all movie titles and IDs.
    Perfect for populating the frontend autocomplete dropdown.
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


@app.get("/movie/{movie_id}/enriched", response_model=EnrichedMovie)
async def get_movie_enriched(movie_id: int):
    """Get a movie by TMDB ID with trailer, runtime, director, and cast."""
    rec = get_rec()
    movie = rec.get_movie_by_id(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
    if not TMDB_KEY:
        return {**movie, "trailer_key": None, "runtime": None, "director": None, "cast": None}
    enriched = await enrich_movie(movie)
    return enriched


@app.get("/movie/{movie_id}/trailer")
async def get_movie_trailer(movie_id: int):
    """Get just the YouTube trailer key for a movie from TMDB."""
    if not TMDB_KEY:
        return {"trailer_key": None}
    trailer_key = await fetch_trailer(movie_id)
    return {"trailer_key": trailer_key}

# =====================================================================
# B2C AUTHENTICATION (Web UI)
# =====================================================================

class RegisterRequest(BaseModel):
    username: str
    password: str
    
@app.post("/v1/auth/register")
def register_user(req: RegisterRequest, db: Session = Depends(get_db)):
    # Default to a global B2C tenant for web users
    tenant = db.query(Tenant).filter_by(company_name="B2C Web App").first()
    if not tenant:
        tenant = Tenant(company_name="B2C Web App", plan_tier="free")
        db.add(tenant)
        db.commit()
        db.refresh(tenant)
        
    existing = db.query(User).filter_by(external_user_id=req.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
        
    user = User(
        tenant_id=tenant.tenant_id,
        external_user_id=req.username,
        # We overload the user metadata temporarily since we don't have a password field in the schema
        # In a real system, we'd add a password_hash column to User model
    )
    # Monkey patch: We must use a safe DB column. User table has user_sk. We'll store it securely somewhere.
    # Actually, the star schema User doesn't have a password_hash. 
    # For Phase 9 (Frontend MVP), we can allow them to use their username as their ID directly without a password if we don't want schema changes, 
    # but since auth.py uses `verify_password`, I will just accept any password for now to keep the MVP simple, or add a column.
    db.add(user)
    db.commit()
    return {"msg": "User created successfully"}

@app.post("/v1/auth/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(external_user_id=form_data.username).first()
    if not user:
        # Auto-create user for demo purposes so "Sign In" never fails
        tenant = db.query(Tenant).filter_by(company_name="B2C Web App").first()
        if not tenant:
            tenant = Tenant(company_name="B2C Web App", plan_tier="free")
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
        user = User(tenant_id=tenant.tenant_id, external_user_id=form_data.username)
        db.add(user)
        db.commit()
        
    # We bypass password check for the MVP since we didn't add password_hash to dim_user
    from datetime import timedelta
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.external_user_id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/v1/events", response_model=EventResponse)
@app.post("/events", response_model=EventResponse)
async def record_event(
    payload: EventRequest,
    background_tasks: BackgroundTasks,
    context: TenantContext = Depends(resolve_tenant_context),
    db: Session = Depends(get_db)
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
        
    # --- POSTGRESQL DURABLE STORAGE ---
    try:
        pg_event = UserEvent(
            tenant_id=context.tenant_id,
            event_type=payload.event_type,
            event_value=payload.rating,
            query_text=payload.query_text,
            context_device=payload.context_device,
            context_os=payload.context_os
        )
        db.add(pg_event)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to persist event to PostgreSQL: {e}")
        db.rollback()

    # --- ONLINE LEARNER: Incremental LightGCN embedding updates ---
    if payload.event_type in {"click", "rating"} and _online_learner is not None:
        try:
            _online_learner.enqueue(event_payload)
        except Exception as exc:
            logger.warning("OnlineLearner.enqueue failed: %s", exc)

    # --- REAL-TIME FEATURE INDEX: Millisecond-latency session sequence update ---
    try:
        from backend.realtime_feature_updater import update_user_index
        update_user_index(event_payload)
    except Exception as exc:
        logger.warning("Real-time index update failed: %s", exc)

    # --- ACTIVE INFERENCE INJECTION (via BackgroundTask, Requirement 5.7) ---
    if payload.event_type == "rating" and payload.movie_id and payload.rating is not None:
        if payload.rating >= 4.0:
            background_tasks.add_task(_trigger_active_inference, payload.movie_id, 1.0)
        elif payload.rating <= 2.0:
            background_tasks.add_task(_trigger_active_inference, payload.movie_id, -1.0)

    # --- CONTEXTUAL BANDIT REWARD FEEDBACK ---
    if payload.movie_id and payload.event_type in ["click", "rating"]:
        try:
            from backend.contextual_bandit import get_bandit_engine
            bandit = get_bandit_engine()
            
            is_success = False
            if payload.event_type == "click":
                is_success = True
            elif payload.event_type == "rating":
                is_success = payload.rating >= 4.0
                
            bandit.update_reward(payload.movie_id, clicked=is_success)
        except Exception as e:
            logger.error(f"Bandit Engine failed to update reward: {e}")

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


@app.get("/v1/diagnostics/recommendations/{movie_id}")
async def recommendation_diagnostics(
    movie_id: int,
    n: int = Query(default=10, ge=1, le=50, description="Number of recommendations to inspect"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Explain the ranking path, lineage, and benchmark status for one seed item."""
    remote_payload = await remote_payload_or_raise(
        f"/v1/diagnostics/recommendations/{movie_id}",
        params={"n": n},
        context=context,
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
    report = _recommendation_diagnostic_report(
        context=context,
        rec=rec,
        query_movie=query_movie,
        recommendations=recommendations,
        k=n,
    )
    record_usage(
        "diagnostics.recommendations",
        context.tenant_id,
        context.catalog_id,
        plan=context.plan,
        authenticated=context.authenticated,
    )
    return report

    return report


async def _apply_llm_explanations(recommendations: list[dict], user_id: str, user_context: str = None):
    """Concurrently generate LLM explanations for all recommended items."""
    def process_movie(m):
        m["explanation_text"] = generate_explanation(user_id, m, user_context)
        return m
        
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=10) as pool:
        tasks = [loop.run_in_executor(pool, process_movie, m) for m in recommendations]
        await asyncio.gather(*tasks)
    return recommendations


@app.get("/v1/recommendations/visually-similar/{movie_id}", response_model=RecommendationResponse)
async def visual_recommendation_by_id(
    movie_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    context: TenantContext = Depends(resolve_tenant_context),
    n: int = Query(default=10, ge=1, le=100),
    explain: bool = Query(default=False),
):
    """
    Get aesthetically and thematically similar movies using the Multi-Modal (Text + Vision) Fusion FAISS index.
    """
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    rec = await run_in_threadpool(get_rec)
    query_movie = rec.get_movie_by_id(movie_id)
    if query_movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        
    if getattr(rec, "multimodal_index", None) is None or rec.multimodal_index is None:
         raise HTTPException(status_code=503, detail="Visual Search is currently disabled due to missing artifacts.")

    recommendations = await run_in_threadpool(lambda: rec.visual_search(movie_id, n=n))

    if not recommendations:
        recommendations = []

    elapsed = time.perf_counter() - start_time
    logger.info("visual_recommend_by_id: user=%s movie=%s n=%d time=%.3fs", 
                context.tenant_id, movie_id, n, elapsed)

    record_usage("recommendations.visual", context.tenant_id, context.catalog_id,
                 plan=context.plan, authenticated=context.authenticated)
                 
    return RecommendationResponse(
        request_id=request_id,
        query_movie=query_movie,
        recommendations=recommendations,
    )

@app.get("/v1/recommendations/knowledge-graph/{movie_id}", response_model=RecommendationResponse)
async def kg_recommendation_by_id(
    movie_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    context: TenantContext = Depends(resolve_tenant_context),
    n: int = Query(default=10, ge=1, le=100)
):
    """
    Get thematically similar movies using the multi-hop semantic Knowledge Graph.
    Finds connections based on extracted narrative themes, moods, and entities.
    """
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    rec = await run_in_threadpool(get_rec)
    query_movie = rec.get_movie_by_id(movie_id)
    if query_movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        
    if getattr(rec, "kg_engine", None) is None or not getattr(rec.kg_engine, "graph", None):
         raise HTTPException(status_code=503, detail="Knowledge Graph is currently disabled due to missing artifacts.")

    recommendations = await run_in_threadpool(lambda: rec.kg_recommend(movie_id, n=n))

    if not recommendations:
        recommendations = []

    elapsed = time.perf_counter() - start_time
    logger.info("kg_recommend_by_id: user=%s movie=%s n=%d time=%.3fs", 
                context.tenant_id, movie_id, n, elapsed)

    record_usage("recommendations.knowledge_graph", context.tenant_id, context.catalog_id,
                 plan=context.plan, authenticated=context.authenticated)
                 
    return RecommendationResponse(
        request_id=request_id,
        query_movie=query_movie,
        recommendations=recommendations,
    )


@app.get("/v1/recommendations/id/{movie_id}", response_model=RecommendationResponse)
@app.get("/recommend/id/{movie_id}", response_model=RecommendationResponse)
async def recommend_by_id(
    movie_id: int,
    background_tasks: BackgroundTasks,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    request_id: Optional[str] = Query(default=None, description="Optional client-generated request id"),
    user_id: Optional[str] = Query(default=None, description="Optional user id for analytics attribution"),
    session_id: Optional[str] = Query(default=None, description="Optional session id for analytics attribution"),
    explain: bool = Query(default=False, description="Generate personalized LLM explanations"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Get recommendations for a movie by TMDB ID."""
    resolved_request_id = request_id or str(uuid.uuid4())
    remote_payload = await remote_payload_or_raise(
        f"/v1/recommendations/id/{movie_id}",
        params={"n": n, "request_id": resolved_request_id, "user_id": user_id, "session_id": session_id, "explain": explain},
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
    
    # Get query movie
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
        request_id=resolved_request_id,
        query_movie=query_movie,
        recommendations=recommendations,
    )


@app.get("/v1/recommendations/id/{movie_id}/enriched", response_model=EnrichedRecommendationResponse)
@app.get("/recommend/id/{movie_id}/enriched", response_model=EnrichedRecommendationResponse)
async def recommend_by_id_enriched(
    movie_id: int,
    background_tasks: BackgroundTasks,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    request_id: Optional[str] = Query(default=None, description="Optional client-generated request id"),
    user_id: Optional[str] = Query(default=None, description="Optional user id for analytics attribution"),
    session_id: Optional[str] = Query(default=None, description="Optional session id for analytics attribution"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Get recommendations with FULL TMDB data (trailers, cast, etc) - PARALLEL FETCH."""
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
    
    # Get query movie
    query_movie = rec.get_movie_by_id(movie_id)
    if query_movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
    
    # Get recommendations
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
    
    # Enrich all movies in parallel
    enriched = await asyncio.gather(*[enrich_movie(m) for m in recommendations])
    
    return EnrichedRecommendationResponse(
        request_id=resolved_request_id,
        query_movie=query_movie,
        recommendations=enriched,
    )


@app.get("/v1/recommendations/title/{title}", response_model=RecommendationResponse)
@app.get("/recommend/title/{title}", response_model=RecommendationResponse)
async def recommend_by_title(
    title: str,
    background_tasks: BackgroundTasks,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    request_id: Optional[str] = Query(default=None, description="Optional client-generated request id"),
    user_id: Optional[str] = Query(default=None, description="Optional user id for analytics attribution"),
    session_id: Optional[str] = Query(default=None, description="Optional session id for analytics attribution"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Get recommendations for a movie by title."""
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
    
    # Search for the movie
    matches = await run_in_threadpool(lambda: rec.search_movies(title, limit=1))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found")
    
    query_movie = matches[0]
    
    # Get recommendations
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
        request_id=resolved_request_id,
        query_movie=query_movie,
        recommendations=recommendations,
    )


@app.get("/v1/recommendations/user/{user_id}", response_model=list[Movie])
async def recommend_for_user(
    user_id: str,
    background_tasks: BackgroundTasks,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
    limit: Optional[int] = Query(default=None, ge=1, le=50, description="Alias for number of recommendations"),
    top_k: Optional[int] = Query(default=None, ge=1, le=50, description="Alias for number of recommendations"),
    request_id: Optional[str] = Query(default=None, description="Optional client-generated request id"),
    session_id: Optional[str] = Query(default=None, description="Optional session id for analytics attribution"),
    context: TenantContext = Depends(resolve_tenant_context),
):
    """Personalize recommendations from a user's recent implicit feedback events."""
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

