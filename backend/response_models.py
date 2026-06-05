"""
Pydantic response models for the APEX Movie Recommendation API.

Extracted from backend/main.py to keep that module under 800 lines.
"""

from typing import Literal

from pydantic import BaseModel


class Movie(BaseModel):
    """Movie response model."""

    id: int
    title: str
    overview: str | None = None
    genres: str | None = None
    vote_average: float | None = None
    vote_count: float | None = None
    popularity: float | None = None
    release_date: str | None = None
    poster_path: str | None = None
    metadata_completeness: float | None = None
    content_quality_score: float | None = None
    quality_bucket: str | None = None
    recommendable: bool | None = None
    similarity_score: float | None = None
    ranker_score: float | None = None
    retrieval_stage: str | None = None
    retrieval_signals: dict | None = None
    semantic_twin: dict | None = None
    semantic_signals: dict | None = None
    explanation_text: str | None = None
    explanation: list[str] | None = None


class MovieTitle(BaseModel):
    """Lightweight movie title response for autocomplete."""

    id: int
    title: str


class EnrichedMovie(BaseModel):
    """Movie with TMDB enrichment data."""

    id: int
    title: str
    overview: str | None = None
    genres: str | None = None
    vote_average: float | None = None
    vote_count: float | None = None
    popularity: float | None = None
    release_date: str | None = None
    poster_path: str | None = None
    metadata_completeness: float | None = None
    content_quality_score: float | None = None
    quality_bucket: str | None = None
    recommendable: bool | None = None
    similarity_score: float | None = None
    retrieval_stage: str | None = None
    retrieval_signals: dict | None = None
    semantic_twin: dict | None = None
    semantic_signals: dict | None = None
    explanation_text: str | None = None
    explanation: list[str] | None = None
    # Enriched fields
    trailer_key: str | None = None
    runtime: int | None = None
    director: str | None = None
    cast: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    movie_count: int
    app_version: str | None = None
    app_commit: str | None = None
    serving_tier: str | None = None
    hardware_profile: dict | None = None
    tier_selection_reason: str | None = None


class RecommendationResponse(BaseModel):
    """Recommendation response."""

    request_id: str | None = None
    query_movie: Movie
    recommendations: list[Movie]


class EnrichedRecommendationResponse(BaseModel):
    """Enriched recommendation response with TMDB data."""

    request_id: str | None = None
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
    tenant_id: str | None = None
    catalog_id: str | None = None
    content_id: str | None = None
    source_content_id: str | None = None
    movie_id: int | None = None
    query_text: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    rating: float | None = None
    request_id: str | None = None
    metadata: dict | None = None


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
    last_seen: str | None = None
    operation_counts: dict[str, int]
    tenant_counts: dict[str, int]
