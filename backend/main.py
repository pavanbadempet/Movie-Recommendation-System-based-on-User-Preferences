"""
FastAPI backend for the Movie Recommendation System.
Provides REST API endpoints for movie search and recommendations.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.recommender import get_recommender, Recommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# CORS configuration for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
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
    similarity_score: Optional[float] = None


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
    similarity_score: Optional[float] = None
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


# Lazy-load recommender on first request
_recommender: Recommender | None = None


def get_rec() -> Recommender:
    """Get recommender instance, loading on first call."""
    global _recommender
    if _recommender is None:
        logger.info("Loading recommender on first request...")
        _recommender = get_recommender()
    return _recommender


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
    try:
        rec = get_rec()
        return HealthResponse(status="healthy", movie_count=len(rec.movies))
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="unhealthy", movie_count=0)


@app.get("/movies", response_model=list[Movie])
async def list_movies(
    limit: int = Query(default=100, le=1000, description="Maximum movies to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    """List movies with pagination."""
    rec = get_rec()
    movies = rec.movies.iloc[offset:offset + limit]
    return movies.to_dict(orient="records")


@app.get("/search", response_model=list[Movie])
async def search_movies(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=20, le=100, description="Maximum results"),
):
    """Search movies by title."""
    rec = get_rec()
    results = rec.search_movies(q, limit=limit)
    return results


@app.get("/movie/{movie_id}", response_model=Movie)
async def get_movie(movie_id: int):
    """Get a movie by TMDB ID."""
    rec = get_rec()
    movie = rec.get_movie_by_id(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
    return movie


@app.get("/recommend/id/{movie_id}", response_model=RecommendationResponse)
async def recommend_by_id(
    movie_id: int,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
):
    """Get recommendations for a movie by TMDB ID."""
    rec = get_rec()
    
    # Get query movie
    query_movie = rec.get_movie_by_id(movie_id)
    if query_movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
    
    # Get recommendations
    recommendations = rec.recommend_by_id(movie_id, n=n)
    
    return RecommendationResponse(
        query_movie=query_movie,
        recommendations=recommendations,
    )


@app.get("/recommend/id/{movie_id}/enriched", response_model=EnrichedRecommendationResponse)
async def recommend_by_id_enriched(
    movie_id: int,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
):
    """Get recommendations with FULL TMDB data (trailers, cast, etc) - PARALLEL FETCH."""
    rec = get_rec()
    
    # Get query movie
    query_movie = rec.get_movie_by_id(movie_id)
    if query_movie is None:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
    
    # Get recommendations
    recommendations = rec.recommend_by_id(movie_id, n=n)
    
    # Enrich all movies in parallel
    enriched = await asyncio.gather(*[enrich_movie(m) for m in recommendations])
    
    return EnrichedRecommendationResponse(
        query_movie=query_movie,
        recommendations=enriched,
    )


@app.get("/recommend/title/{title}", response_model=RecommendationResponse)
async def recommend_by_title(
    title: str,
    n: int = Query(default=10, le=50, description="Number of recommendations"),
):
    """Get recommendations for a movie by title."""
    rec = get_rec()
    
    # Search for the movie
    matches = rec.search_movies(title, limit=1)
    if not matches:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found")
    
    query_movie = matches[0]
    
    # Get recommendations
    recommendations = rec.recommend_by_title(title, n=n)
    
    return RecommendationResponse(
        query_movie=query_movie,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

