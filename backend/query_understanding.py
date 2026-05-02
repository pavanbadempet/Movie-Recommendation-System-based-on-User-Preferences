"""
Free-tier query understanding for Nova search.

This is not pretending to be an LLM. It extracts high-signal intent from a
short natural language query so the hybrid retriever can make better ranking
tradeoffs without paid inference.
"""

from __future__ import annotations

import re
from typing import Any

GENRE_ALIASES = {
    "action": {"action", "fight", "fighting", "explosions"},
    "adventure": {"adventure", "quest", "journey", "exploration"},
    "animation": {"animation", "animated", "anime"},
    "comedy": {"comedy", "funny", "humor", "hilarious"},
    "crime": {"crime", "gangster", "mafia", "heist"},
    "documentary": {"documentary", "docuseries", "real story"},
    "drama": {"drama", "emotional", "serious"},
    "family": {"family", "kids", "children", "child friendly"},
    "fantasy": {"fantasy", "magic", "wizard", "mythical"},
    "horror": {"horror", "scary", "haunted", "ghost"},
    "mystery": {"mystery", "detective", "whodunit"},
    "romance": {"romance", "romantic", "love story"},
    "science fiction": {"science fiction", "sci fi", "sci-fi", "space", "alien", "future"},
    "thriller": {"thriller", "suspense", "tense"},
    "war": {"war", "military", "battlefield"},
    "western": {"western", "cowboy"},
}


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower()).strip()


def parse_query_intent(query: str) -> dict[str, Any]:
    """Extract lightweight ranking intent from a natural language query."""
    text = _normalize_query(query)
    genres = []
    for genre, aliases in GENRE_ALIASES.items():
        if any(alias in text for alias in aliases):
            genres.append(genre)

    return {
        "genres": sorted(set(genres)),
        "recent": any(token in text for token in ("new", "latest", "recent", "modern", "2024", "2025", "2026")),
        "classic": any(token in text for token in ("classic", "old", "retro", "vintage", "80s", "90s")),
        "high_quality": any(token in text for token in ("best", "top rated", "high rated", "critically acclaimed")),
        "family_safe": any(token in text for token in ("family", "kids", "children", "child friendly")),
    }


def intent_score(movie: dict[str, Any], intent: dict[str, Any], current_year: int | None = None) -> tuple[float, list[str]]:
    """Return a bounded score adjustment and explanations for a movie."""
    score = 0.0
    reasons = []
    movie_genres = {
        part.strip().lower()
        for part in str(movie.get("genres") or "").split(",")
        if part.strip()
    }
    desired_genres = set(intent.get("genres") or [])
    if desired_genres:
        overlap = desired_genres & movie_genres
        if overlap:
            score += min(0.14, 0.06 * len(overlap))
            reasons.append("matches requested genre intent")
        else:
            score -= 0.025

    release_year = None
    try:
        release_year = int(str(movie.get("release_date") or "")[:4])
    except (TypeError, ValueError):
        pass

    current_year = current_year or 2026
    if intent.get("recent") and release_year and current_year - release_year <= 5:
        score += 0.06
        reasons.append("matches recent-release intent")
    if intent.get("classic") and release_year and release_year <= 2000:
        score += 0.05
        reasons.append("matches classic-catalog intent")

    vote_average = float(movie.get("vote_average") or 0)
    vote_count = float(movie.get("vote_count") or 0)
    if intent.get("high_quality") and vote_average >= 7.0 and vote_count >= 100:
        score += 0.05
        reasons.append("matches high-quality intent")

    if intent.get("family_safe") and ("family" in movie_genres or "animation" in movie_genres):
        score += 0.04
        reasons.append("matches family-safe intent")

    return max(-0.08, min(0.22, score)), reasons[:3]

