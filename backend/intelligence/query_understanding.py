"""
Neuro-Symbolic Query Understanding & Vibe Compiler for APEX search.

Decomposes natural language queries into:
1. Genre / taxonomy intent
2. Atmospheric & emotional vibe profiles (Cyberpunk, Cozy, Mind-Bending, Noir, etc.)
3. Temporal & quality priors
4. Explainable rationale signals for multi-modal scoring
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

GENRE_ALIASES = {
    "action": {"action", "fight", "fighting", "explosions", "martial arts", "chase"},
    "adventure": {"adventure", "quest", "journey", "exploration", "survival"},
    "animation": {"animation", "animated", "anime", "ghibli", "pixar", "cartoon"},
    "comedy": {"comedy", "funny", "humor", "hilarious", "laugh", "satire"},
    "crime": {"crime", "gangster", "mafia", "heist", "mob", "underworld"},
    "documentary": {"documentary", "docuseries", "real story", "true story", "biopic"},
    "drama": {"drama", "emotional", "serious", "character driven", "intense drama"},
    "family": {"family", "kids", "children", "child friendly", "all ages"},
    "fantasy": {"fantasy", "magic", "wizard", "mythical", "dragons", "spell"},
    "horror": {"horror", "scary", "haunted", "ghost", "creepy", "slasher", "monster"},
    "mystery": {"mystery", "detective", "whodunit", "clues", "investigation"},
    "romance": {"romance", "romantic", "love story", "lovers", "relationship"},
    "science fiction": {"science fiction", "sci fi", "sci-fi", "space", "alien", "future", "ai", "robot"},
    "thriller": {"thriller", "suspense", "tense", "edge of your seat", "plot twist"},
    "war": {"war", "military", "battlefield", "soldier", "army", "combat"},
    "western": {"western", "cowboy", "outlaw", "gunslinger"},
}

VIBE_PROFILES: dict[str, dict[str, Any]] = {
    "cyberpunk": {
        "keywords": {"cyberpunk", "synthwave", "neon", "dystopia", "dystopian", "cyborg", "futuristic neon", "blade runner"},
        "associated_genres": {"science fiction", "action", "thriller"},
        "description": "neon cyberpunk & dystopian futurism",
        "boost": 0.08,
    },
    "mind_bending": {
        "keywords": {"mind bending", "mind-bending", "psychological", "surreal", "plot twist", "reality bending", "dream", "illusion", "time loop"},
        "associated_genres": {"science fiction", "thriller", "mystery"},
        "description": "mind-bending psychological mystery",
        "boost": 0.09,
    },
    "cozy_melancholic": {
        "keywords": {"cozy", "rainy", "melancholic", "peaceful", "comforting", "rain", "autumn", "bittersweet", "slow paced"},
        "associated_genres": {"drama", "romance", "animation"},
        "description": "warm, contemplative & cozy atmosphere",
        "boost": 0.07,
    },
    "dark_noir": {
        "keywords": {"noir", "neo-noir", "dark", "gritty", "brooding", "cynical", "seedy", "shadowy"},
        "associated_genres": {"crime", "mystery", "thriller"},
        "description": "gritty neo-noir & atmospheric tension",
        "boost": 0.08,
    },
    "high_octane": {
        "keywords": {"adrenaline", "high octane", "high-octane", "action packed", "relentless", "fast paced", "explosive"},
        "associated_genres": {"action", "thriller", "adventure"},
        "description": "relentless high-octane energy",
        "boost": 0.07,
    },
    "heartwarming": {
        "keywords": {"heartwarming", "uplifting", "feel good", "feel-good", "wholesome", "inspiring", "joyful"},
        "associated_genres": {"family", "comedy", "drama", "animation"},
        "description": "heartwarming & uplifting inspiration",
        "boost": 0.07,
    },
}


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower()).strip()


def parse_query_intent(query: str) -> dict[str, Any]:
    """Extract multi-modal ranking and vibe intent from a natural language query."""
    text = _normalize_query(query)
    genres = []
    for genre, aliases in GENRE_ALIASES.items():
        if any(alias in text for alias in aliases):
            genres.append(genre)

    matched_vibes = []
    for vibe_key, vibe_info in VIBE_PROFILES.items():
        if any(kw in text for kw in vibe_info["keywords"]):
            matched_vibes.append(vibe_key)

    return {
        "genres": sorted(set(genres)),
        "vibes": sorted(set(matched_vibes)),
        "recent": any(token in text for token in ("new", "latest", "recent", "modern", "2024", "2025", "2026")),
        "classic": any(token in text for token in ("classic", "old", "retro", "vintage", "80s", "90s", "70s")),
        "high_quality": any(token in text for token in ("best", "top rated", "high rated", "critically acclaimed", "masterpiece")),
        "family_safe": any(token in text for token in ("family", "kids", "children", "child friendly")),
    }


def intent_score(
    movie: dict[str, Any], intent: dict[str, Any], current_year: int | None = None
) -> tuple[float, list[str]]:
    """Return a bounded score adjustment and explainable rationale bullets for a movie."""
    score = 0.0
    reasons = []
    movie_genres = {part.strip().lower() for part in str(movie.get("genres") or "").split(",") if part.strip()}
    desired_genres = set(intent.get("genres") or [])

    # 1. Genre Overlap
    if desired_genres:
        overlap = desired_genres & movie_genres
        if overlap:
            score += min(0.14, 0.06 * len(overlap))
            reasons.append(f"matches genre intent ({', '.join(sorted(overlap))})")
        else:
            score -= 0.025

    # 2. Vibe & Mood Affinity
    overview_text = str(movie.get("overview") or "").lower()
    title_text = str(movie.get("title") or "").lower()
    movie_full_text = f"{title_text} {overview_text}"

    for vibe_key in intent.get("vibes", []):
        vibe_info = VIBE_PROFILES.get(vibe_key)
        if not vibe_info:
            continue
        vibe_genre_match = bool(movie_genres & vibe_info["associated_genres"])
        vibe_keyword_match = any(kw in movie_full_text for kw in vibe_info["keywords"])

        if vibe_genre_match or vibe_keyword_match:
            score += vibe_info["boost"]
            reasons.append(f"matches {vibe_info['description']}")

    # 3. Temporal Signals
    release_year = None
    with contextlib.suppress(TypeError, ValueError):
        release_year = int(str(movie.get("release_date") or "")[:4])

    current_year = current_year or 2026
    if intent.get("recent") and release_year and current_year - release_year <= 5:
        score += 0.06
        reasons.append("matches recent-release intent")
    if intent.get("classic") and release_year and release_year <= 2000:
        score += 0.05
        reasons.append("matches classic-catalog intent")

    # 4. Quality & Rating Signals
    vote_average = float(movie.get("vote_average") or 0)
    vote_count = float(movie.get("vote_count") or 0)
    if intent.get("high_quality") and vote_average >= 7.0 and vote_count >= 100:
        score += 0.05
        reasons.append("matches high-quality masterpiece intent")

    if intent.get("family_safe") and ("family" in movie_genres or "animation" in movie_genres):
        score += 0.04
        reasons.append("matches family-safe intent")

    return max(-0.08, min(0.35, score)), reasons[:3]
