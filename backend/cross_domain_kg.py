"""
Cross-Domain Knowledge Graph for APEX.

Extends the existing Knowledge Graph to support cross-domain taste transfer:
  Movies ↔ Books ↔ Music ↔ TV Shows

Key insight: A user who loves "The Dark Knight" (thriller, moral complexity)
likely also enjoys "Gone Girl" (book), "Breaking Bad" (TV), and "Radiohead" (music).
These cross-domain signals dramatically improve cold-start recommendations.

This is the same approach used by Spotify's "Taste Profile" system which
uses listening history to improve podcast and podcast-to-music recommendations.

Architecture:
  - Genre/theme nodes are shared across domains
  - User preference vectors are projected into a shared embedding space
  - Cross-domain edges enable multi-hop reasoning across content types
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# Cross-domain genre/theme mappings
# Maps movie genres to equivalent signals in other domains
CROSS_DOMAIN_MAPPINGS: dict[str, dict[str, list[str]]] = {
    "Action": {
        "books": ["thriller", "adventure", "military fiction"],
        "music": ["rock", "metal", "electronic"],
        "tv": ["action", "crime drama"],
    },
    "Drama": {
        "books": ["literary fiction", "contemporary fiction", "memoir"],
        "music": ["indie", "folk", "classical"],
        "tv": ["drama", "prestige tv"],
    },
    "Comedy": {
        "books": ["humor", "satire", "light fiction"],
        "music": ["pop", "indie pop"],
        "tv": ["sitcom", "comedy"],
    },
    "Horror": {
        "books": ["horror", "dark fantasy", "psychological thriller"],
        "music": ["metal", "dark ambient", "industrial"],
        "tv": ["horror", "supernatural"],
    },
    "Science Fiction": {
        "books": ["science fiction", "hard sci-fi", "cyberpunk"],
        "music": ["electronic", "ambient", "synth"],
        "tv": ["sci-fi", "speculative fiction"],
    },
    "Romance": {
        "books": ["romance", "contemporary romance", "chick lit"],
        "music": ["pop", "r&b", "soul"],
        "tv": ["romance", "drama"],
    },
    "Documentary": {
        "books": ["non-fiction", "biography", "history"],
        "music": ["world music", "jazz", "classical"],
        "tv": ["documentary", "news"],
    },
    "Animation": {
        "books": ["graphic novels", "manga", "children's fiction"],
        "music": ["pop", "electronic", "soundtrack"],
        "tv": ["animation", "anime"],
    },
    "Thriller": {
        "books": ["thriller", "mystery", "crime fiction"],
        "music": ["electronic", "dark pop", "industrial"],
        "tv": ["thriller", "crime drama", "mystery"],
    },
    "Crime": {
        "books": ["crime fiction", "noir", "detective fiction"],
        "music": ["hip-hop", "jazz", "blues"],
        "tv": ["crime drama", "procedural"],
    },
}


def get_cross_domain_signals(
    movie_genres: str,
    user_cross_domain_events: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """
    Generate cross-domain affinity signals for a movie based on its genres.

    Returns a dict of {signal_name: weight} that can be used to boost
    recommendations for users with cross-domain preferences.

    Args:
        movie_genres: Comma-separated genre string
        user_cross_domain_events: Optional list of user events from other domains

    Returns:
        Dict of cross-domain signal weights
    """
    signals: dict[str, float] = {}
    genres = {g.strip() for g in movie_genres.split(",") if g.strip()}

    for genre in genres:
        mapping = CROSS_DOMAIN_MAPPINGS.get(genre, {})
        for domain, tags in mapping.items():
            for tag in tags:
                key = f"{domain}:{tag}"
                signals[key] = signals.get(key, 0.0) + 1.0

    # Normalize
    if signals:
        max_val = max(signals.values())
        if max_val > 0:
            signals = {k: v / max_val for k, v in signals.items()}

    return signals


def cross_domain_user_affinity(
    user_events: list[dict[str, Any]],
    movie_genres: str,
) -> float:
    """
    Compute cross-domain affinity score between a user's taste profile
    and a candidate movie's genres.

    Uses the user's interaction history across all domains (movies, books, etc.)
    to infer taste signals that transfer to the movie domain.

    Args:
        user_events: List of user events (may include cross-domain events)
        movie_genres: Comma-separated genre string for the candidate movie

    Returns:
        Float affinity score in [0, 1]
    """
    if not user_events or not movie_genres:
        return 0.0

    # Build user's cross-domain taste profile from their events
    user_domain_tags: dict[str, float] = defaultdict(float)
    for event in user_events:
        catalog = str(event.get("catalog_id", "")).lower()
        genres_str = str(event.get("genres") or
                         (event.get("metadata") or {}).get("genres", "") or "")

        # Determine domain from catalog_id
        if "book" in catalog:
            domain = "books"
        elif "music" in catalog or "spotify" in catalog:
            domain = "music"
        elif "tv" in catalog or "show" in catalog:
            domain = "tv"
        else:
            continue  # Skip movie events (handled by main recommender)

        for genre in genres_str.split(","):
            genre = genre.strip().lower()
            if genre:
                user_domain_tags[f"{domain}:{genre}"] += 1.0

    if not user_domain_tags:
        return 0.0

    # Get cross-domain signals for the candidate movie
    movie_signals = get_cross_domain_signals(movie_genres)
    if not movie_signals:
        return 0.0

    # Compute overlap between user's cross-domain tags and movie's signals
    overlap = sum(
        user_domain_tags.get(signal, 0.0) * weight
        for signal, weight in movie_signals.items()
    )

    # Normalize by user's total cross-domain activity
    total_user_activity = sum(user_domain_tags.values()) or 1.0
    return float(min(overlap / total_user_activity, 1.0))


def enrich_knowledge_graph_with_cross_domain(kg_engine: Any) -> None:
    """
    Add cross-domain nodes and edges to the existing Knowledge Graph.

    Adds:
    - Domain nodes (BOOKS, MUSIC, TV)
    - Cross-domain genre bridge edges
    - Enables multi-hop queries like: Movie(Action) → BOOKS(thriller) → User

    Args:
        kg_engine: KnowledgeGraphEngine instance
    """
    try:
        graph = kg_engine.graph
        domains = ["BOOKS", "MUSIC", "TV"]

        # Add domain nodes
        for domain in domains:
            if not graph.has_node(f"DOMAIN_{domain}"):
                graph.add_node(f"DOMAIN_{domain}", type="DOMAIN", name=domain)

        # Add cross-domain bridge edges
        for movie_genre, domain_mappings in CROSS_DOMAIN_MAPPINGS.items():
            genre_node = f"GENRE_{movie_genre}"
            if not graph.has_node(genre_node):
                graph.add_node(genre_node, type="GENRE", name=movie_genre)

            for domain, tags in domain_mappings.items():
                domain_node = f"DOMAIN_{domain.upper()}"
                for tag in tags:
                    tag_node = f"{domain.upper()}_TAG_{tag}"
                    if not graph.has_node(tag_node):
                        graph.add_node(tag_node, type="CROSS_DOMAIN_TAG",
                                       name=tag, domain=domain)
                    # Bridge: movie genre → cross-domain tag
                    if not graph.has_edge(genre_node, tag_node):
                        graph.add_edge(genre_node, tag_node,
                                       relation="CROSS_DOMAIN_EQUIVALENT",
                                       weight=0.7)
                    # Tag → domain
                    if not graph.has_edge(tag_node, domain_node):
                        graph.add_edge(tag_node, domain_node,
                                       relation="BELONGS_TO_DOMAIN")

        logger.info(
            "Cross-domain KG enrichment complete: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
    except Exception as exc:
        logger.warning("Cross-domain KG enrichment failed: %s", exc)
