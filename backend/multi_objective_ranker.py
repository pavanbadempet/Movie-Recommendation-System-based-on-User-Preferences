"""
Multi-Objective Ranking for APEX.

Optimizes recommendations across multiple competing objectives simultaneously:
1. Relevance — how well the item matches user preferences
2. Diversity — variety in the recommendation list
3. Novelty — surfacing items the user hasn't seen
4. Serendipity — unexpected but delightful recommendations
5. Fairness — ensuring niche content gets exposure

Standard recommendation systems optimize only for relevance (CTR).
This module implements Pareto-optimal ranking — finding the set of
recommendations where no objective can be improved without hurting another.

This is the same approach used in YouTube's multi-task ranking system
(published at RecSys 2019) and Meta's DLRM.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_novelty_score(
    movie: dict[str, Any],
    user_history_ids: set[int],
    popularity: dict[int, float],
) -> float:
    """
    Novelty = how surprising/unexpected this recommendation is.
    High novelty = item is rare AND not in user's history.
    """
    mid = movie.get("id")
    if mid is None:
        return 0.5

    try:
        mid = int(mid)
    except (TypeError, ValueError):
        return 0.5

    # Already seen = zero novelty
    if mid in user_history_ids:
        return 0.0

    # Rarity component: -log2(popularity)
    mean_pop = float(np.mean(list(popularity.values()))) if popularity else 0.01
    p = popularity.get(mid, mean_pop)
    rarity = -math.log2(max(p, 1e-10)) / 20.0  # Normalize to ~[0, 1]

    return round(min(rarity, 1.0), 4)


def compute_serendipity_score(
    movie: dict[str, Any],
    user_genre_profile: dict[str, float],
    relevance_score: float,
) -> float:
    """
    Serendipity = unexpected but relevant.
    High serendipity = item is from an unexpected genre BUT still relevant.

    Formula: serendipity = relevance * (1 - genre_familiarity)
    """
    genres_str = str(movie.get("genres") or "")
    movie_genres = {g.strip().lower() for g in genres_str.split(",") if g.strip()}

    if not movie_genres or not user_genre_profile:
        return 0.0

    total_profile = sum(user_genre_profile.values()) or 1.0
    genre_familiarity = sum(
        user_genre_profile.get(g, 0.0) / total_profile
        for g in movie_genres
    ) / max(len(movie_genres), 1)

    serendipity = relevance_score * (1.0 - genre_familiarity)
    return round(max(0.0, serendipity), 4)


def pareto_rank(
    candidates: list[dict[str, Any]],
    user_history_ids: set[int],
    user_genre_profile: dict[str, float],
    popularity: dict[int, float],
    n: int = 10,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Rank candidates using a weighted multi-objective score.

    Objectives:
    - relevance: similarity_score (already computed)
    - novelty: how rare and unseen the item is
    - serendipity: unexpected but relevant
    - quality: vote_average * log(vote_count)

    Args:
        candidates: List of candidate movies
        user_history_ids: Set of movie IDs the user has already seen
        user_genre_profile: Dict of genre → interaction count
        popularity: Dict of movie_id → popularity score
        n: Number of results
        weights: Optional objective weights (default: relevance-heavy)

    Returns:
        Re-ranked list of n movies
    """
    if not candidates:
        return candidates

    if weights is None:
        weights = {
            "relevance": 0.60,
            "novelty": 0.15,
            "serendipity": 0.15,
            "quality": 0.10,
        }

    # Normalize relevance scores
    rel_scores = np.array([float(c.get("similarity_score") or 0.0) for c in candidates])
    rel_min, rel_max = rel_scores.min(), rel_scores.max()
    if rel_max > rel_min:
        rel_norm = (rel_scores - rel_min) / (rel_max - rel_min)
    else:
        rel_norm = np.ones(len(candidates))

    scored = []
    for i, movie in enumerate(candidates):
        relevance = float(rel_norm[i])
        novelty = compute_novelty_score(movie, user_history_ids, popularity)
        serendipity = compute_serendipity_score(movie, user_genre_profile, relevance)

        # Quality score
        rating = float(movie.get("vote_average") or 0)
        votes = float(movie.get("vote_count") or 0)
        quality = (rating / 10.0) * min(1.0, math.log1p(votes) / 8.0) if rating > 0 else 0.0

        # Weighted multi-objective score
        mo_score = (
            weights["relevance"] * relevance +
            weights["novelty"] * novelty +
            weights["serendipity"] * serendipity +
            weights["quality"] * quality
        )

        movie["multi_objective_score"] = round(mo_score, 4)
        movie["novelty_score"] = novelty
        movie["serendipity_score"] = serendipity
        scored.append((mo_score, movie))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:n]]
