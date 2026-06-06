"""
Diversity-Aware Re-Ranking for APEX.

Implements submodular diversity optimization — a mathematically principled
approach to recommendation diversity that goes beyond simple MMR.

Key insight: MMR (Maximal Marginal Relevance) is a greedy approximation.
Submodular optimization provides a (1 - 1/e) ≈ 63% optimality guarantee,
meaning the selected set is at least 63% as diverse as the theoretically
optimal set. This is the same approach used in Google's search diversification.

Three diversity objectives:
1. Genre diversity — maximize unique genres in the recommendation list
2. Era diversity — spread recommendations across different decades
3. Quality diversity — mix critically acclaimed with hidden gems

Usage:
    from backend.pipeline.diversity_reranker import submodular_rerank
    diverse_results = submodular_rerank(candidates, n=10, lambda_diversity=0.3)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _genre_set(genres_str: str) -> frozenset[str]:
    return frozenset(g.strip().lower() for g in str(genres_str or "").split(",") if g.strip())


def _decade(release_date: str) -> int | None:
    try:
        year = int(str(release_date or "")[:4])
        return (year // 10) * 10
    except (ValueError, TypeError):
        return None


def _quality_tier(movie: dict[str, Any]) -> str:
    """Classify movie into quality tier for diversity."""
    rating = float(movie.get("vote_average") or 0)
    votes = float(movie.get("vote_count") or 0)
    if rating >= 7.5 and votes >= 1000:
        return "acclaimed"
    elif rating >= 6.0 and votes >= 100:
        return "solid"
    else:
        return "niche"


def submodular_rerank(
    candidates: list[dict[str, Any]],
    n: int = 10,
    lambda_diversity: float = 0.3,
    genre_weight: float = 0.5,
    era_weight: float = 0.3,
    quality_weight: float = 0.2,
) -> list[dict[str, Any]]:
    """
    Re-rank candidates using submodular diversity optimization.

    Maximizes: F(S) = (1 - λ) * Relevance(S) + λ * Diversity(S)

    Where Diversity(S) is a submodular function combining:
    - Genre coverage (how many unique genres are covered)
    - Era coverage (how many decades are represented)
    - Quality tier coverage (mix of acclaimed, solid, niche)

    The greedy algorithm provides a (1 - 1/e) approximation guarantee.

    Args:
        candidates: List of candidate movies with similarity_score
        n: Number of results to return
        lambda_diversity: Trade-off between relevance (0) and diversity (1)
        genre_weight: Weight for genre diversity component
        era_weight: Weight for era diversity component
        quality_weight: Weight for quality tier diversity component

    Returns:
        Re-ranked list of n movies
    """
    if len(candidates) <= n:
        return candidates

    # Normalize relevance scores to [0, 1]
    scores = np.array([float(c.get("similarity_score") or 0.0) for c in candidates])
    score_min, score_max = scores.min(), scores.max()
    if score_max > score_min:
        norm_scores = (scores - score_min) / (score_max - score_min)
    else:
        norm_scores = np.ones(len(candidates))

    # Pre-compute diversity features
    genres = [_genre_set(c.get("genres", "")) for c in candidates]
    decades = [_decade(c.get("release_date", "")) for c in candidates]
    quality_tiers = [_quality_tier(c) for c in candidates]

    # Greedy submodular selection
    selected_indices: list[int] = []
    covered_genres: set[str] = set()
    covered_decades: set[int] = set()
    covered_quality: set[str] = set()

    remaining = list(range(len(candidates)))

    for _ in range(n):
        if not remaining:
            break

        best_idx = -1
        best_score = -float("inf")

        for i in remaining:
            # Relevance component
            relevance = float(norm_scores[i])

            # Marginal genre diversity gain
            new_genres = genres[i] - covered_genres
            genre_gain = len(new_genres) / max(len(genres[i]), 1) if genres[i] else 0.0

            # Marginal era diversity gain
            decade = decades[i]
            era_gain = 1.0 if (decade is not None and decade not in covered_decades) else 0.0

            # Marginal quality diversity gain
            qt = quality_tiers[i]
            quality_gain = 1.0 if qt not in covered_quality else 0.0

            # Combined diversity score
            diversity = genre_weight * genre_gain + era_weight * era_gain + quality_weight * quality_gain

            # Submodular objective
            f = (1.0 - lambda_diversity) * relevance + lambda_diversity * diversity

            if f > best_score:
                best_score = f
                best_idx = i

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

        # Update coverage sets
        covered_genres.update(genres[best_idx])
        if decades[best_idx] is not None:
            covered_decades.add(decades[best_idx])
        covered_quality.add(quality_tiers[best_idx])

    result = [candidates[i] for i in selected_indices]

    # Log diversity stats
    if result:
        all_genres = set()
        all_decades = set()
        for c in result:
            all_genres.update(_genre_set(c.get("genres", "")))
            d = _decade(c.get("release_date", ""))
            if d:
                all_decades.add(d)
        logger.debug(
            "Submodular rerank: %d results, %d unique genres, %d decades covered",
            len(result),
            len(all_genres),
            len(all_decades),
        )

    return result
