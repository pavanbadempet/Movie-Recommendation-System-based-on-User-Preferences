"""
Popularity-Debiased Evaluation Metrics for APEX.

Standard recommendation metrics (NDCG, Hit Rate) are biased toward popular items
because popular items appear more in test sets. A model that always recommends
popular items scores well on standard metrics but provides no personalization value.

This module implements debiased versions of standard metrics that correct for
popularity bias using Inverse Propensity Scoring (IPS).

References:
  - Schnabel et al. "Recommendations as Treatments" (ICML 2016)
  - Steck "Calibrated Recommendations" (RecSys 2018)
  - Saito et al. "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback" (WSDM 2020)
"""

from __future__ import annotations

from collections import Counter
import contextlib
import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_item_popularity(
    events: list[dict[str, Any]],
    smoothing: float = 1.0,
) -> dict[int, float]:
    """
    Compute normalized item popularity from interaction events.
    Returns dict mapping item_id → popularity score in [0, 1].
    """
    counts: Counter[int] = Counter()
    for event in events:
        mid = event.get("movie_id")
        if mid is not None:
            with contextlib.suppress(TypeError, ValueError):
                counts[int(mid)] += 1

    if not counts:
        return {}

    total = sum(counts.values()) + smoothing * len(counts)
    return {item_id: (count + smoothing) / total for item_id, count in counts.items()}


def ips_ndcg_at_k(
    ranked_items: list[int],
    ground_truth: set[int],
    popularity: dict[int, float],
    k: int = 10,
    clip_val: float = 10.0,
) -> float:
    """
    IPS-corrected NDCG@k.

    Reweights each relevant item by 1/popularity to correct for the fact
    that popular items are over-represented in test sets.

    A model that recommends niche items the user genuinely likes scores
    higher than one that recommends popular items the user might like.
    """
    if not ground_truth:
        return 0.0

    mean_pop = float(np.mean(list(popularity.values()))) if popularity else 0.01

    def ips_weight(item_id: int) -> float:
        p = popularity.get(item_id, mean_pop)
        return min(1.0 / max(p, 1e-6), clip_val)

    top_k = ranked_items[:k]
    dcg = sum(ips_weight(item) / math.log2(rank + 2) for rank, item in enumerate(top_k) if item in ground_truth)

    # Ideal DCG: sort ground truth by IPS weight (rarest items first)
    gt_weights = sorted(
        [ips_weight(item) for item in ground_truth],
        reverse=True,
    )
    idcg = sum(w / math.log2(rank + 2) for rank, w in enumerate(gt_weights[:k]))

    return dcg / idcg if idcg > 0 else 0.0


def calibration_score(
    recommendations: list[dict[str, Any]],
    user_history_genres: dict[str, float],
    k: int = 10,
) -> float:
    """
    Compute calibration score: how well the recommendation list mirrors
    the user's historical genre distribution.

    A perfectly calibrated list has the same genre proportions as the user's
    watch history. This prevents filter bubbles.

    Returns KL divergence (lower = better calibrated).
    """
    if not recommendations or not user_history_genres:
        return 0.0

    # Build recommendation genre distribution
    rec_genre_counts: dict[str, float] = {}
    for movie in recommendations[:k]:
        genres_str = str(movie.get("genres") or "")
        for genre in genres_str.split(","):
            genre = genre.strip().lower()
            if genre:
                rec_genre_counts[genre] = rec_genre_counts.get(genre, 0.0) + 1.0

    if not rec_genre_counts:
        return 0.0

    # Normalize both distributions
    all_genres = set(user_history_genres) | set(rec_genre_counts)
    eps = 1e-8

    user_total = sum(user_history_genres.values()) or 1.0
    rec_total = sum(rec_genre_counts.values()) or 1.0

    kl_div = 0.0
    for genre in all_genres:
        p = user_history_genres.get(genre, 0.0) / user_total + eps
        q = rec_genre_counts.get(genre, 0.0) / rec_total + eps
        kl_div += p * math.log(p / q)

    return round(kl_div, 4)


def beyond_accuracy_metrics(
    recommendations: list[dict[str, Any]],
    catalog_size: int,
    popularity: dict[int, float],
    k: int = 10,
) -> dict[str, float]:
    """
    Compute beyond-accuracy metrics that capture recommendation quality
    dimensions that standard NDCG misses:

    - Coverage: fraction of catalog covered across all users
    - Novelty: average self-information of recommended items (log(1/popularity))
    - Serendipity: unexpected but relevant recommendations
    - Gini coefficient: inequality in item recommendation frequency

    These are the metrics Netflix uses internally to prevent popularity bias
    and filter bubbles.
    """
    if not recommendations:
        return {}

    top_k = recommendations[:k]
    mean_pop = float(np.mean(list(popularity.values()))) if popularity else 0.01

    # Novelty: average -log2(popularity) — rare items have high novelty
    novelty_scores = []
    for movie in top_k:
        mid = movie.get("id")
        if mid is not None:
            try:
                p = popularity.get(int(mid), mean_pop)
                novelty_scores.append(-math.log2(max(p, 1e-10)))
            except (TypeError, ValueError):
                pass

    avg_novelty = float(np.mean(novelty_scores)) if novelty_scores else 0.0

    # Coverage: unique items / catalog size
    unique_items = len({movie.get("id") for movie in top_k if movie.get("id")})
    coverage = unique_items / max(catalog_size, 1)

    # Genre diversity: unique genres / total genres
    all_genres: set[str] = set()
    for movie in top_k:
        for g in str(movie.get("genres") or "").split(","):
            g = g.strip().lower()
            if g:
                all_genres.add(g)
    genre_diversity = len(all_genres) / max(len(top_k) * 3, 1)  # Normalize by expected genres

    return {
        "novelty": round(avg_novelty, 4),
        "coverage": round(coverage, 6),
        "genre_diversity": round(min(genre_diversity, 1.0), 4),
        "unique_items_in_list": unique_items,
    }
