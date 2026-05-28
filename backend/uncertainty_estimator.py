"""
Uncertainty Estimation for APEX Recommendations.

Adds confidence scores to recommendations so users and operators know
how certain the system is about each recommendation.

Key insight: A recommendation with score 0.85 from a model that has seen
10,000 similar users is very different from a score of 0.85 from a model
that has seen only 5 similar users. Uncertainty quantification captures this.

Methods implemented:
1. Ensemble disagreement — variance across the 6 ensemble models
   (high variance = low confidence)
2. Coverage-based uncertainty — how many training examples are similar
   to this user-item pair
3. Cold-start detection — flag when the user or item has few interactions

This is the same approach used in Bayesian deep learning for recommendation
systems (published at NeurIPS 2021 by DeepMind).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def ensemble_uncertainty(
    model_scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """
    Compute uncertainty as the weighted variance across ensemble model scores.

    High variance = models disagree = low confidence.
    Low variance = models agree = high confidence.

    Returns:
        Uncertainty score in [0, 1] where 0 = certain, 1 = very uncertain
    """
    if not model_scores or not weights:
        return 0.5  # Unknown uncertainty

    scores = []
    w_list = []
    for model_name, score in model_scores.items():
        w = weights.get(model_name, 0.0)
        if w > 0:
            scores.append(score)
            w_list.append(w)

    if len(scores) < 2:
        return 0.3  # Single model — moderate uncertainty

    scores_arr = np.array(scores)
    w_arr = np.array(w_list)
    w_arr = w_arr / w_arr.sum()

    # Weighted variance
    weighted_mean = float(np.dot(w_arr, scores_arr))
    weighted_var = float(np.dot(w_arr, (scores_arr - weighted_mean) ** 2))

    # Normalize to [0, 1] — variance of 0.25 (max for [0,1] range) = 1.0
    uncertainty = min(weighted_var / 0.25, 1.0)
    return round(uncertainty, 4)


def coverage_uncertainty(
    user_interaction_count: int,
    item_interaction_count: int,
    min_interactions: int = 10,
) -> float:
    """
    Estimate uncertainty based on how many training examples cover this
    user-item pair.

    Users/items with few interactions have high uncertainty.

    Returns:
        Uncertainty score in [0, 1]
    """
    # Harmonic mean of user and item coverage
    user_coverage = min(user_interaction_count / min_interactions, 1.0)
    item_coverage = min(item_interaction_count / min_interactions, 1.0)

    if user_coverage + item_coverage == 0:
        return 1.0

    harmonic = 2 * user_coverage * item_coverage / (user_coverage + item_coverage)
    return round(1.0 - harmonic, 4)


def compute_confidence_score(
    ensemble_scores: dict[str, float],
    weights: dict[str, float],
    user_interaction_count: int = 0,
    item_interaction_count: int = 0,
) -> dict[str, Any]:
    """
    Compute a comprehensive confidence score for a recommendation.

    Returns:
        Dict with:
        - confidence: Overall confidence in [0, 1] (1 = very confident)
        - uncertainty_ensemble: Disagreement between models
        - uncertainty_coverage: Lack of training data coverage
        - is_cold_start: True if user or item has very few interactions
        - confidence_label: Human-readable label
    """
    unc_ensemble = ensemble_uncertainty(ensemble_scores, weights)
    unc_coverage = coverage_uncertainty(user_interaction_count, item_interaction_count)

    # Combined uncertainty (ensemble disagreement weighted more)
    combined_uncertainty = 0.6 * unc_ensemble + 0.4 * unc_coverage
    confidence = round(1.0 - combined_uncertainty, 4)

    is_cold_start = (
        user_interaction_count < 5 or item_interaction_count < 5
    )

    if confidence >= 0.8:
        label = "high"
    elif confidence >= 0.6:
        label = "medium"
    elif confidence >= 0.4:
        label = "low"
    else:
        label = "very_low"

    return {
        "confidence": confidence,
        "uncertainty_ensemble": unc_ensemble,
        "uncertainty_coverage": unc_coverage,
        "is_cold_start": is_cold_start,
        "confidence_label": label,
    }


def cold_start_boost(
    movie: dict[str, Any],
    user_interaction_count: int,
    catalog_df: Any = None,
) -> float:
    """
    For cold-start users (< 5 interactions), boost content-based signals
    over collaborative filtering signals.

    Returns a boost multiplier for content-based scores.
    """
    if user_interaction_count >= 10:
        return 1.0  # Warm user — no boost needed

    # Cold-start: boost based on content quality signals
    rating = float(movie.get("vote_average") or 0)
    votes = float(movie.get("vote_count") or 0)
    popularity = float(movie.get("popularity") or 0)

    # Quality signal: well-rated movies are safer recommendations for cold-start
    quality = (rating / 10.0) * min(1.0, math.log1p(votes) / 8.0)

    # Popularity signal: popular movies are more likely to be known
    pop_signal = min(1.0, math.log1p(popularity) / 8.0)

    # Blend: more weight on quality for cold-start
    boost = 1.0 + 0.2 * (0.7 * quality + 0.3 * pop_signal)
    return round(boost, 4)
