"""
Temporal Preference Modeling for APEX.

Models how user taste evolves over time, not just what they like right now.

Key insight: A user who watched action movies 2 years ago but has been watching
dramas for the last 3 months has SHIFTED preferences. Standard collaborative
filtering treats all historical interactions equally — this module weights
recent interactions more heavily using exponential time decay.

This is the same approach used by Netflix's "taste evolution" system
(published in their 2022 RecSys paper).

Integration: Called from _build_rl_state in recommender.py to produce
a temporally-aware user state vector.
"""

from __future__ import annotations

import math
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Half-life for preference decay: interactions older than this get 50% weight
PREFERENCE_HALF_LIFE_DAYS = 90.0


def _days_ago(event_ts: str) -> float:
    """Return how many days ago an event occurred. Returns 0 for unparseable timestamps."""
    try:
        if event_ts.endswith("Z"):
            event_ts = event_ts[:-1] + "+00:00"
        event_dt = datetime.fromisoformat(event_ts)
        now = datetime.now(UTC)
        if event_dt.tzinfo is None:
            from datetime import timezone
            event_dt = event_dt.replace(tzinfo=timezone.utc)
        delta = now - event_dt
        return max(0.0, delta.total_seconds() / 86400.0)
    except Exception:
        return 0.0


def temporal_decay_weight(event_ts: str, half_life_days: float = PREFERENCE_HALF_LIFE_DAYS) -> float:
    """
    Compute exponential decay weight for an event based on its age.

    w(t) = 2^(-t / half_life)

    Recent events get weight ~1.0, events from half_life_days ago get 0.5,
    events from 2*half_life_days ago get 0.25, etc.
    """
    days = _days_ago(event_ts)
    return math.pow(2.0, -days / half_life_days)


def build_temporal_user_profile(
    user_id: str,
    events: list[dict[str, Any]],
    half_life_days: float = PREFERENCE_HALF_LIFE_DAYS,
) -> dict[str, Any]:
    """
    Build a temporally-weighted user preference profile.

    Returns:
        - recent_genre_weights: dict mapping genre → decayed weight sum
        - temporal_avg_rating: time-weighted average rating
        - preference_velocity: rate of change in taste (high = shifting preferences)
        - recency_score: how recently the user was active (0-1)
        - interaction_count_recent: interactions in last 30 days
        - interaction_count_total: all-time interactions
    """
    if not events:
        return {
            "user_id": user_id,
            "recent_genre_weights": {},
            "temporal_avg_rating": 3.0,
            "preference_velocity": 0.0,
            "recency_score": 0.0,
            "interaction_count_recent": 0,
            "interaction_count_total": 0,
        }

    genre_weights: dict[str, float] = {}
    rating_sum = 0.0
    weight_sum = 0.0
    recent_count = 0
    total_count = len(events)

    # For velocity: compare recent vs old genre preferences
    recent_genres: dict[str, float] = {}
    old_genres: dict[str, float] = {}

    for event in events:
        et = str(event.get("event_type", "")).lower()
        if et not in {"rating", "click", "view"}:
            continue

        ts = str(event.get("event_ts") or "")
        w = temporal_decay_weight(ts, half_life_days)
        days = _days_ago(ts)

        if days <= 30:
            recent_count += 1

        # Genre weighting
        genres_str = str(event.get("genres") or event.get("metadata", {}).get("genres", "") or "")
        for genre in genres_str.split(","):
            genre = genre.strip().lower()
            if genre:
                genre_weights[genre] = genre_weights.get(genre, 0.0) + w
                if days <= 60:
                    recent_genres[genre] = recent_genres.get(genre, 0.0) + w
                else:
                    old_genres[genre] = old_genres.get(genre, 0.0) + w

        # Temporal rating average
        rating = event.get("rating")
        if rating is not None:
            try:
                rating_sum += float(rating) * w
                weight_sum += w
            except (TypeError, ValueError):
                pass

    temporal_avg_rating = rating_sum / weight_sum if weight_sum > 0 else 3.0

    # Preference velocity: cosine distance between recent and old genre vectors
    all_genres = set(recent_genres) | set(old_genres)
    if all_genres and (recent_genres or old_genres):
        r_vec = np.array([recent_genres.get(g, 0.0) for g in all_genres])
        o_vec = np.array([old_genres.get(g, 0.0) for g in all_genres])
        r_norm = np.linalg.norm(r_vec)
        o_norm = np.linalg.norm(o_vec)
        if r_norm > 0 and o_norm > 0:
            cosine_sim = float(np.dot(r_vec, o_vec) / (r_norm * o_norm))
            preference_velocity = 1.0 - cosine_sim  # 0 = stable, 1 = completely shifted
        else:
            preference_velocity = 0.0
    else:
        preference_velocity = 0.0

    # Recency score: how recently was the user active (0-1)
    most_recent_ts = max((str(e.get("event_ts") or "") for e in events), default="")
    most_recent_days = _days_ago(most_recent_ts) if most_recent_ts else 365.0
    recency_score = math.exp(-most_recent_days / 30.0)  # Decays over 30 days

    return {
        "user_id": user_id,
        "recent_genre_weights": genre_weights,
        "temporal_avg_rating": round(temporal_avg_rating, 3),
        "preference_velocity": round(preference_velocity, 4),
        "recency_score": round(recency_score, 4),
        "interaction_count_recent": recent_count,
        "interaction_count_total": total_count,
    }


def temporal_score_boost(
    candidate_genres: str,
    temporal_profile: dict[str, Any],
    boost_scale: float = 0.05,
) -> float:
    """
    Compute a temporal preference boost for a candidate item.

    Items matching the user's RECENT genre preferences get a positive boost.
    Items matching only OLD preferences get a smaller boost.
    Items matching neither get no boost.

    Args:
        candidate_genres: Comma-separated genre string for the candidate
        temporal_profile: Output of build_temporal_user_profile
        boost_scale: Maximum boost magnitude (default 0.05 = 5% of base score)

    Returns:
        Float boost value in [-boost_scale, +boost_scale]
    """
    genre_weights = temporal_profile.get("recent_genre_weights", {})
    if not genre_weights:
        return 0.0

    candidate_genre_set = {g.strip().lower() for g in candidate_genres.split(",") if g.strip()}
    if not candidate_genre_set:
        return 0.0

    # Sum of temporal weights for matching genres
    match_weight = sum(genre_weights.get(g, 0.0) for g in candidate_genre_set)
    total_weight = sum(genre_weights.values()) or 1.0

    # Normalize to [-1, 1] then scale
    normalized = (match_weight / total_weight) * 2.0 - 1.0
    return float(np.clip(normalized * boost_scale, -boost_scale, boost_scale))
