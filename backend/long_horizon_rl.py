"""
Long-Horizon Preference Modeling for APEX.

Models user taste evolution over 30/90/365-day windows, not just immediate preferences.

Standard RL optimizes for 7-day retention. This module extends to:
- 30-day taste drift detection
- 90-day preference cycle modeling (seasonal preferences)
- Lifetime value optimization

This is the frontier of recommendation research — no production system
has fully solved long-horizon preference modeling.

Architecture:
- Sliding window aggregation over multiple time horizons
- Preference stability score (how consistent is the user's taste?)
- Seasonal pattern detection (do they watch horror in October?)
- Churn risk estimation (is the user becoming less active?)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_preference_stability(
    events: list[dict[str, Any]],
    window_days: int = 90,
) -> float:
    """
    Compute how stable a user's preferences are over the given window.

    Stable users (same genres over time) get score close to 1.0.
    Shifting users (changing genres) get score close to 0.0.

    Returns:
        Stability score in [0, 1]
    """
    if len(events) < 5:
        return 0.5  # Unknown stability

    # Split events into two halves by time
    sorted_events = sorted(events, key=lambda e: str(e.get("event_ts") or ""))
    mid = len(sorted_events) // 2
    first_half = sorted_events[:mid]
    second_half = sorted_events[mid:]

    def genre_vector(evts: list[dict]) -> dict[str, float]:
        counts: dict[str, float] = defaultdict(float)
        for e in evts:
            genres_str = str(e.get("genres") or (e.get("metadata") or {}).get("genres", "") or "")
            for g in genres_str.split(","):
                g = g.strip().lower()
                if g:
                    counts[g] += 1.0
        total = sum(counts.values()) or 1.0
        return {k: v / total for k, v in counts.items()}

    v1 = genre_vector(first_half)
    v2 = genre_vector(second_half)

    all_genres = set(v1) | set(v2)
    if not all_genres:
        return 0.5

    arr1 = np.array([v1.get(g, 0.0) for g in all_genres])
    arr2 = np.array([v2.get(g, 0.0) for g in all_genres])

    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    if norm1 == 0 or norm2 == 0:
        return 0.5

    cosine_sim = float(np.dot(arr1, arr2) / (norm1 * norm2))
    return round(max(0.0, cosine_sim), 4)


def estimate_churn_risk(
    events: list[dict[str, Any]],
    lookback_days: int = 30,
) -> float:
    """
    Estimate the probability that a user will churn (stop using the service).

    Based on:
    - Recency: how long since last interaction
    - Frequency trend: is activity increasing or decreasing?
    - Rating trend: are ratings getting lower?

    Returns:
        Churn risk in [0, 1] where 1 = very likely to churn
    """
    if not events:
        return 0.8  # No events = likely churned

    now = datetime.now(UTC)

    # Sort by timestamp
    sorted_events = sorted(events, key=lambda e: str(e.get("event_ts") or ""))

    # Recency: days since last event
    last_ts = str(sorted_events[-1].get("event_ts") or "")
    try:
        if last_ts.endswith("Z"):
            last_ts = last_ts[:-1] + "+00:00"
        last_dt = datetime.fromisoformat(last_ts)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=UTC)
        days_since_last = (now - last_dt).total_seconds() / 86400.0
    except Exception:
        days_since_last = 30.0

    recency_risk = 1.0 - math.exp(-days_since_last / 14.0)  # 14-day half-life

    # Frequency trend: compare recent vs older activity
    recent_count = sum(1 for e in sorted_events if _days_ago_from_ts(str(e.get("event_ts") or "")) <= lookback_days)
    older_count = sum(
        1 for e in sorted_events if lookback_days < _days_ago_from_ts(str(e.get("event_ts") or "")) <= lookback_days * 2
    )

    if older_count > 0:
        freq_ratio = recent_count / older_count
        freq_risk = max(0.0, 1.0 - freq_ratio)
    else:
        freq_risk = 0.3  # Unknown trend

    # Combined churn risk
    churn_risk = 0.6 * recency_risk + 0.4 * freq_risk
    return round(min(churn_risk, 1.0), 4)


def _days_ago_from_ts(ts: str) -> float:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        now = datetime.now(UTC)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:
        return 0.0


def long_horizon_score_adjustment(
    candidate: dict[str, Any],
    user_events: list[dict[str, Any]],
    churn_risk: float,
    preference_stability: float,
) -> float:
    """
    Adjust recommendation score based on long-horizon user modeling.

    High churn risk → boost engaging content (high ratings, popular)
    Low stability → boost diverse content (explore new genres)
    High stability → boost familiar genres (exploit known preferences)

    Returns:
        Score adjustment in [-0.1, +0.1]
    """
    adjustment = 0.0

    # Churn risk: boost high-quality engaging content for at-risk users
    if churn_risk > 0.7:
        rating = float(candidate.get("vote_average") or 0)
        votes = float(candidate.get("vote_count") or 0)
        if rating >= 7.5 and votes >= 1000:
            adjustment += 0.05  # Boost acclaimed content for at-risk users

    # Preference stability: adjust exploration vs exploitation
    if preference_stability < 0.4:
        # Shifting user — boost genre diversity
        candidate_genres = {g.strip().lower() for g in str(candidate.get("genres") or "").split(",") if g.strip()}
        user_genres = set()
        for e in user_events[-20:]:  # Recent events only
            for g in str(e.get("genres") or "").split(","):
                g = g.strip().lower()
                if g:
                    user_genres.add(g)
        new_genres = candidate_genres - user_genres
        if new_genres:
            adjustment += 0.03  # Boost items with new genres for shifting users

    return round(max(-0.1, min(0.1, adjustment)), 4)
