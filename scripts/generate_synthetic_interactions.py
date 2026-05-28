"""
Synthetic Interaction Generator

Generates realistic user behavior events from the movie catalog using
persona-based simulation. This bootstraps the Event Store so that:
  - SASRec gets real session sequences
  - The ensemble weight optimizer has a validation set
  - The Two-Tower fine-tuning script has positive pairs
  - The RL policy trains on meaningful reward signals

Persona types (each with distinct taste profiles):
  1. Action Junkie     — loves action/thriller, high ratings for blockbusters
  2. Arthouse Fan      — prefers drama/foreign, rates critically
  3. Comedy Lover      — watches comedies, clicks a lot, rates generously
  4. Horror Enthusiast — horror/mystery, binge-watches in sessions
  5. Family Viewer     — animation/family, consistent moderate ratings
  6. Sci-Fi Nerd       — sci-fi/fantasy, very selective, high standards
  7. Romance Fan       — romance/drama, emotional rater
  8. Documentary Buff  — documentary/history, rates thoughtfully
  9. Casual Viewer     — watches everything, average ratings
  10. Cinephile        — all genres, very high standards, rare 5-star ratings

Usage:
    python scripts/generate_synthetic_interactions.py [--users N] [--events-per-user N] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.events import append_event

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

PERSONAS = [
    {
        "name": "action_junkie",
        "preferred_genres": ["Action", "Thriller", "Adventure", "Crime"],
        "avoided_genres": ["Romance", "Documentary", "Animation"],
        "avg_rating": 3.8,
        "rating_std": 0.8,
        "click_rate": 0.6,       # probability of clicking a recommended movie
        "view_rate": 0.4,
        "session_length": (3, 8), # min/max movies per session
        "sessions_per_month": (8, 15),
        "quality_threshold": 6.0, # min vote_average to consider watching
    },
    {
        "name": "arthouse_fan",
        "preferred_genres": ["Drama", "Foreign", "History", "War"],
        "avoided_genres": ["Action", "Horror", "Animation", "Comedy"],
        "avg_rating": 3.5,
        "rating_std": 1.2,
        "click_rate": 0.3,
        "view_rate": 0.5,
        "session_length": (1, 3),
        "sessions_per_month": (4, 8),
        "quality_threshold": 7.0,
    },
    {
        "name": "comedy_lover",
        "preferred_genres": ["Comedy", "Romance", "Family", "Animation"],
        "avoided_genres": ["Horror", "War", "Documentary"],
        "avg_rating": 4.1,
        "rating_std": 0.6,
        "click_rate": 0.7,
        "view_rate": 0.5,
        "session_length": (2, 6),
        "sessions_per_month": (10, 20),
        "quality_threshold": 5.5,
    },
    {
        "name": "horror_enthusiast",
        "preferred_genres": ["Horror", "Mystery", "Thriller", "Crime"],
        "avoided_genres": ["Animation", "Family", "Romance", "Comedy"],
        "avg_rating": 3.6,
        "rating_std": 1.0,
        "click_rate": 0.65,
        "view_rate": 0.45,
        "session_length": (3, 7),
        "sessions_per_month": (6, 12),
        "quality_threshold": 5.0,
    },
    {
        "name": "family_viewer",
        "preferred_genres": ["Animation", "Family", "Comedy", "Adventure"],
        "avoided_genres": ["Horror", "Crime", "War", "Thriller"],
        "avg_rating": 3.9,
        "rating_std": 0.5,
        "click_rate": 0.5,
        "view_rate": 0.6,
        "session_length": (2, 5),
        "sessions_per_month": (5, 10),
        "quality_threshold": 6.0,
    },
    {
        "name": "scifi_nerd",
        "preferred_genres": ["Science Fiction", "Fantasy", "Adventure", "Action"],
        "avoided_genres": ["Romance", "Documentary", "History"],
        "avg_rating": 3.4,
        "rating_std": 1.3,
        "click_rate": 0.4,
        "view_rate": 0.35,
        "session_length": (2, 5),
        "sessions_per_month": (6, 12),
        "quality_threshold": 6.5,
    },
    {
        "name": "romance_fan",
        "preferred_genres": ["Romance", "Drama", "Comedy", "Music"],
        "avoided_genres": ["Horror", "Action", "War", "Science Fiction"],
        "avg_rating": 4.2,
        "rating_std": 0.7,
        "click_rate": 0.6,
        "view_rate": 0.55,
        "session_length": (2, 4),
        "sessions_per_month": (8, 16),
        "quality_threshold": 5.5,
    },
    {
        "name": "documentary_buff",
        "preferred_genres": ["Documentary", "History", "Biography", "Crime"],
        "avoided_genres": ["Animation", "Fantasy", "Horror", "Comedy"],
        "avg_rating": 3.7,
        "rating_std": 0.9,
        "click_rate": 0.35,
        "view_rate": 0.6,
        "session_length": (1, 3),
        "sessions_per_month": (4, 8),
        "quality_threshold": 7.0,
    },
    {
        "name": "casual_viewer",
        "preferred_genres": [],  # watches everything
        "avoided_genres": [],
        "avg_rating": 3.5,
        "rating_std": 0.9,
        "click_rate": 0.5,
        "view_rate": 0.4,
        "session_length": (2, 6),
        "sessions_per_month": (5, 12),
        "quality_threshold": 5.0,
    },
    {
        "name": "cinephile",
        "preferred_genres": ["Drama", "Crime", "Thriller", "History", "War"],
        "avoided_genres": ["Animation", "Family"],
        "avg_rating": 3.2,
        "rating_std": 1.5,
        "click_rate": 0.25,
        "view_rate": 0.5,
        "session_length": (1, 3),
        "sessions_per_month": (4, 10),
        "quality_threshold": 7.5,
    },
]


# ---------------------------------------------------------------------------
# Movie scoring for a persona
# ---------------------------------------------------------------------------

def _genre_match_score(movie_genres: str, persona: dict) -> float:
    """Return a 0–1 affinity score between a movie's genres and a persona."""
    if not persona["preferred_genres"] and not persona["avoided_genres"]:
        return 0.5  # casual viewer — neutral

    genres = {g.strip().lower() for g in str(movie_genres or "").split(",")}
    preferred = {g.lower() for g in persona["preferred_genres"]}
    avoided = {g.lower() for g in persona["avoided_genres"]}

    overlap_pref = len(genres & preferred)
    overlap_avoid = len(genres & avoided)

    score = 0.5 + (overlap_pref * 0.15) - (overlap_avoid * 0.20)
    return max(0.0, min(1.0, score))


def _simulate_rating(base_rating: float, persona: dict, rng: random.Random) -> float:
    """Simulate a user rating given the movie's quality and persona bias."""
    # Base: persona's average rating shifted by movie quality
    quality_signal = (base_rating - 6.0) / 4.0  # normalize 2–10 → -1 to 1
    raw = persona["avg_rating"] + quality_signal * 1.5
    noise = rng.gauss(0, persona["rating_std"] * 0.5)
    rating = raw + noise
    return max(1.0, min(5.0, round(rating * 2) / 2))  # snap to 0.5 increments


# ---------------------------------------------------------------------------
# Timestamp generation
# ---------------------------------------------------------------------------

def _random_timestamp(base_dt: datetime, rng: random.Random, spread_days: int = 90) -> str:
    """Return a random ISO timestamp within spread_days of base_dt."""
    offset = timedelta(
        days=rng.randint(0, spread_days),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
        seconds=rng.randint(0, 59),
    )
    return (base_dt + offset).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def generate(
    num_users: int = 500,
    events_per_user: int = 40,
    seed: int = 42,
) -> int:
    """
    Generate synthetic interactions and write them to the Event Store.

    Returns the total number of events written.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Load movie catalog
    movies_path = DATA_DIR / "movies_transformed.parquet"
    if not movies_path.exists():
        logger.error("movies_transformed.parquet not found at %s", movies_path)
        sys.exit(1)

    logger.info("Loading movie catalog...")
    cols = ["id", "title", "genres", "vote_average", "vote_count", "release_date",
            "popularity", "recommendable"]
    try:
        movies_df = pd.read_parquet(movies_path, columns=cols)
    except (KeyError, ValueError):
        movies_df = pd.read_parquet(movies_path)

    # Filter to recommendable, quality movies
    movies_df = movies_df[
        (pd.to_numeric(movies_df["vote_average"], errors="coerce").fillna(0) >= 5.0) &
        (pd.to_numeric(movies_df["vote_count"], errors="coerce").fillna(0) >= 50)
    ].copy()
    movies_df["vote_average"] = pd.to_numeric(movies_df["vote_average"], errors="coerce").fillna(6.0)
    movies_df["id"] = pd.to_numeric(movies_df["id"], errors="coerce").dropna().astype(int)
    movies_df = movies_df.dropna(subset=["id"])

    logger.info("Catalog: %d quality movies available", len(movies_df))

    if len(movies_df) == 0:
        logger.error("No movies found after filtering.")
        sys.exit(1)

    base_dt = datetime.now(UTC) - timedelta(days=90)
    total_events = 0
    persona_cycle = PERSONAS * (num_users // len(PERSONAS) + 1)

    logger.info("Generating interactions for %d synthetic users (%d events each)...",
                num_users, events_per_user)

    for user_idx in range(num_users):
        persona = persona_cycle[user_idx % len(PERSONAS)]
        user_id = f"synthetic_{persona['name']}_{user_idx:04d}"
        session_id = str(uuid.uuid4())

        # Score all movies for this persona
        scores = movies_df.apply(
            lambda row: _genre_match_score(row.get("genres", ""), persona) *
                        (float(row.get("vote_average", 6.0)) / 10.0),
            axis=1,
        ).values.astype(np.float64)

        # Softmax to get sampling probabilities
        scores = scores - scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()

        # Sample movies this user interacts with
        n_movies = min(events_per_user * 2, len(movies_df))
        sampled_indices = np_rng.choice(len(movies_df), size=n_movies, replace=False, p=probs)
        sampled_movies = movies_df.iloc[sampled_indices]

        events_written = 0
        for _, movie in sampled_movies.iterrows():
            if events_written >= events_per_user:
                break

            movie_id = int(movie["id"])
            vote_avg = float(movie.get("vote_average", 6.0))
            genre_score = _genre_match_score(movie.get("genres", ""), persona)

            # Skip if below quality threshold
            if vote_avg < persona["quality_threshold"] and rng.random() > 0.2:
                continue

            ts = _random_timestamp(base_dt, rng)

            # View event (always)
            if rng.random() < persona["view_rate"]:
                try:
                    append_event({
                        "event_type": "view",
                        "user_id": user_id,
                        "session_id": session_id,
                        "movie_id": movie_id,
                        "event_ts": ts,
                        "tenant_id": "synthetic",
                        "catalog_id": "tmdb-movies",
                    })
                    total_events += 1
                    events_written += 1
                except Exception:
                    pass

            # Click event
            click_prob = persona["click_rate"] * (0.5 + genre_score)
            if rng.random() < click_prob:
                try:
                    append_event({
                        "event_type": "click",
                        "user_id": user_id,
                        "session_id": session_id,
                        "movie_id": movie_id,
                        "event_ts": ts,
                        "tenant_id": "synthetic",
                        "catalog_id": "tmdb-movies",
                    })
                    total_events += 1
                    events_written += 1
                except Exception:
                    pass

            # Rating event (subset of clicks)
            if rng.random() < 0.6 and genre_score > 0.3:
                rating = _simulate_rating(vote_avg, persona, rng)
                try:
                    append_event({
                        "event_type": "rating",
                        "user_id": user_id,
                        "session_id": session_id,
                        "movie_id": movie_id,
                        "rating": rating,
                        "event_ts": ts,
                        "tenant_id": "synthetic",
                        "catalog_id": "tmdb-movies",
                    })
                    total_events += 1
                    events_written += 1
                except Exception:
                    pass

        if (user_idx + 1) % 50 == 0:
            logger.info("  %d/%d users processed, %d events written so far",
                        user_idx + 1, num_users, total_events)

    logger.info("=" * 60)
    logger.info("Synthetic interaction generation complete.")
    logger.info("  Users: %d", num_users)
    logger.info("  Total events written: %d", total_events)
    logger.info("  Avg events per user: %.1f", total_events / max(num_users, 1))
    logger.info("  Persona distribution: %d personas x %d users each",
                len(PERSONAS), num_users // len(PERSONAS))
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. python scripts/optimize_ensemble_weights.py")
    logger.info("  2. python scripts/train_rl_policy_compact.py")
    logger.info("  3. python scripts/finetune_two_tower.py")
    logger.info("=" * 60)

    return total_events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic user interactions to bootstrap the APEX Event Store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--users", type=int, default=500,
        help="Number of synthetic users to generate",
    )
    parser.add_argument(
        "--events-per-user", type=int, default=40,
        help="Target number of events per user",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    total = generate(
        num_users=args.users,
        events_per_user=args.events_per_user,
        seed=args.seed,
    )
    print(f"\nDone. {total:,} events written to the Event Store.")
    print("Run the calibration scripts to activate all models.")
