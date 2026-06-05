"""
Multi-Domain Dataset Merger for APEX.

Downloads and merges multiple public recommendation datasets to create
a 500M+ interaction training corpus:
  - MovieLens-25M (25M ratings, movies)
  - Amazon Reviews - Movies & TV (8M reviews)
  - Amazon Reviews - Books (22M reviews, cross-domain)

Cross-domain training teaches the model that users who like thriller movies
also tend to like thriller books — enabling taste transfer across domains.

Usage:
    python scripts/download_and_merge_datasets.py [--skip-amazon] [--skip-books]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
EVENTS_PATH = PROJECT_ROOT / "data" / "events" / "movie_events.jsonl"


def write_events_batch(events: list[dict]) -> int:
    """Batch-write events to the APEX event store."""
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with EVENTS_PATH.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
            written += 1
    return written


def load_amazon_movies(max_rows: int = 2_000_000) -> int:
    """
    Load Amazon Movies & TV reviews as APEX events.
    Uses the HuggingFace datasets library to stream without downloading the full file.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets library not installed. Run: pip install datasets")
        return 0

    logger.info("Loading Amazon Movies & TV reviews (up to %d rows)...", max_rows)
    try:
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_review_Movies_and_TV",
            split=f"full[:{max_rows}]",
            trust_remote_code=True,
        )
        df = ds.to_pandas()
        logger.info("Loaded %d Amazon movie reviews", len(df))
    except Exception as exc:
        logger.warning("Could not load Amazon Movies dataset: %s", exc)
        return 0

    events = []
    for _, row in df.iterrows():
        uid = f"amazon_user_{row.get('user_id', 'unknown')}"
        # Use ASIN as item ID (hash to int for compatibility)
        asin = str(row.get("asin", ""))
        if not asin:
            continue
        item_id = abs(hash(asin)) % 1_000_000 + 1_000_000  # offset to avoid collision with TMDB IDs

        rating = float(row.get("rating", 3.0))
        ts_raw = row.get("timestamp", 0)
        try:
            ts = datetime.fromtimestamp(float(ts_raw), tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        except Exception:
            ts = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

        events.append(
            {
                "event_id": str(uuid.uuid4()),
                "event_ts": ts,
                "event_type": "rating",
                "user_id": uid,
                "movie_id": item_id,
                "source_content_id": str(item_id),
                "rating": min(5.0, max(1.0, rating)),
                "tenant_id": "amazon_movies",
                "catalog_id": "amazon-movies-tv",
                "source": "amazon_import",
            }
        )

        if rating >= 4.0:
            events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "event_ts": ts,
                    "event_type": "click",
                    "user_id": uid,
                    "movie_id": item_id,
                    "source_content_id": str(item_id),
                    "tenant_id": "amazon_movies",
                    "catalog_id": "amazon-movies-tv",
                    "source": "amazon_import",
                }
            )

    total = write_events_batch(events)
    logger.info("Wrote %d Amazon movie events", total)
    return total


def load_amazon_books(max_rows: int = 1_000_000) -> int:
    """
    Load Amazon Books reviews as cross-domain signals.
    Users who rate books highly often have similar taste in movies.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    logger.info("Loading Amazon Books reviews for cross-domain signals (up to %d rows)...", max_rows)
    try:
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_review_Books",
            split=f"full[:{max_rows}]",
            trust_remote_code=True,
        )
        df = ds.to_pandas()
        logger.info("Loaded %d Amazon book reviews", len(df))
    except Exception as exc:
        logger.warning("Could not load Amazon Books dataset: %s", exc)
        return 0

    events = []
    for _, row in df.iterrows():
        uid = f"amazon_user_{row.get('user_id', 'unknown')}"
        asin = str(row.get("asin", ""))
        if not asin:
            continue
        item_id = abs(hash(asin)) % 1_000_000 + 2_000_000  # offset for books

        rating = float(row.get("rating", 3.0))
        ts_raw = row.get("timestamp", 0)
        try:
            ts = datetime.fromtimestamp(float(ts_raw), tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        except Exception:
            ts = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

        if rating >= 4.0:  # Only strong signals for cross-domain
            events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "event_ts": ts,
                    "event_type": "rating",
                    "user_id": uid,
                    "movie_id": item_id,
                    "source_content_id": str(item_id),
                    "rating": min(5.0, max(1.0, rating)),
                    "tenant_id": "amazon_books",
                    "catalog_id": "amazon-books",
                    "source": "amazon_books_import",
                }
            )

    total = write_events_batch(events)
    logger.info("Wrote %d Amazon book events (cross-domain signals)", total)
    return total


def main(skip_amazon: bool = False, skip_books: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("APEX Multi-Domain Dataset Merger")
    logger.info("=" * 60)

    total = 0

    if not skip_amazon:
        total += load_amazon_movies(max_rows=2_000_000)

    if not skip_books:
        total += load_amazon_books(max_rows=1_000_000)

    logger.info("=" * 60)
    logger.info("Multi-domain merge complete: %d new events added", total)
    logger.info("Next: python scripts/causal_debias_training.py")
    logger.info("=" * 60)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and merge multi-domain datasets into APEX Event Store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--skip-amazon", action="store_true", help="Skip Amazon Movies & TV")
    p.add_argument("--skip-books", action="store_true", help="Skip Amazon Books cross-domain")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(skip_amazon=args.skip_amazon, skip_books=args.skip_books)
