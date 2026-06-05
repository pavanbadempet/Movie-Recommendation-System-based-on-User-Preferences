"""
Offline evaluation pipeline for the APEX Movie Recommendation System.

Implements leave-one-out evaluation against MovieLens 100K.

Usage:
    python scripts/run_offline_evaluation.py [--output reports/offline_eval_report.json]
"""

import argparse
import contextlib
from datetime import UTC, datetime
import json
import logging
from math import log2
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

# Determinism: set seed before any sampling
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_DATA_FILE = "u.data"
MOVIELENS_ZIP_SUBDIR = "ml-100k"


# ---------------------------------------------------------------------------
# Task 1.1 — Dataset loading with auto-download
# ---------------------------------------------------------------------------


def load_movielens_100k(data_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """Load MovieLens 100K ratings, downloading the dataset if necessary.

    Checks for ``data_dir/u.data`` (tab-separated: user_id, item_id, rating,
    timestamp). If absent, downloads the zip from GroupLens, extracts it to
    ``data_dir``, and logs an INFO message.

    Returns
    -------
    pd.DataFrame
        Columns: user_id (int), item_id (int), rating (float), timestamp (int)
    """
    data_dir = Path(data_dir)
    target_file = data_dir / MOVIELENS_DATA_FILE

    if not target_file.exists():
        logger.info(
            "MovieLens 100K data not found at '%s'. Downloading from %s …",
            target_file,
            MOVIELENS_100K_URL,
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        _download_and_extract_movielens(data_dir)
        logger.info("Download complete. Data extracted to '%s'.", data_dir)

    logger.info("Loading ratings from '%s' …", target_file)
    df = pd.read_csv(
        target_file,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )
    logger.info(
        "Loaded %d ratings for %d users and %d items.",
        len(df),
        df["user_id"].nunique(),
        df["item_id"].nunique(),
    )
    return df


def _download_and_extract_movielens(data_dir: Path) -> None:
    """Download the MovieLens 100K zip and extract ``u.data`` to *data_dir*."""
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(MOVIELENS_100K_URL, tmp_path)
        with zipfile.ZipFile(tmp_path, "r") as zf:
            # The zip contains a subdirectory ml-100k/; extract everything there
            # then move u.data up to data_dir.
            zf.extractall(data_dir)

        # ml-100k/ is extracted as a subdirectory; move u.data to data_dir root
        extracted_subdir = data_dir / MOVIELENS_ZIP_SUBDIR
        source_file = extracted_subdir / MOVIELENS_DATA_FILE
        dest_file = data_dir / MOVIELENS_DATA_FILE

        if source_file.exists() and not dest_file.exists():
            source_file.rename(dest_file)
            logger.info("Moved '%s' → '%s'.", source_file, dest_file)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Task 1.2 — Leave-one-out split
# ---------------------------------------------------------------------------


def leave_one_out_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """Split ratings using leave-one-out per user.

    Sorts all interactions by (user_id, timestamp) ascending. For each user,
    holds out the last interaction as the test item; the remaining interactions
    form the training set.

    Parameters
    ----------
    df:
        DataFrame with columns user_id, item_id, rating, timestamp.

    Returns
    -------
    train_df : pd.DataFrame
        All interactions except each user's last.
    test_df : pd.DataFrame
        Each user's last interaction (one row per user).
    cold_start_users : set
        Set of user_ids whose training interaction count is ≤5.
    """
    df = df.sort_values(["user_id", "timestamp"], ascending=True).reset_index(drop=True)

    # For each user, the last row (by timestamp) is the test item
    last_idx = df.groupby("user_id").tail(1).index
    test_mask = df.index.isin(last_idx)

    train_df = df[~test_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    # Cold-start users: ≤5 training interactions.
    # Users with 0 training rows (only 1 total interaction) are also cold-start.
    all_users = set(df["user_id"].unique())
    train_counts = train_df.groupby("user_id").size()
    users_with_few_train = set(train_counts[train_counts <= 5].index.tolist())
    users_with_no_train = all_users - set(train_counts.index.tolist())
    cold_start_users: set = users_with_few_train | users_with_no_train

    logger.info(
        "Split complete: %d train rows, %d test rows, %d cold-start users (≤5 train interactions).",
        len(train_df),
        len(test_df),
        len(cold_start_users),
    )
    return train_df, test_df, cold_start_users


# ---------------------------------------------------------------------------
# Task 1.3 — Per-user recommendation and metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cold_start_users: set,
    output_path: Path,
    recommender=None,
) -> dict:
    """Compute NDCG@10, Recall@50, ILD, Cold-Start NDCG@10 and write the report."""
    if recommender is None:
        logger.warning("No recommender provided — skipping metric computation.")
        return {}

    # Build a lookup: user_id -> last training item_id
    last_train_item = train_df.sort_values(["user_id", "timestamp"]).groupby("user_id")["item_id"].last().to_dict()

    ndcg_scores = []
    recall_scores = []
    cold_ndcg_scores = []

    total_users = len(test_df)
    for i, row in enumerate(test_df.itertuples(index=False)):
        user_id = row.user_id
        test_item = row.item_id

        seed_item = last_train_item.get(user_id)
        if seed_item is None:
            # User has no training items — use test item itself as seed (cold-start)
            seed_item = test_item

        try:
            recs = recommender.recommend_by_id(int(seed_item), n=50)
        except Exception as exc:
            logger.warning("recommend_by_id failed for seed=%s user=%s: %s", seed_item, user_id, exc)
            ndcg_scores.append(0.0)
            recall_scores.append(0.0)
            if user_id in cold_start_users:
                cold_ndcg_scores.append(0.0)
            continue

        rec_ids = [r.get("id") for r in recs if r.get("id") is not None]

        # NDCG@10
        ndcg = 0.0
        if test_item in rec_ids[:10]:
            rank = rec_ids[:10].index(test_item)  # 0-indexed
            ndcg = 1.0 / log2(rank + 2)
        ndcg_scores.append(ndcg)

        # Recall@50
        recall = 1.0 if test_item in rec_ids[:50] else 0.0
        recall_scores.append(recall)

        # Cold-start NDCG@10
        if user_id in cold_start_users:
            cold_ndcg_scores.append(ndcg)

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d users evaluated.", i + 1, total_users)

    ndcg_at_10 = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    recall_at_50 = float(np.mean(recall_scores)) if recall_scores else 0.0
    cold_start_ndcg = float(np.mean(cold_ndcg_scores)) if cold_ndcg_scores else None

    logger.info(
        "Metrics: NDCG@10=%.4f, Recall@50=%.4f, Cold-Start NDCG@10=%s",
        ndcg_at_10,
        recall_at_50,
        f"{cold_start_ndcg:.4f}" if cold_start_ndcg is not None else "N/A",
    )

    # Task 1.4 — ILD
    ild = compute_ild(test_df, recommender)

    # Task 1.5 — Build report
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "num_users": total_users,
        "ndcg_at_10": ndcg_at_10,
        "recall_at_50": recall_at_50,
        "ild": ild,
        "cold_start_ndcg_at_10": cold_start_ndcg,
        "evaluation_protocol": "leave_one_out",
        "model_version": os.getenv("APP_VERSION", "2.0.0"),
    }

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Offline eval report written to '%s'.", output_path)

    # Update whitepaper placeholders
    _update_whitepaper(report)

    return report


# ---------------------------------------------------------------------------
# Task 1.4 — ILD computation
# ---------------------------------------------------------------------------


def compute_ild(
    test_df: pd.DataFrame,
    recommender,
    embeddings_path: Path = Path("models/sbert_embeddings.npy"),
) -> float | None:
    """Compute Intra-List Diversity (ILD) as mean pairwise cosine distance of top-10 recs."""
    if not embeddings_path.exists():
        logger.warning("SBERT embeddings not found at '%s'; ILD will be null.", embeddings_path)
        return None

    try:
        from sklearn.metrics.pairwise import cosine_distances

        embeddings = np.load(str(embeddings_path))
    except Exception as exc:
        logger.warning("Failed to load SBERT embeddings: %s; ILD will be null.", exc)
        return None

    ild_scores = []
    for row in test_df.itertuples(index=False):
        try:
            recs = recommender.recommend_by_id(int(row.item_id), n=10)
            rec_ids = [r.get("id") for r in recs if r.get("id") is not None]
            # Map movie IDs to embedding indices (use modulo for safety)
            indices = [rid % len(embeddings) for rid in rec_ids[:10]]
            if len(indices) < 2:
                continue
            embs = embeddings[indices]
            dists = cosine_distances(embs)
            # Mean of upper triangle (pairwise distances, no self-distance)
            n = len(indices)
            upper = [dists[i][j] for i in range(n) for j in range(i + 1, n)]
            ild_scores.append(float(np.mean(upper)))
        except Exception:
            continue

    return float(np.mean(ild_scores)) if ild_scores else None


# ---------------------------------------------------------------------------
# Task 1.5 — Whitepaper update helper
# ---------------------------------------------------------------------------


def _update_whitepaper(report: dict) -> None:
    """Replace Section 6.1 placeholders in APEX_WHITEPAPER.md with computed metrics."""
    whitepaper_path = Path("docs/APEX_WHITEPAPER.md")
    if not whitepaper_path.exists():
        logger.warning("Whitepaper not found at '%s'; skipping update.", whitepaper_path)
        return
    try:
        content = whitepaper_path.read_text(encoding="utf-8")
        replacements = {
            "NDCG@10": report.get("ndcg_at_10"),
            "Recall@50": report.get("recall_at_50"),
            "Diversity (ILD)": report.get("ild"),
            "Cold-Start NDCG@10": report.get("cold_start_ndcg_at_10"),
        }
        placeholder = r"Requires local execution — run scripts/run_offline_evaluation\.py"
        for metric, value in replacements.items():
            if value is not None:
                formatted = f"{value:.3f}"
                # Replace the placeholder in the row that contains this metric name
                content = re.sub(
                    rf"(\| {re.escape(metric)} \|[^|]*\|[^|]*\|) {placeholder} \|",
                    rf"\1 {formatted} |",
                    content,
                )
        whitepaper_path.write_text(content, encoding="utf-8")
        logger.info("Whitepaper Section 6.1 updated with computed metrics.")
    except Exception as exc:
        logger.warning("Failed to update whitepaper: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline leave-one-out evaluation against MovieLens 100K.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/offline_eval_report.json"),
        help="Path to write the evaluation report JSON.",
    )
    args = parser.parse_args()

    # Load data
    df = load_movielens_100k()

    # Split
    train_df, test_df, cold_start_users = leave_one_out_split(df)

    # Load recommender
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from backend.recommender import get_recommender

        rec = get_recommender()
        logger.info("Recommender loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load recommender: %s", exc)
        rec = None

    # Compute metrics and write report
    report = compute_metrics(
        train_df=train_df,
        test_df=test_df,
        cold_start_users=cold_start_users,
        output_path=args.output,
        recommender=rec,
    )

    if report:
        logger.info("Offline evaluation complete. Report: %s", args.output)
    else:
        logger.warning("Offline evaluation produced no report.")


if __name__ == "__main__":
    main()
