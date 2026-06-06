"""
Two-Tower Model Fine-Tuning Script

Fine-tunes the Two-Tower candidate generation model on live interaction data
from the Event Store, using InfoNCE contrastive loss.

Data flow:
  1. Read positive interaction pairs from the Event Store
     (rating >= 3.5 or click events)
  2. Guard: exit with WARNING if fewer than 100 positive pairs
  3. Build user/item feature vectors (ALS embeddings if available, else simple proxies)
  4. Construct hard negatives (4 per positive)
  5. Load base two_tower.pth weights as starting point
  6. Train for --epochs with InfoNCE loss; check for NaN after each epoch
  7. Evaluate Hit_Rate@10 on held-out 20% validation set
  8. Save fine-tuned weights to models/two_tower_finetuned.pth

Usage:
    python scripts/finetune_two_tower.py [--epochs N] [--lr LR] [--negatives K]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.events import iter_events
from backend.models.two_tower import TwoTowerModel

# Re-use dataset and feature helpers from the base training script
from scripts.train_two_tower import (
    TwoTowerDataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = PROJECT_ROOT / "data" / "datalake" / "gold"
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================
# Task 8.1 — Positive pair extraction
# ============================================================


def extract_positive_pairs() -> list[tuple[str, int]]:
    """
    Read events from the Event Store and return (user_id, movie_id) positive pairs.

    Positive pair criteria:
      - event_type == "rating" with rating >= 3.5, OR
      - event_type == "click"

    Both user_id and movie_id must be non-None.
    """
    pairs: list[tuple[str, int]] = []

    for event in iter_events():
        event_type = str(event.get("event_type", "")).lower()

        if event_type == "rating":
            rating = event.get("rating")
            if rating is None:
                continue
            try:
                if float(rating) < 3.5:
                    continue
            except (TypeError, ValueError):
                continue
        elif event_type == "click":
            pass  # always include clicks
        else:
            continue  # exclude all other event types

        user_id = event.get("user_id")
        movie_id = event.get("movie_id")

        if user_id is None or movie_id is None:
            continue

        try:
            movie_id_int = int(movie_id)
        except (TypeError, ValueError):
            continue

        pairs.append((str(user_id), movie_id_int))

    return pairs


# ============================================================
# Task 8.2 — Feature construction for live events
# ============================================================


def _load_als_embeddings() -> tuple[dict[str, np.ndarray] | None, dict[int, np.ndarray] | None]:
    """
    Attempt to load Gold ALS user and item embeddings.
    Returns (user_emb_dict, item_emb_dict) or (None, None) if unavailable.
    """
    user_emb_path = GOLD_DIR / "model_user_embeddings"
    item_emb_path = GOLD_DIR / "model_item_embeddings"

    if not user_emb_path.exists() or not item_emb_path.exists():
        return None, None

    try:
        import pandas as pd

        user_emb_df = pd.read_parquet(user_emb_path)
        item_emb_df = pd.read_parquet(item_emb_path)

        user_als: dict[str, np.ndarray] = {}
        for _, row in user_emb_df.iterrows():
            uid = str(row["id"])
            user_als[uid] = np.array(row["features"], dtype=np.float32)

        item_als: dict[int, np.ndarray] = {}
        for _, row in item_emb_df.iterrows():
            mid = int(row["id"])
            item_als[mid] = np.array(row["features"], dtype=np.float32)

        logger.info(
            "Loaded Gold ALS embeddings: %d users, %d items",
            len(user_als),
            len(item_als),
        )
        return user_als, item_als

    except Exception as exc:
        logger.warning("Could not load Gold ALS embeddings: %s — using simple proxies", exc)
        return None, None


def build_live_user_features(
    pairs: list[tuple[str, int]],
    als_user_embs: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """
    Build 18-dimensional user feature vectors from live event data.

    If Gold ALS embeddings are available, use them (16d ALS + 2d stats = 18d).
    Otherwise, construct simple proxy features:
      [log1p(num_interactions)/log1p(100), avg_rating/5.0] + zeros to pad to 18d
    """
    # Aggregate per-user stats from the pairs list
    user_interaction_counts: dict[str, int] = {}
    user_rating_sums: dict[str, float] = {}
    user_rating_counts: dict[str, int] = {}

    for event in iter_events():
        event_type = str(event.get("event_type", "")).lower()
        user_id = event.get("user_id")
        if user_id is None:
            continue
        uid = str(user_id)

        user_interaction_counts[uid] = user_interaction_counts.get(uid, 0) + 1

        if event_type == "rating":
            rating = event.get("rating")
            if rating is not None:
                try:
                    user_rating_sums[uid] = user_rating_sums.get(uid, 0.0) + float(rating)
                    user_rating_counts[uid] = user_rating_counts.get(uid, 0) + 1
                except (TypeError, ValueError):
                    pass

    features: dict[str, np.ndarray] = {}
    unique_users = {uid for uid, _ in pairs}

    for uid in unique_users:
        if als_user_embs is not None and uid in als_user_embs:
            # Use Gold ALS embedding (cap to 16d regardless of GPU training dim) + activity stats (2d)
            als_emb = als_user_embs[uid][:16]  # cap to 16d
            num_interactions = user_interaction_counts.get(uid, 0)
            avg_rating = (
                user_rating_sums.get(uid, 0.0) / user_rating_counts[uid] if user_rating_counts.get(uid, 0) > 0 else 3.0
            )
            activity = np.array(
                [
                    math.log1p(num_interactions) / math.log1p(100),
                    avg_rating / 5.0,
                ],
                dtype=np.float32,
            )
            features[uid] = np.concatenate([als_emb, activity])
        else:
            # Simple proxy: [log1p(num_interactions)/log1p(100), avg_rating/5.0] + zeros to 18d
            num_interactions = user_interaction_counts.get(uid, 0)
            avg_rating = (
                user_rating_sums.get(uid, 0.0) / user_rating_counts[uid] if user_rating_counts.get(uid, 0) > 0 else 3.0
            )
            proxy = np.array(
                [
                    math.log1p(num_interactions) / math.log1p(100),
                    avg_rating / 5.0,
                ],
                dtype=np.float32,
            )
            # Pad to 18d with zeros
            features[uid] = np.concatenate([proxy, np.zeros(16, dtype=np.float32)])

    return features


def build_live_item_features(
    pairs: list[tuple[str, int]],
    als_item_embs: dict[int, np.ndarray] | None,
) -> dict[int, np.ndarray]:
    """
    Build 20-dimensional item feature vectors for live event items.

    If Gold ALS embeddings are available, use them (16d ALS + 4d metadata = 20d).
    Otherwise, use zeros padded to 20d (we don't have ALS embeddings for live items).
    """
    features: dict[int, np.ndarray] = {}
    unique_items = {mid for _, mid in pairs}

    for mid in unique_items:
        if als_item_embs is not None and mid in als_item_embs:
            als_emb = als_item_embs[mid][:16]  # cap to 16d regardless of GPU training dim
            # No live metadata available; pad with zeros for the 4 metadata dims
            features[mid] = np.concatenate([als_emb, np.zeros(4, dtype=np.float32)])
        else:
            # Zeros padded to 20d
            features[mid] = np.zeros(20, dtype=np.float32)

    return features


# ============================================================
# Task 8.3 — Training loop and output
# ============================================================


def compute_hit_rate_at_k(
    model: TwoTowerModel,
    val_pairs: list[tuple[str, int]],
    user_features: dict[str, np.ndarray],
    item_features: dict[int, np.ndarray],
    k: int = 10,
    device: str = "cpu",
) -> float:
    """
    Compute Hit_Rate@K on the validation set.

    For each validation user, encode their features through the user tower,
    find top-K nearest items from the validation positive items pool,
    check if the ground truth item is in top-K.
    """
    if not val_pairs:
        return 0.0

    model.eval()

    # Collect all unique items in the validation pool
    val_item_ids = list({mid for _, mid in val_pairs})
    if not val_item_ids:
        return 0.0

    # Encode all validation items
    val_item_feats = np.array([item_features[mid] for mid in val_item_ids], dtype=np.float32)
    with torch.no_grad():
        item_tensor = torch.tensor(val_item_feats, dtype=torch.float32).to(device)
        item_embs = model.item_tower(item_tensor)  # [N_items, D]

    hits = 0
    total = 0

    for user_id, gt_item_id in val_pairs:
        if user_id not in user_features or gt_item_id not in item_features:
            continue

        user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            user_emb = model.user_tower(user_feat)  # [1, D]

        # Cosine similarity scores (both towers output L2-normalised embeddings)
        scores = (item_embs * user_emb).sum(dim=-1)  # [N_items]
        top_k_indices = torch.topk(scores, min(k, len(val_item_ids))).indices.cpu().tolist()
        top_k_item_ids = {val_item_ids[i] for i in top_k_indices}

        if gt_item_id in top_k_item_ids:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0


def finetune(
    epochs: int = 5,
    lr: float = 1e-4,
    num_negatives: int = 4,
) -> None:
    """Main fine-tuning routine."""

    # ── Task 8.1: Extract positive pairs ──────────────────────────────────────
    logger.info("Extracting positive interaction pairs from Event Store...")
    pairs = extract_positive_pairs()
    logger.info("Found %d positive pairs", len(pairs))

    if len(pairs) < 100:
        logger.warning(
            "Fewer than 100 positive pairs found (%d). Skipping fine-tuning — no model file written.",
            len(pairs),
        )
        sys.exit(0)

    # ── Task 8.2: Build features ───────────────────────────────────────────────
    logger.info("Loading ALS embeddings (if available)...")
    als_user_embs, als_item_embs = _load_als_embeddings()

    logger.info("Building user and item feature vectors...")
    user_features = build_live_user_features(pairs, als_user_embs)
    item_features = build_live_item_features(pairs, als_item_embs)

    # Filter pairs to those with features on both sides
    valid_pairs = [(uid, mid) for uid, mid in pairs if uid in user_features and mid in item_features]
    if len(valid_pairs) < 100:
        logger.warning(
            "After feature filtering, fewer than 100 valid pairs remain (%d). "
            "Skipping fine-tuning — no model file written.",
            len(valid_pairs),
        )
        sys.exit(0)

    # ── Task 8.3: Train/val split ──────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(valid_pairs))
    split = int(len(valid_pairs) * 0.8)
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_pairs = [valid_pairs[i] for i in train_indices]
    val_pairs = [valid_pairs[i] for i in val_indices]

    logger.info(
        "Split: %d train pairs, %d validation pairs",
        len(train_pairs),
        len(val_pairs),
    )

    # Build a DataFrame-like structure for TwoTowerDataset
    # TwoTowerDataset expects a DataFrame with columns userId, movieId, rating
    import pandas as pd

    train_df = pd.DataFrame(
        [(uid, mid, 4.0) for uid, mid in train_pairs],
        columns=["userId", "movieId", "rating"],
    )
    # Convert userId to int if possible (TwoTowerDataset uses int keys)
    # We use string user IDs from live events; build a mapping
    unique_user_ids = list({uid for uid, _ in valid_pairs})
    user_id_to_int: dict[str, int] = {uid: i for i, uid in enumerate(unique_user_ids)}
    {i: uid for uid, i in user_id_to_int.items()}

    # Remap user features to int keys for TwoTowerDataset compatibility
    user_features_int: dict[int, np.ndarray] = {user_id_to_int[uid]: feat for uid, feat in user_features.items()}

    train_df["userId"] = train_df["userId"].map(user_id_to_int)

    dataset = TwoTowerDataset(
        ratings_df=train_df,
        user_features=user_features_int,
        item_features=item_features,
        num_negatives=num_negatives,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=min(256, len(dataset)),
        shuffle=True,
        num_workers=0,
    )

    # ── Task 8.3: Initialize model ─────────────────────────────────────────────
    model = TwoTowerModel(
        user_input_dim=18,
        item_input_dim=20,
        embedding_dim=128,
        temperature=0.07,
    )

    base_weights_path = MODELS_DIR / "two_tower.pth"
    if base_weights_path.exists():
        try:
            state_dict = torch.load(base_weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            logger.info("Loaded base weights from %s", base_weights_path)
        except Exception as exc:
            logger.warning(
                "Could not load base weights from %s: %s — starting from scratch",
                base_weights_path,
                exc,
            )
    else:
        logger.info("No base weights found at %s — starting from scratch", base_weights_path)

    device = "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ── Task 8.3: Training loop ────────────────────────────────────────────────
    logger.info("Starting fine-tuning for %d epochs (lr=%.2e, negatives=%d)...", epochs, lr, num_negatives)

    final_loss = float("nan")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for user_feat, pos_feat, neg_feats in dataloader:
            user_feat = user_feat.to(device)
            pos_feat = pos_feat.to(device)
            neg_feats = neg_feats.to(device)

            loss = model.compute_contrastive_loss(user_feat, pos_feat, neg_feats)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # Check for NaN loss
        if math.isnan(avg_loss):
            logger.error(
                "NaN loss detected at epoch %d/%d. Aborting fine-tuning — no model file written.",
                epoch + 1,
                epochs,
            )
            sys.exit(1)

        final_loss = avg_loss
        logger.info("Epoch %d/%d | Loss: %.4f", epoch + 1, epochs, avg_loss)

    # ── Task 8.3: Validation Hit_Rate@10 ──────────────────────────────────────
    # Remap val_pairs back to int user IDs for evaluation
    val_pairs_int = [(user_id_to_int[uid], mid) for uid, mid in val_pairs if uid in user_id_to_int]

    hit_rate = compute_hit_rate_at_k(
        model=model,
        val_pairs=val_pairs_int,
        user_features=user_features_int,
        item_features=item_features,
        k=10,
        device=device,
    )

    # ── Task 8.3: Save fine-tuned model ───────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "two_tower_finetuned.pth"
    try:
        torch.save(model.state_dict(), output_path)
        logger.info("Fine-tuned model saved to %s", output_path)
    except Exception as exc:
        logger.error("Failed to save fine-tuned model to %s: %s", output_path, exc)
        sys.exit(1)

    # ── Task 8.3: Print results to stdout ─────────────────────────────────────
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Validation Hit_Rate@10: {hit_rate:.4f}")


# ============================================================
# CLI entry point
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the Two-Tower model on live interaction data from the Event Store."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--negatives",
        type=int,
        default=4,
        help="Number of hard negatives per positive pair (default: 4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune(
        epochs=args.epochs,
        lr=args.lr,
        num_negatives=args.negatives,
    )
