"""
Two-Tower Model Training Script

Trains the Two-Tower candidate generation model on real MovieLens data,
using ALS embeddings from the Gold layer as input features.

Data flow:
  1. Load Gold ALS user/item embeddings (from PySpark Medallion pipeline)
  2. Load ratings + movie metadata
  3. Build (user, positive_item, negative_items) training triplets
  4. Train with InfoNCE contrastive loss
  5. Export trained item embeddings to FAISS index

Usage:
    python scripts/train_two_tower.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from pathlib import Path
import time

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from backend.models.two_tower import TwoTowerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = PROJECT_ROOT / "data" / "datalake" / "gold"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# Dataset
# ============================================================


class TwoTowerDataset(Dataset):
    """
    Builds (user_features, positive_item_features, negative_item_features) triplets.

    User features: ALS embedding (16d) + [total_ratings, avg_rating] (2d) = 18d
    Item features: ALS embedding (16d) + [vote_avg, log_vote_count, log_popularity, num_genres] (4d) = 20d
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        user_features: dict[int, np.ndarray],
        item_features: dict[int, np.ndarray],
        num_negatives: int = 10,
    ):
        self.num_negatives = num_negatives
        self.user_features = user_features
        self.item_features = item_features

        # Filter to only users and items that have features
        valid_users = set(user_features.keys())
        valid_items = set(item_features.keys())
        filtered = ratings_df[ratings_df["userId"].isin(valid_users) & ratings_df["movieId"].isin(valid_items)]

        # Positive interactions: rating >= 3.5
        self.positives = filtered[filtered["rating"] >= 3.5][["userId", "movieId"]].values
        self.all_item_ids = list(valid_items)

        # Build per-user positive item sets (for hard negative sampling)
        self.user_positives: dict[int, set[int]] = {}
        for uid, mid in self.positives:
            self.user_positives.setdefault(uid, set()).add(mid)

        logger.info(
            f"  Dataset: {len(self.positives):,} positive pairs, {len(valid_users)} users, {len(valid_items)} items"
        )

    def __len__(self) -> int:
        return len(self.positives)

    def __getitem__(self, idx: int):
        user_id, pos_item_id = self.positives[idx]

        # User features
        user_feat = self.user_features[user_id]

        # Positive item features
        pos_feat = self.item_features[pos_item_id]

        # Sample negatives (items the user has NOT interacted with)
        user_pos = self.user_positives.get(user_id, set())
        neg_feats = []
        attempts = 0
        while len(neg_feats) < self.num_negatives and attempts < self.num_negatives * 5:
            neg_id = self.all_item_ids[np.random.randint(len(self.all_item_ids))]
            if neg_id not in user_pos:
                neg_feats.append(self.item_features[neg_id])
            attempts += 1

        # Pad if not enough negatives
        while len(neg_feats) < self.num_negatives:
            neg_feats.append(np.zeros(len(pos_feat), dtype=np.float32))

        return (
            torch.tensor(user_feat, dtype=torch.float32),
            torch.tensor(pos_feat, dtype=torch.float32),
            torch.tensor(np.array(neg_feats), dtype=torch.float32),
        )


# ============================================================
# Feature Engineering
# ============================================================


def build_user_features(
    user_emb_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """Build 18d user feature vectors: ALS embedding (16d) + activity stats (2d)."""
    logger.info("Building user features...")

    # Aggregate rating stats per user
    user_stats = (
        ratings_df.groupby("userId")
        .agg(
            total_ratings=("rating", "count"),
            avg_rating=("rating", "mean"),
        )
        .reset_index()
    )

    # Normalize stats
    user_stats["total_ratings"] = np.log1p(user_stats["total_ratings"])
    max_total = user_stats["total_ratings"].max()
    if max_total > 0:
        user_stats["total_ratings"] /= max_total
    user_stats["avg_rating"] /= 5.0  # Normalize to 0-1

    features = {}
    for _, row in user_emb_df.iterrows():
        uid = int(row["id"])
        als_emb = np.array(row["features"], dtype=np.float32)

        stats = user_stats[user_stats["userId"] == uid]
        if len(stats) > 0:
            activity = np.array(
                [
                    stats.iloc[0]["total_ratings"],
                    stats.iloc[0]["avg_rating"],
                ],
                dtype=np.float32,
            )
        else:
            activity = np.zeros(2, dtype=np.float32)

        features[uid] = np.concatenate([als_emb, activity])

    logger.info(f"  Built features for {len(features)} users (dim={18})")
    return features


def build_item_features(
    item_emb_df: pd.DataFrame,
    movies_df: pd.DataFrame,
) -> dict[int, np.ndarray]:
    """Build 20d item feature vectors: ALS embedding (16d) + metadata (4d)."""
    logger.info("Building item features...")

    features = {}
    for _, row in item_emb_df.iterrows():
        mid = int(row["id"])
        als_emb = np.array(row["features"], dtype=np.float32)

        movie = movies_df[movies_df["id"] == mid]
        if len(movie) > 0:
            m = movie.iloc[0]
            vote_avg = float(m.get("vote_average", 0)) / 10.0
            vote_count = np.log1p(float(m.get("vote_count", 0))) / 15.0
            popularity = np.log1p(float(m.get("popularity", 0))) / 10.0
            genres_str = str(m.get("genres", ""))
            num_genres = len(genres_str.split(",")) / 10.0 if genres_str else 0.0
            metadata = np.array([vote_avg, vote_count, popularity, num_genres], dtype=np.float32)
        else:
            metadata = np.zeros(4, dtype=np.float32)

        features[mid] = np.concatenate([als_emb, metadata])

    logger.info(f"  Built features for {len(features)} items (dim={20})")
    return features


# ============================================================
# Training Loop
# ============================================================


def train(
    model: TwoTowerModel,
    dataloader: DataLoader,
    num_epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Train the Two-Tower model with InfoNCE loss."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    losses = []

    for epoch in range(num_epochs):
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

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return losses


# ============================================================
# FAISS Index Export
# ============================================================


def export_to_faiss(
    model: TwoTowerModel,
    item_features: dict[int, np.ndarray],
    output_path: Path,
    id_map_path: Path,
) -> None:
    """Encode all items and build a FAISS index for ANN retrieval."""
    logger.info("Exporting item embeddings to FAISS index...")

    item_ids = sorted(item_features.keys())
    item_feats = np.array([item_features[iid] for iid in item_ids], dtype=np.float32)

    # Encode through item tower
    model.eval()
    with torch.no_grad():
        item_tensor = torch.tensor(item_feats, dtype=torch.float32)
        # Process in batches to avoid OOM
        batch_size = 1024
        embeddings = []
        for i in range(0, len(item_tensor), batch_size):
            batch = item_tensor[i : i + batch_size]
            emb = model.item_tower(batch).numpy()
            embeddings.append(emb)
        all_embeddings = np.vstack(embeddings).astype(np.float32)

    # Build FAISS index (Inner Product = cosine similarity since vectors are L2-normalized)
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)

    faiss.write_index(index, str(output_path))

    # Save ID mapping
    np.save(str(id_map_path), np.array(item_ids))

    logger.info(f"  FAISS index: {index.ntotal} items, {dim}d → {output_path}")
    logger.info(f"  ID map: {id_map_path}")


# ============================================================
# Main
# ============================================================


def main():
    logger.info("=" * 60)
    logger.info("PHASE 3: Training Two-Tower Candidate Generation Model")
    logger.info("=" * 60)

    start_time = time.time()

    # 1. Load Gold ALS embeddings
    logger.info("Loading Gold ALS embeddings...")
    user_emb_df = pd.read_parquet(GOLD_DIR / "model_user_embeddings")
    item_emb_df = pd.read_parquet(GOLD_DIR / "model_item_embeddings")
    logger.info(f"  Users: {len(user_emb_df)}, Items: {len(item_emb_df)}")

    # 2. Load ratings + movies
    logger.info("Loading ratings and movie metadata...")
    ratings_df = pd.read_parquet(DATA_DIR / "ratings_transformed.parquet")
    movies_df = pd.read_parquet(DATA_DIR / "movies_transformed.parquet")
    logger.info(f"  Ratings: {len(ratings_df):,}, Movies: {len(movies_df):,}")

    # 3. Build features
    user_features = build_user_features(user_emb_df, ratings_df)
    item_features = build_item_features(item_emb_df, movies_df)

    # 4. Create dataset and dataloader
    dataset = TwoTowerDataset(
        ratings_df=ratings_df,
        user_features=user_features,
        item_features=item_features,
        num_negatives=10,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # 5. Initialize model
    model = TwoTowerModel(
        user_input_dim=18,
        item_input_dim=20,
        embedding_dim=128,
        temperature=0.07,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {total_params:,}")

    # 6. Train
    logger.info("Training Two-Tower model...")
    losses = train(model, dataloader, num_epochs=30, lr=1e-3)

    # 7. Save model weights
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "two_tower.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"  Model saved to {model_path}")

    # 8. Export to FAISS
    faiss_path = MODELS_DIR / "two_tower_faiss.index"
    id_map_path = MODELS_DIR / "two_tower_item_ids.npy"
    export_to_faiss(model, item_features, faiss_path, id_map_path)

    # 9. Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PHASE 3 COMPLETE — Two-Tower Model Trained")
    logger.info(f"  Final loss: {losses[-1]:.4f}")
    logger.info(f"  Initial loss: {losses[0]:.4f}")
    logger.info(f"  Loss reduction: {((losses[0] - losses[-1]) / losses[0]) * 100:.1f}%")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  FAISS index: {faiss_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
