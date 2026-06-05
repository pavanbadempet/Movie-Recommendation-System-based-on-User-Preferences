"""
Train the Multi-gate Mixture-of-Experts (MMoE) Ranker.

Synthesizes a multi-task objective dataset from the MovieLens ratings and trains
the MMoE architecture to balance Click-Through Rate, Watch Time, and Satisfaction.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from backend.mmoe_ranker import MMoERanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


class MultiTaskDataset(Dataset):
    def __init__(self, df, num_items):
        self.users = torch.tensor(df["userId"].values, dtype=torch.long)
        self.items = torch.tensor(df["movieId"].values, dtype=torch.long)

        # Target 1: Click (1.0 for all observed interactions)
        self.click = torch.ones(len(df), dtype=torch.float32)

        # Target 2: Watch Time (Synthesized based on rating)
        # Ratings 1-5 scaled to 0-1 range + noise
        ratings = df["rating"].values
        watch_time = (ratings / 5.0) + np.random.normal(0, 0.1, len(ratings))
        watch_time = np.clip(watch_time, 0.0, 1.0)
        self.watch = torch.tensor(watch_time, dtype=torch.float32)

        # Target 3: Satisfaction (Binary: 1 if rating >= 4.0 else 0)
        satisfaction = (ratings >= 4.0).astype(float)
        self.sat = torch.tensor(satisfaction, dtype=torch.float32)

        # Position bias simulation
        # Random positions 0-499 to simulate search result ranks during training
        self.positions = torch.randint(0, 500, (len(df),))

        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # We also generate one negative sample per positive sample on the fly
        if torch.rand(1).item() < 0.5:
            return (
                self.users[idx],
                self.items[idx],
                self.positions[idx],
                self.click[idx],
                self.watch[idx],
                self.sat[idx],
            )
        else:
            # Negative sample
            neg_item = torch.randint(0, self.num_items, (1,)).item()
            return (
                self.users[idx],
                torch.tensor(neg_item, dtype=torch.long),
                self.positions[idx],
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
            )


def main():
    logger.info("============================================================")
    logger.info("PHASE 5: Training MMoE Multi-Task Ranker")
    logger.info("============================================================")

    # 1. Load Data
    ratings_path = DATA_DIR / "ratings_transformed.parquet"
    if not ratings_path.exists():
        logger.error(f"Missing data: {ratings_path}. Run Phase 1 data pipeline.")
        return

    df = pd.read_parquet(ratings_path)
    num_users = df["userId"].max() + 1
    num_items = df["movieId"].max() + 1

    logger.info(f"Loaded {len(df)} interactions. Users: {num_users}, Items: {num_items}")

    # 2. Dataset & DataLoader
    # Train/Val split (80/20 based on time)
    df_sorted = df.sort_values("timestamp")
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    val_df = df_sorted.iloc[split_idx:]

    train_dataset = MultiTaskDataset(train_df, num_items)
    val_dataset = MultiTaskDataset(val_df, num_items)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # 3. Model
    model = MMoERanker(
        user_vocab_size=num_users,
        item_vocab_size=num_items,
        emb_dim=16,
        num_experts=4,
        expert_hidden_dim=64,
        expert_out_dim=32,
    )

    # Try to load ALS Gold embeddings for initialization
    gold_dir = Path("data/datalake/gold")
    user_emb_path = gold_dir / "model_user_embeddings"
    item_emb_path = gold_dir / "model_item_embeddings"
    if user_emb_path.exists() and item_emb_path.exists():
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()
            u_df = spark.read.parquet(str(user_emb_path)).orderBy("id").toPandas()
            i_df = spark.read.parquet(str(item_emb_path)).orderBy("id").toPandas()
            u_mat = np.stack(u_df["features"].values)
            i_mat = np.stack(i_df["features"].values)

            # Pad or truncate to match dimensions
            if len(u_mat) >= num_users:
                u_mat = u_mat[:num_users]
            else:
                u_mat = np.pad(u_mat, ((0, num_users - len(u_mat)), (0, 0)))

            if len(i_mat) >= num_items:
                i_mat = i_mat[:num_items]
            else:
                i_mat = np.pad(i_mat, ((0, num_items - len(i_mat)), (0, 0)))

            model.user_emb.weight.data.copy_(torch.from_numpy(u_mat).float())
            model.item_emb.weight.data.copy_(torch.from_numpy(i_mat).float())
            logger.info("✅ Injected Gold ALS priors into MMoE ranker embeddings.")
        except Exception as e:
            logger.warning(f"Failed to inject PySpark priors: {e}")

    # 4. Losses & Optimizer
    # We use BCE for Click/Sat and MSE for Watch Time
    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Training on device: {device}")

    # 5. Training Loop
    epochs = 10
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_ctr, total_watch, total_sat = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for u, i, pos, t_ctr, t_watch, t_sat in pbar:
            u, i, pos = u.to(device), i.to(device), pos.to(device)
            t_ctr, t_watch, t_sat = t_ctr.to(device), t_watch.to(device), t_sat.to(device)

            optimizer.zero_grad()

            p_ctr, p_watch, p_sat = model(u, i, pos)

            loss_ctr = criterion_bce(p_ctr, t_ctr)
            loss_watch = criterion_mse(p_watch, t_watch)
            loss_sat = criterion_bce(p_sat, t_sat)

            # Weighted multi-task loss (In production, these weights are learned via Uncertainty Weighting)
            loss = loss_ctr * 1.0 + loss_watch * 0.5 + loss_sat * 1.0

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ctr += loss_ctr.item()
            total_watch += loss_watch.item()
            total_sat += loss_sat.item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for u, i, pos, t_ctr, t_watch, t_sat in val_loader:
                u, i, pos = u.to(device), i.to(device), pos.to(device)
                t_ctr, t_watch, t_sat = t_ctr.to(device), t_watch.to(device), t_sat.to(device)

                p_ctr, p_watch, p_sat = model(u, i, pos)

                l_ctr = criterion_bce(p_ctr, t_ctr)
                l_watch = criterion_mse(p_watch, t_watch)
                l_sat = criterion_bce(p_sat, t_sat)

                val_loss += (l_ctr * 1.0 + l_watch * 0.5 + l_sat * 1.0).item()

        val_loss /= len(val_loader)

        logger.info(
            f"Epoch {epoch:2d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | CTR: {total_ctr / len(train_loader):.4f} | Watch: {total_watch / len(train_loader):.4f} | Sat: {total_sat / len(train_loader):.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODELS_DIR / "mmoe_ranker.pth")

    logger.info("✅ Training Complete. Best model saved to models/mmoe_ranker.pth")


if __name__ == "__main__":
    main()
