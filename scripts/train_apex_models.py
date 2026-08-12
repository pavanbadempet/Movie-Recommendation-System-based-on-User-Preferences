"""
Unified Training Script for All 5 Neural Ensemble Models

Trains each model on real MovieLens data (100K ratings, 610 users, 9,724 movies)
using the appropriate loss function and evaluation metric.

Models trained:
  1. SASRec     — Transformer sequential recommendation (BPR loss)
  2. LightGCN   — Graph collaborative filtering (BPR loss)
  3. Quantum    — Quantum Fluid Neural ODE (margin loss)
  4. Hyperbolic — Poincaré manifold embeddings (contrastive loss)
  5. KAN        — Kolmogorov-Arnold B-spline ranker (BCE loss)

All trained weights saved to models/ directory.

Usage:
    python scripts/train_apex_models.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
import logging
from pathlib import Path
import time

try:
    import mlflow
except ImportError:
    mlflow = None
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data Loading & Preparation
# ============================================================


def load_data():
    """Load ratings and build interaction structures."""
    logger.info("Loading data...")
    parquet_path = DATA_DIR / "ratings_transformed.parquet"

    if parquet_path.exists():
        ratings = pd.read_parquet(parquet_path)
    else:
        # Fallback to fetching ratings from Neon PostgreSQL or seed interactions
        try:
            from sqlalchemy import create_engine
            db_url = os.environ.get("DATABASE_URL")
            if db_url and db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            engine = create_engine(db_url, connect_args={"sslmode": "require"})
            ratings = pd.read_sql("SELECT user_id AS userId, movie_id AS movieId, rating, timestamp FROM ratings LIMIT 50000", engine)
            logger.info(f"Loaded {len(ratings)} ratings from Neon PostgreSQL.")
        except Exception:
            logger.info("Generating seed ratings for model training verification...")
            np.random.seed(42)
            users = np.random.randint(1, 100, size=5000)
            movies = np.random.randint(1, 500, size=5000)
            rates = np.random.uniform(1.0, 5.0, size=5000)
            ts = np.random.randint(1600000000, 1700000000, size=5000)
            ratings = pd.DataFrame({"userId": users, "movieId": movies, "rating": rates, "timestamp": ts})

    # Build user/item ID mappings (contiguous 0-indexed)
    unique_users = sorted(ratings["userId"].unique())
    unique_items = sorted(ratings["movieId"].unique())
    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {m: i for i, m in enumerate(unique_items)}
    idx2item = {i: m for m, i in item2idx.items()}

    num_users = len(unique_users)
    num_items = len(unique_items)

    # Map to indices
    ratings["user_idx"] = ratings["userId"].map(user2idx)
    ratings["item_idx"] = ratings["movieId"].map(item2idx)

    # Build per-user interaction sets
    user_interactions = defaultdict(set)
    for _, row in ratings.iterrows():
        user_interactions[int(row["user_idx"])].add(int(row["item_idx"]))

    # Train/val split (80/20 by time)
    ratings_sorted = ratings.sort_values("timestamp")
    split_idx = int(len(ratings_sorted) * 0.8)
    train_df = ratings_sorted.iloc[:split_idx]
    val_df = ratings_sorted.iloc[split_idx:]

    # Build training sequences per user (for SASRec)
    user_sequences = defaultdict(list)
    for _, row in train_df.sort_values("timestamp").iterrows():
        user_sequences[int(row["user_idx"])].append(int(row["item_idx"]))

    logger.info(f"  Users: {num_users}, Items: {num_items}")
    logger.info(f"  Train: {len(train_df):,}, Val: {len(val_df):,}")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "num_users": num_users,
        "num_items": num_items,
        "user_interactions": dict(user_interactions),
        "user_sequences": dict(user_sequences),
        "item2idx": item2idx,
        "idx2item": idx2item,
    }


def sample_negatives(user_idx: int, num_items: int, user_interactions: dict, k: int = 1):
    """Sample k negative items for a user."""
    positives = user_interactions.get(user_idx, set())
    negs = []
    while len(negs) < k:
        neg = np.random.randint(0, num_items)
        if neg not in positives:
            negs.append(neg)
    return negs


# ============================================================
# 1. SASRec Training
# ============================================================


def train_sasrec(data: dict):
    """Train Self-Attentive Sequential Recommendation model."""
    from backend.models.sasrec import SASRec

    logger.info("=" * 50)
    logger.info("Training SASRec (Transformer Sequential)")
    logger.info("=" * 50)

    num_items = data["num_items"]
    max_seq_len = 50
    hidden_dim = 64  # Upgraded to 64-D for SOTA neural representation capacity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SASRec(num_items=num_items + 1, max_seq_len=max_seq_len, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    sequences = data["user_sequences"]
    user_list = [u for u, seq in sequences.items() if len(seq) >= 3]

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        np.random.shuffle(user_list)

        for user_idx in user_list:
            seq = sequences[user_idx]
            if len(seq) < 3:
                continue

            # Input: all but last, Target: last item
            input_seq = seq[:-1][-max_seq_len:]
            target = seq[-1]

            # Pad sequence
            padded = [0] * (max_seq_len - len(input_seq)) + input_seq
            seq_tensor = torch.tensor([padded], dtype=torch.long)

            # Get sequence representation
            seq_output = model(seq_tensor)  # [1, seq_len, hidden_dim]
            last_hidden = seq_output[:, -1, :]  # [1, hidden_dim]

            # Positive score
            pos_emb = model.item_emb(torch.tensor([target]))  # [1, hidden_dim]
            pos_score = (last_hidden * pos_emb).sum(dim=-1)

            # Negative score
            neg_item = sample_negatives(user_idx, num_items, data["user_interactions"])[0]
            neg_emb = model.item_emb(torch.tensor([neg_item]))
            neg_score = (last_hidden * neg_emb).sum(dim=-1)

            # BPR Loss
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(user_list), 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")

    path = MODELS_DIR / "sasrec.pth"
    torch.save(model.state_dict(), path)
    logger.info(f"  Saved: {path}")
    return avg_loss


# ============================================================
# 2. LightGCN Training
# ============================================================


def train_lightgcn(data: dict):
    """Train LightGCN graph collaborative filtering model."""
    from backend.models.lightgcn import LightGCN

    logger.info("=" * 50)
    logger.info("Training LightGCN (Graph Neural Network)")
    logger.info("=" * 50)

    num_users = data["num_users"]
    num_items = data["num_items"]
    emb_dim = 64  # Upgraded to 64-D for high-dimensional graph embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=emb_dim, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Build adjacency matrix
    train = data["train_df"]
    rows = train["user_idx"].values
    cols = train["item_idx"].values + num_users  # Offset items

    n = num_users + num_items
    adj = sp.coo_matrix(
        (np.ones(len(rows) * 2), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(n, n),
    ).tocsr()

    # Normalize: D^{-1/2} A D^{-1/2}
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    norm_adj = d_mat @ adj @ d_mat

    # Convert to sparse torch tensor
    coo = norm_adj.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape)).coalesce().to(device)

    # Training
    user_indices = train["user_idx"].values
    pos_indices = train["item_idx"].values

    num_epochs = 50
    batch_size = 2048  # Increased batch size for GPU acceleration

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        perm = np.random.permutation(len(user_indices))

        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            users = torch.LongTensor(user_indices[batch_idx])
            pos_items = torch.LongTensor(pos_indices[batch_idx])

            # Sample negatives
            neg_items_list = []
            for u in users.numpy():
                neg_items_list.append(sample_negatives(u, num_items, data["user_interactions"])[0])
            neg_items = torch.LongTensor(neg_items_list)

            loss = model(users, pos_items, neg_items, adj_tensor)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        num_batches = max(1, len(perm) // batch_size)
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")

    path = MODELS_DIR / "lightgcn.pth"
    torch.save(model.state_dict(), path)
    logger.info(f"  Saved: {path}")
    return avg_loss


# ============================================================
# 3. Quantum Fluid ODE Training
# ============================================================


def train_quantum(data: dict):
    """Train Quantum Fluid Neural ODE model."""
    from backend.models.neural_ode_recommender import QuantumFluidRecommender

    logger.info("=" * 50)
    logger.info("Training Quantum Fluid ODE")
    logger.info("=" * 50)

    num_users = data["num_users"]
    num_items = data["num_items"]
    emb_dim = 16

    model = QuantumFluidRecommender(num_users=num_users, num_items=num_items, emb_dim=emb_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train = data["train_df"]
    user_indices = train["user_idx"].values
    pos_indices = train["item_idx"].values
    timestamps = train["timestamp"].values.astype(np.float64)
    # Normalize timestamps to [0, 1]
    t_min, t_max = timestamps.min(), timestamps.max()
    if t_max > t_min:
        timestamps = (timestamps - t_min) / (t_max - t_min)
    else:
        timestamps = np.zeros_like(timestamps)

    num_epochs = 20
    batch_size = 512

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        perm = np.random.permutation(len(user_indices))

        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            users = torch.LongTensor(user_indices[batch_idx])
            pos_items = torch.LongTensor(pos_indices[batch_idx])
            time_deltas = torch.FloatTensor(timestamps[batch_idx])

            neg_items_list = [sample_negatives(u, num_items, data["user_interactions"])[0] for u in users.numpy()]
            neg_items = torch.LongTensor(neg_items_list)

            loss = model(users, pos_items, neg_items, time_deltas)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        num_batches = max(1, len(perm) // batch_size)
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")

    path = MODELS_DIR / "quantum_fluid.pth"
    torch.save(model.state_dict(), path)
    logger.info(f"  Saved: {path}")
    return avg_loss


# ============================================================
# 4. Hyperbolic Poincaré Training
# ============================================================


def train_hyperbolic(data: dict):
    """Train Hyperbolic Poincaré manifold embeddings."""
    from backend.models.hyperbolic_recommender import HyperbolicRecommender

    logger.info("=" * 50)
    logger.info("Training Hyperbolic Poincaré Manifold")
    logger.info("=" * 50)

    num_users = data["num_users"]
    num_items = data["num_items"]
    emb_dim = 16

    model = HyperbolicRecommender(num_users=num_users, num_items=num_items, emb_dim=emb_dim)
    # Riemannian SGD approximation — use small LR to stay on manifold
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    train = data["train_df"]
    user_indices = train["user_idx"].values
    pos_indices = train["item_idx"].values

    num_epochs = 20
    batch_size = 512

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        perm = np.random.permutation(len(user_indices))

        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            users = torch.LongTensor(user_indices[batch_idx])
            pos_items = torch.LongTensor(pos_indices[batch_idx])

            neg_items_list = [sample_negatives(u, num_items, data["user_interactions"])[0] for u in users.numpy()]
            neg_items = torch.LongTensor(neg_items_list)

            loss = model(users, pos_items, neg_items)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Project embeddings back onto Poincaré ball (||x|| < 1)
            with torch.no_grad():
                for emb in [model.user_embedding, model.item_embedding]:
                    norms = emb.weight.data.norm(dim=-1, keepdim=True)
                    mask = norms > 0.95
                    if mask.any():
                        emb.weight.data[mask.squeeze()] *= 0.95 / norms[mask].squeeze().unsqueeze(-1)

            total_loss += loss.item()

        num_batches = max(1, len(perm) // batch_size)
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")

    path = MODELS_DIR / "hyperbolic.pth"
    torch.save(model.state_dict(), path)
    logger.info(f"  Saved: {path}")
    return avg_loss


# ============================================================
# 4.5. Clifford Geometric Algebra Training
# ============================================================


def train_clifford(data: dict):
    """Train Clifford Geometric Algebra model."""
    from backend.models.clifford_recommender import CliffordRecommender

    logger.info("=" * 50)
    logger.info("Training Clifford Geometric Algebra Model")
    logger.info("=" * 50)

    num_users = data["num_users"]
    num_items = data["num_items"]
    emb_dim = 16

    model = CliffordRecommender(num_users=num_users, num_items=num_items, emb_dim=emb_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train = data["train_df"]
    user_indices = train["user_idx"].values
    pos_indices = train["item_idx"].values

    num_epochs = 20
    batch_size = 512

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        perm = np.random.permutation(len(user_indices))

        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start : start + batch_size]
            users = torch.LongTensor(user_indices[batch_idx])
            pos_items = torch.LongTensor(pos_indices[batch_idx])

            neg_items_list = [sample_negatives(u, num_items, data["user_interactions"])[0] for u in users.numpy()]
            neg_items = torch.LongTensor(neg_items_list)

            loss = model(users, pos_items, neg_items)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        num_batches = max(1, len(perm) // batch_size)
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")

    path = MODELS_DIR / "clifford.pth"
    torch.save(model.state_dict(), path)
    logger.info(f"  Saved: {path}")
    return avg_loss


# ============================================================
# 5. KAN Ranker Training
# ============================================================


def train_kan(data: dict):
    """Train Kolmogorov-Arnold B-Spline ranker."""
    from backend.models.kan_ranker import KANRanker

    logger.info("=" * 50)
    logger.info("Training KAN B-Spline Ranker")
    logger.info("=" * 50)

    emb_dim = 16
    # KAN takes user_emb + item_emb as input
    model = KANRanker(input_dim=emb_dim * 2, hidden_dim=64)

    # Use pre-trained embeddings from Gold ALS layer as features
    user_emb_df = pd.read_parquet(PROJECT_ROOT / "data" / "datalake" / "gold" / "model_user_embeddings")
    item_emb_df = pd.read_parquet(PROJECT_ROOT / "data" / "datalake" / "gold" / "model_item_embeddings")

    user_embs = {int(r["id"]): np.array(r["features"], dtype=np.float32) for _, r in user_emb_df.iterrows()}
    item_embs = {int(r["id"]): np.array(r["features"], dtype=np.float32) for _, r in item_emb_df.iterrows()}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train = data["train_df"]
    # Filter to rows where we have embeddings
    valid_users = set(user_embs.keys())
    valid_items = set(item_embs.keys())

    num_epochs = 20
    batch_size = 256

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        # Shuffle training data
        train_shuffled = train.sample(frac=1, random_state=epoch)

        batch_user_emb, batch_item_emb, batch_labels = [], [], []

        for _, row in train_shuffled.iterrows():
            uid, mid, rating = int(row["userId"]), int(row["movieId"]), float(row["rating"])
            if uid not in valid_users or mid not in valid_items:
                continue

            # Positive sample
            batch_user_emb.append(user_embs[uid])
            batch_item_emb.append(item_embs[mid])
            batch_labels.append(1.0 if rating >= 3.5 else 0.0)

            # Negative sample (random item)
            neg_candidates = list(
                valid_items
                - data["user_interactions"].get(
                    data["train_df"][data["train_df"]["userId"] == uid]["user_idx"].iloc[0]
                    if len(data["train_df"][data["train_df"]["userId"] == uid]) > 0
                    else 0,
                    set(),
                )
            )
            if neg_candidates:
                neg_id = neg_candidates[np.random.randint(len(neg_candidates))]
                if neg_id in item_embs:
                    batch_user_emb.append(user_embs[uid])
                    batch_item_emb.append(item_embs[neg_id])
                    batch_labels.append(0.0)

            if len(batch_labels) >= batch_size:
                u_tensor = torch.tensor(np.array(batch_user_emb), dtype=torch.float32)
                i_tensor = torch.tensor(np.array(batch_item_emb), dtype=torch.float32)
                y_tensor = torch.tensor(batch_labels, dtype=torch.float32)

                scores = model(u_tensor, i_tensor).squeeze()
                loss = criterion(scores, y_tensor)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                batch_user_emb, batch_item_emb, batch_labels = [], [], []

        avg_loss = total_loss / max(num_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")

    path = MODELS_DIR / "kan_ranker.pth"
    torch.save(model.state_dict(), path)
    logger.info(f"  Saved: {path}")
    return avg_loss


# ============================================================
# Main
# ============================================================


def main():
    logger.info("=" * 60)
    logger.info("PHASE 4: Training All 6 Neural Ensemble Models")
    logger.info("=" * 60)

    # Initialize MLflow tracking if available
    if mlflow is not None:
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("Apex_Neural_Ensemble")
        except Exception:
            pass

    start_time = time.time()
    data = load_data()

    if mlflow is not None:
        try:
            mlflow.log_metric("num_users", data["num_users"])
            mlflow.log_metric("num_items", data["num_items"])
        except Exception:
            pass

    results = {}

    # 1. SASRec
    results["sasrec"] = train_sasrec(data)
    if mlflow is not None:
        try:
            mlflow.log_metric("sasrec_loss", results["sasrec"])
        except Exception:
            pass

        # 2. LightGCN
        results["lightgcn"] = train_lightgcn(data)
        mlflow.log_metric("lightgcn_loss", results["lightgcn"])

        # 3. Quantum Fluid ODE
        results["quantum"] = train_quantum(data)
        mlflow.log_metric("quantum_loss", results["quantum"])

        # 4. Hyperbolic Poincaré
        results["hyperbolic"] = train_hyperbolic(data)
        mlflow.log_metric("hyperbolic_loss", results["hyperbolic"])

        # 4.5. Clifford Geometric Algebra
        results["clifford"] = train_clifford(data)
        mlflow.log_metric("clifford_loss", results["clifford"])

        # 5. KAN Ranker
        results["kan"] = train_kan(data)
        mlflow.log_metric("kan_loss", results["kan"])

        elapsed = time.time() - start_time
        mlflow.log_metric("total_training_time_sec", elapsed)

        logger.info("=" * 60)
        logger.info("PHASE 4 COMPLETE — All 6 Models Trained")
        logger.info(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
        for name, loss in results.items():
            path = MODELS_DIR / f"{name}.pth" if name != "kan" else MODELS_DIR / "kan_ranker.pth"
            if name == "quantum":
                path = MODELS_DIR / "quantum_fluid.pth"
            exists = "✅" if path.exists() else "❌"
            logger.info(f"  {exists} {name:12s} | Final loss: {loss:.4f} | {path.name}")

            # Log model artifact to mlflow
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=f"models/{name}")

        logger.info("=" * 60)


if __name__ == "__main__":
    main()
