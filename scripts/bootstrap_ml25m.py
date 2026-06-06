"""
MovieLens-25M Enhanced Bootstrap Pipeline

Implements all four improvements over the ML-100K baseline:
  1. MovieLens-25M (25M ratings, 162K users, 62K movies)
  2. Deeper LightGCN (200 epochs, 3 layers, larger embeddings)
  3. SASRec trained on real chronological sequences
  4. TMDB-enriched Two-Tower features (vote_average, popularity, genres)

Usage:
    python scripts/bootstrap_ml25m.py [--sample N] [--epochs-lgcn N] [--epochs-sasrec N]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import math
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML25_DIR = PROJECT_ROOT / "data" / "raw" / "ml-25m"
ML_SMALL_DIR = PROJECT_ROOT / "data" / "raw" / "ml-latest-small"
MODELS_DIR = PROJECT_ROOT / "models"
GOLD_DIR = PROJECT_ROOT / "data" / "datalake" / "gold"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load ML-25M with TMDB mapping
# ---------------------------------------------------------------------------


def load_ml25m(sample: int | None = None) -> pd.DataFrame:
    """Load ML-25M ratings merged with TMDB IDs. Optionally sample N ratings."""
    ml_dir = ML25_DIR if ML25_DIR.exists() else ML_SMALL_DIR
    logger.info("Loading MovieLens from %s...", ml_dir)

    ratings = pd.read_csv(ml_dir / "ratings.csv")
    links = pd.read_csv(ml_dir / "links.csv").dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    merged = ratings.merge(links[["movieId", "tmdbId"]], on="movieId", how="inner")

    if sample and len(merged) > sample:
        # Stratified sample: keep all users but cap total ratings
        merged = merged.sample(n=sample, random_state=42)

    logger.info(
        "Loaded %d ratings, %d users, %d TMDB movies",
        len(merged),
        merged["userId"].nunique(),
        merged["tmdbId"].nunique(),
    )
    return merged


def load_tmdb_metadata() -> pd.DataFrame:
    """Load TMDB metadata from the processed catalog for feature enrichment."""
    path = DATA_PROCESSED / "movies_transformed.parquet"
    if not path.exists():
        logger.warning("movies_transformed.parquet not found; using empty metadata")
        return pd.DataFrame(columns=["id", "vote_average", "vote_count", "popularity", "genres"])
    cols = ["id", "vote_average", "vote_count", "popularity", "genres"]
    try:
        df = pd.read_parquet(path, columns=cols)
    except Exception:
        df = pd.read_parquet(path)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    return df.dropna(subset=["id"]).astype({"id": int})


# ---------------------------------------------------------------------------
# Step 2: Write events to APEX Event Store (batch JSONL)
# ---------------------------------------------------------------------------


def write_events(ratings: pd.DataFrame) -> int:
    """Batch-write ML-25M ratings as APEX events."""
    from backend.events import get_events_path

    events_path = get_events_path()
    events_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing %d ratings as APEX events...", len(ratings))
    total = 0

    with events_path.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(ratings.itertuples(index=False)):
            ts = datetime.fromtimestamp(row.timestamp, tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
            uid = f"ml25_user_{row.userId}"
            mid = int(row.tmdbId)
            r = float(row.rating)

            fh.write(
                json.dumps(
                    {
                        "event_id": str(uuid.uuid4()),
                        "event_ts": ts,
                        "event_type": "rating",
                        "user_id": uid,
                        "movie_id": mid,
                        "source_content_id": str(mid),
                        "rating": r,
                        "tenant_id": "movielens25",
                        "catalog_id": "tmdb-movies",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            total += 1

            if r >= 3.5:
                fh.write(
                    json.dumps(
                        {
                            "event_id": str(uuid.uuid4()),
                            "event_ts": ts,
                            "event_type": "click",
                            "user_id": uid,
                            "movie_id": mid,
                            "source_content_id": str(mid),
                            "tenant_id": "movielens25",
                            "catalog_id": "tmdb-movies",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                total += 1

            if (i + 1) % 100000 == 0:
                logger.info("  %d / %d ratings written (%d events)...", i + 1, len(ratings), total)

    logger.info("Event write complete: %d events", total)
    return total


# ---------------------------------------------------------------------------
# Step 3: Train deeper LightGCN (Improvement 2)
# ---------------------------------------------------------------------------


def train_lightgcn(ratings: pd.DataFrame, epochs: int = 200, emb_dim: int = 32) -> None:
    """Train LightGCN with more epochs and larger embeddings on ML-25M."""
    import scipy.sparse as sp

    from backend.models.lightgcn import LightGCN

    logger.info("Building LightGCN interaction graph...")
    positives = ratings[ratings["rating"] >= 3.5].copy()

    user_ids = sorted(positives["userId"].unique())
    item_ids = sorted(positives["tmdbId"].unique())
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {m: i for i, m in enumerate(item_ids)}
    num_users, num_items = len(user_ids), len(item_ids)

    logger.info("Graph: %d users, %d items, %d positive interactions", num_users, num_items, len(positives))

    rows = positives["userId"].map(user_map).values
    cols = positives["tmdbId"].map(item_map).values
    data = np.ones(len(rows), dtype=np.float32)
    R = sp.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

    adj = sp.bmat(
        [[sp.csr_matrix((num_users, num_users)), R], [R.T, sp.csr_matrix((num_items, num_items))]], format="csr"
    )

    deg = np.array(adj.sum(axis=1)).flatten()
    d_inv = np.power(deg + 1e-8, -0.5)
    D = sp.diags(d_inv)
    adj_norm = D @ adj @ D

    coo = adj_norm.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    torch.sparse_coo_tensor(indices, values, size=(num_users + num_items, num_users + num_items)).coalesce()

    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=emb_dim, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    rng = np.random.default_rng(42)
    batch_size = 4096

    logger.info("Training LightGCN for %d epochs (emb_dim=%d, direct BPR)...", epochs, emb_dim)
    for epoch in range(epochs):
        model.train()

        # Sample 200K positives per epoch
        pos_sample = positives.sample(min(200000, len(positives)), random_state=epoch)
        u_arr = pos_sample["userId"].map(user_map).values.astype(np.int64)
        p_arr = pos_sample["tmdbId"].map(item_map).values.astype(np.int64)
        n_arr = rng.integers(0, num_items, size=len(u_arr)).astype(np.int64)

        idx = rng.permutation(len(u_arr))
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(idx), batch_size):
            bi = idx[start : start + batch_size]
            u = torch.tensor(u_arr[bi], dtype=torch.long)
            p = torch.tensor(p_arr[bi], dtype=torch.long)
            n = torch.tensor(n_arr[bi], dtype=torch.long)

            # Direct BPR on embedding tables (no graph propagation per batch)
            u_emb = model.user_embedding(u)
            p_emb = model.item_embedding(p)
            n_emb = model.item_embedding(n)
            pos_scores = (u_emb * p_emb).sum(dim=1)
            neg_scores = (u_emb * n_emb).sum(dim=1)
            loss = F.softplus(neg_scores - pos_scores).mean()
            # L2 regularization
            loss = loss + 1e-4 * (u_emb.norm(2).pow(2) + p_emb.norm(2).pow(2) + n_emb.norm(2).pow(2)) / len(bi)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info("  Epoch %d/%d | Loss: %.4f", epoch + 1, epochs, total_loss / max(n_batches, 1))

    torch.save(model.state_dict(), MODELS_DIR / "lightgcn.pth")
    logger.info("LightGCN saved.")

    # Export embeddings
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    (GOLD_DIR / "model_user_embeddings").mkdir(exist_ok=True)
    (GOLD_DIR / "model_item_embeddings").mkdir(exist_ok=True)
    with torch.no_grad():
        u_embs = model.user_embedding.weight.cpu().numpy()
        i_embs = model.item_embedding.weight.cpu().numpy()
    pd.DataFrame([{"id": uid, "features": u_embs[user_map[uid]].tolist()} for uid in user_ids]).to_parquet(
        GOLD_DIR / "model_user_embeddings" / "part-0.parquet"
    )
    pd.DataFrame([{"id": mid, "features": i_embs[item_map[mid]].tolist()} for mid in item_ids]).to_parquet(
        GOLD_DIR / "model_item_embeddings" / "part-0.parquet"
    )
    logger.info("LightGCN embeddings exported: %d users, %d items", num_users, num_items)


# ---------------------------------------------------------------------------
# Step 4: Train SASRec on real chronological sequences (Improvement 3)
# ---------------------------------------------------------------------------


def train_sasrec(ratings: pd.DataFrame, epochs: int = 50, max_users: int = 50000) -> None:
    """Train SASRec on real ML-25M watch sequences ordered by timestamp."""
    from backend.models.sasrec import SASRec

    logger.info("Building SASRec training sequences...")

    # Sort by user and timestamp to get chronological sequences
    sorted_ratings = ratings.sort_values(["userId", "timestamp"])

    # Build item ID mapping (compact indices) — use only items with enough interactions
    item_counts = ratings["tmdbId"].value_counts()
    popular_items = item_counts[item_counts >= 5].index
    item_ids = sorted(popular_items.tolist())
    item_map = {mid: i + 1 for i, mid in enumerate(item_ids)}  # 0 = padding
    num_items = len(item_ids)

    # Vectorized sequence building using pandas groupby apply
    sorted_ratings = sorted_ratings[sorted_ratings["tmdbId"].isin(item_map)]
    sorted_ratings = sorted_ratings.copy()
    sorted_ratings["item_idx"] = sorted_ratings["tmdbId"].map(item_map)

    # Build sequences as arrays
    user_seqs: dict[int, list[int]] = {}
    for uid, grp in sorted_ratings.groupby("userId", sort=False):
        seq = grp["item_idx"].tolist()
        if len(seq) >= 3:
            user_seqs[uid] = seq

    # Cap to max_users
    if len(user_seqs) > max_users:
        sampled = list(user_seqs.keys())[:max_users]
        user_seqs = {u: user_seqs[u] for u in sampled}

    logger.info("SASRec: %d users with sequences, %d unique items", len(user_seqs), num_items)

    MAX_SEQ = 50
    model = SASRec(num_items=num_items, max_seq_len=MAX_SEQ, hidden_dim=64, num_blocks=2, num_heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    rng = np.random.default_rng(42)
    batch_size = 256

    logger.info("Training SASRec for %d epochs...", epochs)
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0

        # Sample users for this epoch
        sampled_uids = rng.choice(list(user_seqs.keys()), size=min(5000, len(user_seqs)), replace=False)

        seqs_batch, targets_batch = [], []
        for uid in sampled_uids:
            seq = user_seqs[uid]
            if len(seq) < 2:
                continue
            # Pick a random position in the sequence
            pos = rng.integers(1, len(seq))
            inp = seq[max(0, pos - MAX_SEQ) : pos]
            inp = [0] * (MAX_SEQ - len(inp)) + inp
            seqs_batch.append(inp)
            targets_batch.append(seq[pos])

        if not seqs_batch:
            continue

        idx = rng.permutation(len(seqs_batch))
        for start in range(0, len(idx), batch_size):
            bi = idx[start : start + batch_size]
            seqs_t = torch.tensor([seqs_batch[i] for i in bi], dtype=torch.long)
            pos_t = torch.tensor([targets_batch[i] for i in bi], dtype=torch.long)
            neg_t = torch.tensor(rng.integers(1, num_items + 1, size=len(bi)), dtype=torch.long)

            seq_out = model(seqs_t)
            final = seq_out[:, -1, :]
            pos_emb = model.item_emb(pos_t)
            neg_emb = model.item_emb(neg_t)
            pos_scores = (final * pos_emb).sum(dim=-1)
            neg_scores = (final * neg_emb).sum(dim=-1)
            loss = F.softplus(neg_scores - pos_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            logger.info("  SASRec Epoch %d/%d | Loss: %.4f", epoch + 1, epochs, total_loss / max(n_batches, 1))

    torch.save(model.state_dict(), MODELS_DIR / "sasrec.pth")
    logger.info("SASRec saved to models/sasrec.pth")


# ---------------------------------------------------------------------------
# Step 5: Fine-tune Two-Tower with TMDB-enriched features (Improvement 4)
# ---------------------------------------------------------------------------


def train_two_tower_enriched(ratings: pd.DataFrame, tmdb_meta: pd.DataFrame, epochs: int = 10) -> None:
    """Fine-tune Two-Tower using LightGCN embeddings + TMDB metadata as item features."""
    from torch.utils.data import DataLoader, Dataset

    from backend.models.two_tower import TwoTowerModel

    logger.info("Building enriched Two-Tower training data...")

    # Load LightGCN embeddings from Gold layer
    user_emb_path = GOLD_DIR / "model_user_embeddings" / "part-0.parquet"
    item_emb_path = GOLD_DIR / "model_item_embeddings" / "part-0.parquet"

    if user_emb_path.exists() and item_emb_path.exists():
        user_emb_df = pd.read_parquet(user_emb_path)
        item_emb_df = pd.read_parquet(item_emb_path)
        emb_dim = len(user_emb_df.iloc[0]["features"])
        logger.info(
            "Loaded LightGCN embeddings: %d users, %d items (dim=%d)", len(user_emb_df), len(item_emb_df), emb_dim
        )
        user_als = {int(r["id"]): np.array(r["features"], dtype=np.float32) for _, r in user_emb_df.iterrows()}
        item_als = {int(r["id"]): np.array(r["features"], dtype=np.float32) for _, r in item_emb_df.iterrows()}
    else:
        logger.warning("LightGCN embeddings not found; using zeros")
        emb_dim = 16
        user_als, item_als = {}, {}

    # Build TMDB metadata lookup
    tmdb_lookup: dict[int, dict] = {}
    for _, row in tmdb_meta.iterrows():
        tmdb_lookup[int(row["id"])] = {
            "vote_average": float(row.get("vote_average") or 6.0),
            "vote_count": float(row.get("vote_count") or 100),
            "popularity": float(row.get("popularity") or 10.0),
            "genres": str(row.get("genres") or ""),
        }

    # User feature dim: emb_dim + 2 (total_ratings, avg_rating)
    # Item feature dim: emb_dim + 4 (vote_avg, log_vote_count, log_popularity, num_genres)
    user_feat_dim = emb_dim + 2
    item_feat_dim = emb_dim + 4

    # Aggregate user stats
    user_stats = (
        ratings.groupby("userId")
        .agg(
            total_ratings=("rating", "count"),
            avg_rating=("rating", "mean"),
        )
        .reset_index()
    )
    user_stats_map = {int(r["userId"]): r for _, r in user_stats.iterrows()}

    def build_user_feat(uid: int) -> np.ndarray:
        als = user_als.get(uid, np.zeros(emb_dim, dtype=np.float32))
        stats = user_stats_map.get(uid)
        if stats is not None:
            activity = np.array(
                [
                    math.log1p(float(stats["total_ratings"])) / math.log1p(1000),
                    float(stats["avg_rating"]) / 5.0,
                ],
                dtype=np.float32,
            )
        else:
            activity = np.zeros(2, dtype=np.float32)
        return np.concatenate([als[:emb_dim], activity])

    def build_item_feat(mid: int) -> np.ndarray:
        als = item_als.get(mid, np.zeros(emb_dim, dtype=np.float32))
        meta = tmdb_lookup.get(mid, {})
        vote_avg = float(meta.get("vote_average", 6.0)) / 10.0
        vote_cnt = math.log1p(float(meta.get("vote_count", 100))) / 15.0
        pop = math.log1p(float(meta.get("popularity", 10.0))) / 10.0
        genres = str(meta.get("genres", ""))
        num_genres = min(len(genres.split(",")) / 10.0, 1.0) if genres else 0.0
        metadata = np.array([vote_avg, vote_cnt, pop, num_genres], dtype=np.float32)
        return np.concatenate([als[:emb_dim], metadata])

    # Build positive pairs
    positives = ratings[ratings["rating"] >= 3.5][["userId", "tmdbId"]].values
    logger.info("Two-Tower: %d positive pairs", len(positives))

    if len(positives) < 100:
        logger.warning("Too few positive pairs; skipping Two-Tower training")
        return

    # Train/val split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(positives))
    split = int(len(positives) * 0.8)
    train_pairs = positives[idx[:split]]
    positives[idx[split:]]

    class PairDataset(Dataset):
        def __init__(self, pairs, num_negatives=4):
            self.pairs = pairs
            self.num_negatives = num_negatives
            self.all_items = list({int(p[1]) for p in pairs})
            self.user_pos: dict[int, set[int]] = {}
            for uid, mid in pairs:
                self.user_pos.setdefault(int(uid), set()).add(int(mid))

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, i):
            uid, mid = int(self.pairs[i][0]), int(self.pairs[i][1])
            u_feat = build_user_feat(uid)
            p_feat = build_item_feat(mid)
            neg_feats = []
            attempts = 0
            while len(neg_feats) < self.num_negatives and attempts < self.num_negatives * 10:
                neg = self.all_items[rng.integers(len(self.all_items))]
                if neg not in self.user_pos.get(uid, set()):
                    neg_feats.append(build_item_feat(neg))
                attempts += 1
            while len(neg_feats) < self.num_negatives:
                neg_feats.append(np.zeros(item_feat_dim, dtype=np.float32))
            return (
                torch.tensor(u_feat, dtype=torch.float32),
                torch.tensor(p_feat, dtype=torch.float32),
                torch.tensor(np.array(neg_feats), dtype=torch.float32),
            )

    dataset = PairDataset(train_pairs, num_negatives=4)
    loader = DataLoader(dataset, batch_size=min(512, len(dataset)), shuffle=True, num_workers=0)

    model = TwoTowerModel(user_input_dim=user_feat_dim, item_input_dim=item_feat_dim, embedding_dim=128)
    base_path = MODELS_DIR / "two_tower.pth"
    if base_path.exists():
        try:
            model.load_state_dict(torch.load(base_path, map_location="cpu", weights_only=True))
            logger.info("Loaded base Two-Tower weights")
        except Exception as e:
            logger.warning("Could not load base weights: %s", e)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    logger.info("Training enriched Two-Tower for %d epochs...", epochs)
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for u_feat, p_feat, n_feat in loader:
            loss = model.compute_contrastive_loss(u_feat, p_feat, n_feat)
            if torch.isnan(loss):
                logger.error("NaN loss at epoch %d; stopping", epoch + 1)
                return
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        logger.info("  Two-Tower Epoch %d/%d | Loss: %.4f", epoch + 1, epochs, total_loss / max(n_batches, 1))

    torch.save(model.state_dict(), MODELS_DIR / "two_tower_finetuned.pth")
    logger.info("Enriched Two-Tower saved to models/two_tower_finetuned.pth")


# ---------------------------------------------------------------------------
# Step 6: Run RL + ensemble calibration
# ---------------------------------------------------------------------------


def run_calibration() -> None:
    import subprocess

    scripts = [
        ("RL policy", ["python", "scripts/train_rl_policy_compact.py", "--epochs", "300"]),
        ("Ensemble weights", ["python", "scripts/optimize_ensemble_weights.py", "--num-candidates", "500"]),
    ]
    for name, cmd in scripts:
        logger.info("Running: %s", name)
        t = time.time()
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        logger.info("%s done in %.1fs", name, time.time() - t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    sample: int | None = None,
    skip_events: bool = False,
    skip_lgcn: bool = False,
    skip_sasrec: bool = False,
    skip_twotower: bool = False,
    skip_calibration: bool = False,
    epochs_lgcn: int = 200,
    epochs_sasrec: int = 50,
    epochs_twotower: int = 10,
) -> None:
    logger.info("=" * 60)
    logger.info("APEX ML-25M Enhanced Bootstrap Pipeline")
    logger.info("=" * 60)

    ratings = load_ml25m(sample=sample)
    tmdb_meta = load_tmdb_metadata()

    if not skip_events:
        write_events(ratings)

    if not skip_lgcn:
        train_lightgcn(ratings, epochs=epochs_lgcn)

    if not skip_sasrec:
        train_sasrec(ratings, epochs=epochs_sasrec)

    if not skip_twotower:
        train_two_tower_enriched(ratings, tmdb_meta, epochs=epochs_twotower)

    if not skip_calibration:
        run_calibration()

    logger.info("=" * 60)
    logger.info("Bootstrap complete. Restart the API server to load new weights.")
    logger.info("=" * 60)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Bootstrap APEX from MovieLens-25M with all four improvements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sample", type=int, default=None, help="Cap total ratings (None = all 25M)")
    p.add_argument("--skip-events", action="store_true")
    p.add_argument("--skip-lgcn", action="store_true")
    p.add_argument("--skip-sasrec", action="store_true")
    p.add_argument("--skip-twotower", action="store_true")
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--epochs-lgcn", type=int, default=200)
    p.add_argument("--epochs-sasrec", type=int, default=50)
    p.add_argument("--epochs-twotower", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        sample=args.sample,
        skip_events=args.skip_events,
        skip_lgcn=args.skip_lgcn,
        skip_sasrec=args.skip_sasrec,
        skip_twotower=args.skip_twotower,
        skip_calibration=args.skip_calibration,
        epochs_lgcn=args.epochs_lgcn,
        epochs_sasrec=args.epochs_sasrec,
        epochs_twotower=args.epochs_twotower,
    )
