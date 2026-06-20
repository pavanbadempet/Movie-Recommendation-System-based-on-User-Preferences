"""Train the Quantum Fluid model from real Gold embeddings and ratings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.models.neural_ode_recommender import QuantumFluidRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = PROJECT_ROOT / "data" / "datalake" / "gold"
RATINGS_PATH = PROJECT_ROOT / "data" / "processed" / "ratings_transformed.parquet"
MODEL_EXPORT_PATH = PROJECT_ROOT / "models" / "quantum_fluid.pth"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "quantum_fluid.metadata.json"


@dataclass(frozen=True)
class EmbeddingBundle:
    user_tensor: torch.Tensor
    item_tensor: torch.Tensor
    user_id_to_index: dict[int, int]
    item_id_to_index: dict[int, int]
    user_ids: list[int]
    item_ids: list[int]


def _load_embedding_frame(path: Path, embedding_dim: int | None) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    frames: list[pd.DataFrame] = []
    observed_dims: set[int] = set()
    for file_path in files:
        frame = pd.read_parquet(file_path)
        if frame.empty or not {"id", "features"}.issubset(frame.columns):
            continue
        dimension = len(frame.iloc[0]["features"])
        observed_dims.add(dimension)
        if embedding_dim is None or dimension == embedding_dim:
            frames.append(frame[["id", "features"]])
    if not frames:
        raise ValueError(f"No usable embeddings found in {path} for dimension {embedding_dim}")
    if embedding_dim is None and len(observed_dims) > 1:
        raise ValueError(f"Multiple embedding dimensions found in {path}: {sorted(observed_dims)}")
    result = pd.concat(frames, ignore_index=True).sort_values("id").drop_duplicates("id", keep="last")
    return result.reset_index(drop=True)


def load_pyspark_embeddings(
    gold_dir: Path | str = GOLD_DIR,
    embedding_dim: int | None = None,
) -> EmbeddingBundle:
    """Load real user/item embeddings and their external-ID mappings."""
    gold_dir = Path(gold_dir)
    user_path = gold_dir / "model_user_embeddings"
    item_path = gold_dir / "model_item_embeddings"
    if not user_path.exists() or not item_path.exists():
        raise FileNotFoundError(
            f"Gold embeddings are required at {user_path} and {item_path}. "
            "Run scripts/pyspark_medallion_pipeline.py first."
        )

    users = _load_embedding_frame(user_path, embedding_dim)
    items = _load_embedding_frame(item_path, embedding_dim)
    user_tensor = torch.tensor(np.vstack(users["features"]), dtype=torch.float32)
    item_tensor = torch.tensor(np.vstack(items["features"]), dtype=torch.float32)
    if user_tensor.shape[1] != item_tensor.shape[1]:
        raise ValueError(
            f"User/item embedding dimensions differ: {user_tensor.shape[1]} != {item_tensor.shape[1]}"
        )

    user_ids = [int(value) for value in users["id"].tolist()]
    item_ids = [int(value) for value in items["id"].tolist()]
    return EmbeddingBundle(
        user_tensor=user_tensor,
        item_tensor=item_tensor,
        user_id_to_index={value: index for index, value in enumerate(user_ids)},
        item_id_to_index={value: index for index, value in enumerate(item_ids)},
        user_ids=user_ids,
        item_ids=item_ids,
    )


def load_positive_interactions(
    ratings_path: Path | str,
    bundle: EmbeddingBundle,
    minimum_rating: float = 3.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map observed positive ratings to embedding row indices."""
    ratings_path = Path(ratings_path)
    if not ratings_path.is_file():
        raise FileNotFoundError(f"Ratings evidence not found: {ratings_path}")
    ratings = pd.read_parquet(ratings_path)
    required = {"userId", "movieId", "rating"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"Ratings evidence must contain {sorted(required)}")

    positive = ratings[pd.to_numeric(ratings["rating"], errors="coerce") >= minimum_rating]
    users: list[int] = []
    items: list[int] = []
    for row in positive.itertuples(index=False):
        user_index = bundle.user_id_to_index.get(int(row.userId))
        item_index = bundle.item_id_to_index.get(int(row.movieId))
        if user_index is None or item_index is None:
            continue
        users.append(user_index)
        items.append(item_index)
    if not users:
        raise ValueError("No positive ratings align with the selected Gold embeddings")
    return torch.tensor(users, dtype=torch.long), torch.tensor(items, dtype=torch.long)


def train_quantum_hybrid(
    bundle: EmbeddingBundle,
    positive_users: torch.Tensor,
    positive_items: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 1024,
) -> QuantumFluidRecommender:
    """Fine-tune from observed positive pairs and sampled unobserved negatives."""
    num_users, embedding_dim = bundle.user_tensor.shape
    num_items = bundle.item_tensor.shape[0]
    if len(positive_users) != len(positive_items) or len(positive_users) == 0:
        raise ValueError("Observed positive user/item pairs are required")

    model = QuantumFluidRecommender(num_users, num_items, embedding_dim)
    with torch.no_grad():
        model.user_embedding.amplitude.weight.copy_(bundle.user_tensor)
        model.item_embedding.amplitude.weight.copy_(bundle.item_tensor)

    positives_by_user: dict[int, set[int]] = {}
    for user, item in zip(positive_users.tolist(), positive_items.tolist(), strict=True):
        positives_by_user.setdefault(user, set()).add(item)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(42)
    model.train()
    for epoch in range(epochs):
        sample_indices = torch.randint(
            len(positive_users),
            (min(batch_size, len(positive_users)),),
            generator=generator,
        )
        batch_users = positive_users[sample_indices]
        batch_pos = positive_items[sample_indices]
        batch_neg = torch.randint(num_items, batch_pos.shape, generator=generator)
        for index, (user, negative) in enumerate(zip(batch_users.tolist(), batch_neg.tolist(), strict=True)):
            attempts = 0
            while negative in positives_by_user[user] and attempts < num_items:
                negative = (negative + 1) % num_items
                attempts += 1
            if negative in positives_by_user[user]:
                raise ValueError(f"User index {user} has no unobserved item for negative sampling")
            batch_neg[index] = negative

        optimizer.zero_grad()
        loss = model(batch_users, batch_pos, batch_neg, torch.ones(len(batch_users)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        logger.info("Epoch %d/%d | loss %.6f", epoch + 1, epochs, loss.item())
    return model


def export_production_binary(model: nn.Module, bundle: EmbeddingBundle) -> None:
    """Save serving weights plus the external-ID mapping provenance."""
    MODEL_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_EXPORT_PATH)
    MODEL_METADATA_PATH.write_text(
        json.dumps(
            {
                "source": "gold_embeddings_and_observed_ratings",
                "num_users": len(bundle.user_ids),
                "num_items": len(bundle.item_ids),
                "embedding_dim": int(bundle.user_tensor.shape[1]),
                "user_ids": bundle.user_ids,
                "item_ids": bundle.item_ids,
            }
        ),
        encoding="utf-8",
    )
    logger.info("Production weights exported to %s", MODEL_EXPORT_PATH)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--minimum-rating", type=float, default=3.5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bundle = load_pyspark_embeddings(embedding_dim=args.embedding_dim)
    users, items = load_positive_interactions(RATINGS_PATH, bundle, args.minimum_rating)
    model = train_quantum_hybrid(bundle, users, items, epochs=args.epochs, batch_size=args.batch_size)
    export_production_binary(model, bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
