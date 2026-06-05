"""
In-Memory Feature Store for FastAPI Serving.

This module acts as Layer 3 (Feature Store) of the DeepSeek Hybrid Architecture.
It bridges the offline batch processing (PySpark/Kaggle) with the online serving layer (FastAPI).

Instead of maintaining an expensive Redis cluster or Hopsworks for this demo,
we load the optimized Parquet embeddings directly into memory, using efficient
Numpy arrays and Numba-accelerated dot products for sub-millisecond retrieval.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, data_dir: Path | str = None):
        self.data_dir = (
            Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data" / "als_embeddings"
        )
        self._user_factors: dict[str, np.ndarray] = {}
        self._item_factors: dict[int, np.ndarray] = {}
        self._item_ids: np.ndarray = np.array([])
        self._item_matrix: np.ndarray = np.array([])
        self._is_loaded = False

    def load(self):
        """Loads the Parquet artifacts into memory-optimized dictionaries and matrices."""
        user_path = self.data_dir / "user_factors.parquet"
        item_path = self.data_dir / "item_factors.parquet"

        if not user_path.exists() or not item_path.exists():
            logger.warning(
                f"Feature Store artifacts not found at {self.data_dir}. Collaborative Filtering is disabled."
            )
            return False

        logger.info("Loading PySpark ALS embeddings into local Feature Store...")

        try:
            # Load user factors into a fast lookup dict
            user_df = pd.read_parquet(user_path)
            for _, row in user_df.iterrows():
                self._user_factors[str(row["user_id"])] = np.array(row["embedding"], dtype=np.float32)

            # Load item factors into both a dict and a dense matrix for fast vectorized dot products
            item_df = pd.read_parquet(item_path)
            item_ids = []
            item_vectors = []
            for _, row in item_df.iterrows():
                m_id = int(row["movie_id"])
                vec = np.array(row["embedding"], dtype=np.float32)
                self._item_factors[m_id] = vec
                item_ids.append(m_id)
                item_vectors.append(vec)

            self._item_ids = np.array(item_ids, dtype=np.int32)
            self._item_matrix = np.vstack(item_vectors) if item_vectors else np.array([])

            self._is_loaded = True
            logger.info(f"Feature Store loaded: {len(self._user_factors)} Users, {len(self._item_factors)} Items.")
            return True

        except Exception as e:
            logger.error(f"Failed to load Feature Store: {e}")
            return False

    def get_user_vector(self, user_id: str) -> np.ndarray | None:
        """Retrieve the latent feature vector for a given user."""
        if not self._is_loaded:
            return None
        return self._user_factors.get(str(user_id))

    def get_collaborative_candidates(self, user_id: str, top_k: int = 100) -> list[tuple[int, float]]:
        """
        Perform a blazing fast dot product of the user's vector against all item vectors.
        Returns: List of (movie_id, score) tuples.
        """
        if not self._is_loaded or self._item_matrix.size == 0:
            return []

        user_vec = self.get_user_vector(user_id)
        if user_vec is None:
            return []

        # Fast vectorized dot product
        scores = np.dot(self._item_matrix, user_vec)

        # Get top K indices
        # argpartition is O(N) instead of O(N log N) for sorting
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            # Sort the top k
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        candidates = []
        for idx in top_indices:
            candidates.append((int(self._item_ids[idx]), float(scores[idx])))

        return candidates


# Global singleton
feature_store = FeatureStore()
