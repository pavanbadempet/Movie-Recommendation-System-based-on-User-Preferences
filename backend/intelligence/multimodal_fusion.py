import logging
from pathlib import Path

import numpy as np
from turbovec import TurboQuantIndex

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class MultiModalFusionIndex:
    """
    Fuses textual SBERT embeddings (768d) with visual CLIP embeddings (512d)
    into a unified multi-modal latent space (1280d) using L2 normalization
    and concatenation. Builds and serves a TurboVec index.
    """

    def __init__(self):
        self.index = None
        self.movie_ids = None
        self.text_dim = 768
        self.vision_dim = 512
        self.total_dim = self.text_dim + self.vision_dim

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalization of vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        return vectors / norms

    def build_fusion_index(self):
        """
        Loads independent text and vision embeddings, joins them on movie_id,
        fuses the vectors, and builds `multimodal_turbovec.tq`.
        """
        logger.info("Building Multi-Modal Fusion Index...")

        # Load Text Embeddings
        text_emb_path = MODELS_DIR / "sbert_embeddings.npy"
        text_ids_path = MODELS_DIR / "movie_ids.npy"

        if not text_emb_path.exists() or not text_ids_path.exists():
            logger.error("Text embeddings not found. Run rebuild_serving_artifacts.py first.")
            return False

        text_vectors = np.load(text_emb_path)
        text_ids = np.load(text_ids_path)

        # Load Vision Embeddings
        vision_emb_path = MODELS_DIR / "poster_embeddings.npy"
        vision_ids_path = MODELS_DIR / "poster_movie_ids.npy"

        if not vision_emb_path.exists() or not vision_ids_path.exists():
            logger.error("Vision embeddings not found. Run generate_vision_embeddings.py first.")
            return False

        vision_vectors = np.load(vision_emb_path)
        vision_ids = np.load(vision_ids_path)

        logger.info(f"Loaded Text vectors: {text_vectors.shape}")
        logger.info(f"Loaded Vision vectors: {vision_vectors.shape}")

        # We need an inner join on movie_id
        # Build mapping dicts
        text_dict = {tid: text_vectors[i] for i, tid in enumerate(text_ids)}
        vision_dict = {vid: vision_vectors[i] for i, vid in enumerate(vision_ids)}

        common_ids = sorted(set(text_dict.keys()) & set(vision_dict.keys()))
        logger.info(f"Found {len(common_ids)} movies with BOTH text and vision data.")

        if len(common_ids) == 0:
            logger.error("No intersection between text and vision IDs!")
            return False

        # Construct fused array
        fused_vectors = np.zeros((len(common_ids), self.total_dim), dtype=np.float32)
        fused_ids = np.array(common_ids, dtype=np.int64)

        # Weighted Fusion (Text 0.6, Vision 0.4)
        # First L2 normalize each modality independently so they contribute equally,
        # then apply alpha weights, then concatenate.
        text_weight = 0.6
        vision_weight = 0.4

        for i, mid in enumerate(common_ids):
            t_vec = text_dict[mid]
            v_vec = vision_dict[mid]

            # Normalize
            t_norm = t_vec / (np.linalg.norm(t_vec) + 1e-10)
            v_norm = v_vec / (np.linalg.norm(v_vec) + 1e-10)

            # Weight and concatenate
            fused_vectors[i] = np.concatenate([t_norm * text_weight, v_norm * vision_weight])

        # Final L2 normalization of the fused 1280d vector for cosine similarity search
        fused_vectors = self._normalize(fused_vectors)

        # Build TurboQuantIndex
        logger.info("Initializing TurboQuantIndex (bit_width=4)")
        self.index = TurboQuantIndex(self.total_dim, bit_width=4)
        self.index.add(fused_vectors)

        # Save artifacts
        self.index.write(str(MODELS_DIR / "multimodal_turbovec.tq"))
        np.save(str(MODELS_DIR / "multimodal_movie_ids.npy"), fused_ids)

        logger.info("Multi-Modal TurboVec Index built and saved successfully!")
        return True

    def load_fusion_index(self):
        """Load the pre-built multi-modal index for serving."""
        index_path = MODELS_DIR / "multimodal_turbovec.tq"
        ids_path = MODELS_DIR / "multimodal_movie_ids.npy"

        if not index_path.exists() or not ids_path.exists():
            return False

        self.index = TurboQuantIndex.load(str(index_path))
        self.movie_ids = np.load(ids_path)
        return True

    def search(self, query_text_vector: np.ndarray, query_vision_vector: np.ndarray, top_k: int = 10):
        """
        Execute a search using both modalities.
        Requires the query to have both text and vision representations.
        """
        if self.index is None and not self.load_fusion_index():
            raise RuntimeError("Multi-modal TurboVec index not found.")

        # L2 Normalize inputs
        t_norm = query_text_vector / (np.linalg.norm(query_text_vector) + 1e-10)
        v_norm = query_vision_vector / (np.linalg.norm(query_vision_vector) + 1e-10)

        # Apply weights and concatenate
        fused_query = np.concatenate([t_norm * 0.6, v_norm * 0.4])
        fused_query = fused_query / (np.linalg.norm(fused_query) + 1e-10)
        fused_query = fused_query.reshape(1, -1).astype(np.float32)

        distances, indices = self.index.search(fused_query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx != -1:
                results.append((self.movie_ids[idx], float(dist)))

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fusion = MultiModalFusionIndex()
    fusion.build_fusion_index()
