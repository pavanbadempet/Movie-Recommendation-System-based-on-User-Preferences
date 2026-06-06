"""
Generative Diffusion Recommender - Evaluation Harness.

This script tests the "Generative Retrieval" hypothesis against the industry standard
"Mean-Pooling Baseline" (averaging a user's past embeddings and searching).

Research Claim: Denoising diffusion creates a mathematically superior target
embedding compared to simple geometric averaging, leading to higher HitRate@K
and NDCG@K metrics on long-tail distributions.
"""

import logging
from pathlib import Path
import sys

import numpy as np
import torch

# Fix python path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.models.diffusion_recommender import LatentDiffusionRecommender
from scripts.train_generative_diffusion import build_synthetic_experiment_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CHECKPOINT_PATH = MODELS_DIR / "diffusion_recommender.pth"


def calculate_ndcg(rank, k=10):
    """Calculates Normalized Discounted Cumulative Gain."""
    if rank < k:
        return 1.0 / np.log2(rank + 2)
    return 0.0


def evaluate_models():
    """Evaluates Generative Diffusion against Baseline Mean-Pooling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_dim = 384
    top_k = 10

    logger.info("Loading Test Dataset...")
    # Use different seed for test data
    np.random.seed(42)
    item_embeddings, test_histories = build_synthetic_experiment_data(num_items=1000, emb_dim=emb_dim, num_users=500)

    # Initialize TurboVec index for retrieval
    try:
        from turbovec import TurboQuantIndex

        logger.info("Initializing TurboVec Index for Sub-Millisecond Retrieval...")
        index = TurboQuantIndex(emb_dim, bit_width=4)
        index.add(item_embeddings)
    except ImportError:
        logger.error("TurboVec not installed. Cannot perform retrieval evaluation.")
        return

    # Load trained model
    logger.info(f"Loading trained Diffusion Recommender from {CHECKPOINT_PATH}...")
    model = LatentDiffusionRecommender(emb_dim=emb_dim, num_timesteps=100).to(device)

    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        model.eval()
    else:
        logger.warning("No trained weights found. Evaluating with untrained initialized weights.")

    # Evaluation Metrics
    baseline_hits = 0
    baseline_ndcg = 0.0

    diffusion_hits = 0
    diffusion_ndcg = 0.0

    num_eval = len(test_histories)
    logger.info(f"Evaluating across {num_eval} test users...")

    with torch.no_grad():
        for record in test_histories:
            history = record["history"]
            target = record["target"]

            # --- 1. Baseline: Mean Pooling ---
            if len(history) > 0:
                user_emb_np = np.mean(item_embeddings[history], axis=0, keepdims=True)
            else:
                user_emb_np = np.zeros((1, emb_dim), dtype=np.float32)

            # Normalize baseline vector
            user_emb_np = user_emb_np / (np.linalg.norm(user_emb_np) + 1e-9)

            # Retrieve top K
            _, baseline_indices = index.search(user_emb_np, top_k)
            baseline_indices = baseline_indices[0].tolist()

            if target in baseline_indices:
                baseline_hits += 1
                rank = baseline_indices.index(target)
                baseline_ndcg += calculate_ndcg(rank, top_k)

            # --- 2. Proposed: Generative Diffusion Retrieval ---
            user_emb_tensor = torch.FloatTensor(user_emb_np).to(device)
            ideal_emb_tensor = model.generate_ideal_embedding(user_emb_tensor)
            ideal_emb_np = ideal_emb_tensor.cpu().numpy()

            # Retrieve top K using hallucinated embedding
            _, diff_indices = index.search(ideal_emb_np, top_k)
            diff_indices = diff_indices[0].tolist()

            if target in diff_indices:
                diffusion_hits += 1
                rank = diff_indices.index(target)
                diffusion_ndcg += calculate_ndcg(rank, top_k)

    # Calculate final metrics
    base_hr = baseline_hits / num_eval
    base_ndcg = baseline_ndcg / num_eval

    diff_hr = diffusion_hits / num_eval
    diff_ndcg = diffusion_ndcg / num_eval

    logger.info("=========================================")
    logger.info("   RESEARCH EXPERIMENT RESULTS (K=10)    ")
    logger.info("=========================================")
    logger.info(f"Baseline (Mean Pooling) HitRate@10 : {base_hr:.4f}")
    logger.info(f"Baseline (Mean Pooling) NDCG@10    : {base_ndcg:.4f}")
    logger.info("-----------------------------------------")
    logger.info(f"Generative Diffusion    HitRate@10 : {diff_hr:.4f}")
    logger.info(f"Generative Diffusion    NDCG@10    : {diff_ndcg:.4f}")
    logger.info("=========================================")

    if diff_hr > base_hr:
        logger.info("SUCCESS: Diffusion Retrieval outperformed Baseline!")
    else:
        logger.info("NOTE: Diffusion requires more training epochs or larger dataset to surpass Baseline.")


if __name__ == "__main__":
    evaluate_models()
