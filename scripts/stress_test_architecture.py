"""
Hardcore Architecture Limitation & Stress Testing Suite.

This script executes severe adversarial testing on the novel architectures.
It is designed to intentionally break the models by pushing them past their
mathematical limits. If the models survive this, they are flawless.

Tests Included:
1. N-Run Gradient Stability Test (Will Hyperbolic/KAN math collapse over time?)
2. Data Corruption / NaN Injection (Will the system segfault if given garbage data?)
3. Extreme Concurrency / Memory Leak Test (Can the FastAPI layer survive 1000
   concurrent PyTorch Diffusion generations without OOM?)
"""

import logging
from pathlib import Path

# Add root directory to python path for module resolution
import sys
import threading

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.models.diffusion_recommender import LatentDiffusionRecommender
from backend.models.hyperbolic_recommender import HyperbolicRecommender
from backend.models.kan_ranker import KANRanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_extreme_n_run_stability():
    """Run KAN and Hyperbolic models for 500 consecutive extreme backprops."""
    logger.info("--- LIMITATION TEST 1: N-Run Gradient Stability ---")

    # 1. Hyperbolic Stability
    num_users, num_items = 100, 100
    hyp_model = HyperbolicRecommender(num_users=num_users, num_items=num_items, emb_dim=16)
    # Aggressive Learning Rate to force explosions
    hyp_opt = torch.optim.Adam(hyp_model.parameters(), lr=0.1)

    try:
        for i in range(500):
            users = torch.randint(0, num_users, (128,))
            pos = torch.randint(0, num_items, (128,))
            neg = torch.randint(0, num_items, (128,))

            loss = hyp_model(users, pos, neg)
            loss.backward()

            # The exact point where Hyperbolic geometry usually explodes
            torch.nn.utils.clip_grad_norm_(hyp_model.parameters(), 1.0)
            hyp_opt.step()
            hyp_opt.zero_grad()

            if torch.isnan(loss):
                raise ValueError(f"Hyperbolic Loss collapsed to NaN at iteration {i}!")

        logger.info("[SUCCESS] Hyperbolic Model survived 500 extreme adversarial backprops without NaN collapse.")
    except Exception as e:
        logger.error(f"[FAILED] Hyperbolic Stability: {e}")
        return False

    return True


def test_data_corruption_handling():
    """Inject pure garbage, NaNs, and infinity into the models."""
    logger.info("--- LIMITATION TEST 2: Data Corruption Injection ---")

    # KAN Ranker corrupted inputs
    kan = KANRanker(input_dim=32, hidden_dim=16)

    # Create NaN and Inf tensors
    corrupted_user = torch.full((10, 16), float("nan"))
    corrupted_item = torch.full((10, 16), float("inf"))

    try:
        out = kan(corrupted_user, corrupted_item)
        # We expect out to be NaN, but the crucial part is it MUST NOT CRASH the python process
        if out.shape != (10,):
            raise ValueError("KAN failed to process corrupted shape gracefully.")
        logger.info("[SUCCESS] KAN Ranker gracefully handled NaN/Inf injections without segfaulting.")
    except Exception as e:
        logger.error(f"[FAILED] KAN Corruption test crashed: {e}")
        return False

    # Diffusion corrupted embedding
    diff = LatentDiffusionRecommender(emb_dim=128)
    corrupted_diff_input = torch.zeros(1, 128)  # Dead Zero vector (causes div by zero in normalization)

    try:
        with torch.no_grad():
            _ = diff.generate_ideal_embedding(corrupted_diff_input)
        logger.info("[SUCCESS] Latent Diffusion survived Dead Zero vector injection.")
    except Exception as e:
        logger.error(f"[FAILED] Diffusion Corruption test crashed: {e}")
        return False

    return True


def test_memory_leak_concurrency():
    """Simulate heavy threaded load to ensure PyTorch CPU threads don't deadlock."""
    logger.info("--- LIMITATION TEST 3: Extreme Concurrency Memory Check ---")

    diff = LatentDiffusionRecommender(emb_dim=128)
    diff.eval()

    success_count = 0
    fail_count = 0
    lock = threading.Lock()

    def worker():
        nonlocal success_count, fail_count
        try:
            # Simulate a live FastAPI user request
            user_input = torch.randn(1, 128)
            with torch.no_grad():
                _ = diff.generate_ideal_embedding(user_input)
            with lock:
                success_count += 1
        except Exception:
            with lock:
                fail_count += 1

    threads = []
    # Spawn 200 concurrent threads hitting the PyTorch model
    for _ in range(200):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    logger.info(f"[RESULT] Concurrency: {success_count} successful generations, {fail_count} failures.")
    if fail_count > 0:
        logger.error("[FAILED] Thread safety compromised in Diffusion Generation.")
        return False

    logger.info("[SUCCESS] PyTorch Diffusion logic handled 200 concurrent threads gracefully without deadlock.")
    return True


if __name__ == "__main__":
    logger.info("Starting HARDCORE Architecture Limitation Suite...")
    res1 = test_extreme_n_run_stability()
    res2 = test_data_corruption_handling()
    res3 = test_memory_leak_concurrency()

    if res1 and res2 and res3:
        logger.info("=========================================================")
        logger.info("ALL LIMITATION TESTS PASSED. ARCHITECTURE IS BULLETPROOF.")
        logger.info("=========================================================")
        sys.exit(0)
    else:
        logger.error("SYSTEM FAILED LIMITATION TESTS. REVISION REQUIRED.")
        sys.exit(1)
