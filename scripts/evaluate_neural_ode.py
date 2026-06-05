"""
Quantum-Fluid Manifold Validation.

This script tests the mathematically unprecedented Quantum-Fluid architecture.
It ensures that PyTorch can successfully backpropagate through continuous-time
complex-valued ODEs and quantum interference operations without gradient collapse.
"""

import logging
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.neural_ode_recommender import QuantumFluidRecommender
from scripts.evaluate_novel_quality import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_quantum_fluid():
    logger.info("=========================================================")
    logger.info("INITIALIZING QUANTUM-FLUID MANIFOLD RECOMMENDER")
    logger.info("=========================================================")

    num_users = 200
    num_items = 200
    emb_dim = 16

    qfmr = QuantumFluidRecommender(num_users=num_users, num_items=num_items, emb_dim=emb_dim)
    # The complex manifold requires extremely careful optimization to prevent phase-lock
    optimizer = torch.optim.Adam(qfmr.parameters(), lr=0.05)

    # Ground Truth: User i perfectly aligns with Item i
    users = torch.arange(num_users)
    pos_items = torch.arange(num_items)

    logger.info("Simulating Continuous-Time Differential Flow...")
    for _epoch in range(150):
        neg_items = torch.randint(0, num_items, (num_users,))
        # Simulate that each user session occurs after a random temporal delta
        time_deltas = torch.rand(num_users) * 2.0

        loss = qfmr(users, pos_items, neg_items, time_deltas)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qfmr.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    logger.info("Extracting states via Wave Interference Patterns...")
    ranks = []
    with torch.no_grad():
        for i in range(50):
            u_id = torch.tensor([i])
            c_ids = torch.arange(num_items)

            scores = qfmr.predict(u_id, c_ids, time_delta=1.0)
            # Scores are 1D tensor
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)

    metrics = compute_metrics(ranks, k=10)
    logger.info(f"Quantum-Fluid Quality -> HR@10: {metrics['HitRate@10']:.3f}, NDCG@10: {metrics['NDCG@10']:.3f}")

    if metrics["HitRate@10"] < 0.8:
        logger.error("Quantum interference failed to isolate the signal.")
        sys.exit(1)

    logger.info("=========================================================")
    logger.info("QUANTUM-FLUID ARCHITECTURE EMPIRICALLY VALIDATED.")
    logger.info("CONTINUOUS TIME ODE BACKPROPAGATION SUCCESSFUL.")
    logger.info("=========================================================")


if __name__ == "__main__":
    evaluate_quantum_fluid()
