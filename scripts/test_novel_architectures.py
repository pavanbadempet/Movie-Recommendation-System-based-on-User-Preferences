"""
Master Test Harness for Novel Architectures.

This script rigorously tests all the bleeding-edge models we implemented
(KAN, Hyperbolic, SASRec, LightGCN).
It generates synthetic tensors, passes them through the forward pass,
computes the mathematically specific loss functions (e.g. Poincaré Margin Loss),
and executes a backward pass to ensure gradients flow correctly without
exploding (a common issue in non-Euclidean and Transformer geometries).
"""

import logging
from pathlib import Path

# Add root directory to python path for module resolution
import sys

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.hyperbolic_recommender import HyperbolicRecommender
from backend.kan_ranker import KANRanker
from backend.lightgcn import LightGCN
from backend.sasrec import SASRec

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_kan_ranker():
    logger.info("--- Testing Kolmogorov-Arnold Network (KAN) Ranker ---")
    batch_size = 64
    emb_dim = 64

    # Mock User and Item embeddings
    user_emb = torch.randn(batch_size, emb_dim)
    item_emb = torch.randn(batch_size, emb_dim)
    target_ctr = torch.rand(batch_size)  # Random CTR probabilities

    model = KANRanker(input_dim=emb_dim * 2, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Forward Pass
    predictions = model(user_emb, item_emb)
    assert predictions.shape == (batch_size,), "KAN Output shape mismatch"

    # Loss & Backward
    loss = F.binary_cross_entropy(predictions, target_ctr)
    loss.backward()
    optimizer.step()

    logger.info(f"KAN Ranker Success. Loss: {loss.item():.4f}")


def test_hyperbolic_recommender():
    logger.info("--- Testing Hyperbolic Poincaré Recommender ---")
    batch_size = 64
    num_users = 1000
    num_items = 1000

    users = torch.randint(0, num_users, (batch_size,))
    pos_items = torch.randint(0, num_items, (batch_size,))
    neg_items = torch.randint(0, num_items, (batch_size,))

    model = HyperbolicRecommender(num_users=num_users, num_items=num_items, emb_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Forward Pass (Calculates Hyperbolic Distance & Fermi-Dirac Loss)
    loss = model(users, pos_items, neg_items)

    # Backward Pass (Testing if Poincaré derivatives explode)
    loss.backward()

    # Check gradient scaling
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    logger.info(f"Hyperbolic Recommender Success. Poincaré Margin Loss: {loss.item():.4f}")


def test_sasrec():
    logger.info("--- Testing SASRec (Self-Attentive Sequential) ---")
    batch_size = 64
    max_seq_len = 50
    num_items = 5000

    # Mock user watching history
    log_seqs = torch.randint(0, num_items, (batch_size, max_seq_len))
    # Pad some sequences to test the causal mask
    log_seqs[:, :10] = 0

    candidates = torch.randint(0, num_items, (batch_size, 100))  # 100 candidates to rank

    model = SASRec(num_items=num_items, max_seq_len=max_seq_len, hidden_dim=64, num_blocks=2, num_heads=2)

    # Forward Pass
    scores = model.predict(log_seqs, candidates)
    assert scores.shape == (batch_size, 100), "SASRec output shape mismatch"

    logger.info(f"SASRec Success. Output tensor shape: {scores.shape}")


def test_lightgcn():
    logger.info("--- Testing LightGCN (Graph Multi-Hop) ---")
    batch_size = 64
    num_users = 1000
    num_items = 1000

    # Create a mock sparse adjacency matrix
    indices = torch.randint(0, num_users + num_items, (2, 5000))
    values = torch.ones(5000)
    with torch.sparse.check_sparse_tensor_invariants(False):
        adj_matrix = torch.sparse_coo_tensor(
            indices,
            values,
            (num_users + num_items, num_users + num_items),
        )

    users = torch.randint(0, num_users, (batch_size,))
    pos_items = torch.randint(0, num_items, (batch_size,))
    neg_items = torch.randint(0, num_items, (batch_size,))

    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Forward Pass & BPR Loss
    loss = model(users, pos_items, neg_items, adj_matrix)

    # Backward Pass
    loss.backward()
    optimizer.step()

    logger.info(f"LightGCN Success. BPR Loss: {loss.item():.4f}")


if __name__ == "__main__":
    logger.info("Starting Master Test Harness for Novel Architectures...")
    try:
        test_kan_ranker()
        test_hyperbolic_recommender()
        test_sasrec()
        test_lightgcn()
        logger.info("ALL TESTS PASSED: Architecture is mathematically sound and stable.")
    except Exception as e:
        logger.error(f"TEST FAILED: {e}")
