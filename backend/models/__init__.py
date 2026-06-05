"""
ML model implementations for the APEX ensemble.

Models:
    LightGCN              — Graph collaborative filtering (backend/lightgcn.py)
    SASRec                — Sequential Transformer (backend/sasrec.py)
    KANRanker             — Kolmogorov-Arnold Network (backend/kan_ranker.py)
    QuantumFluidRecommender — Neural ODE + complex embeddings (backend/neural_ode_recommender.py)
    HyperbolicRecommender — Poincaré ball manifold (backend/hyperbolic_recommender.py)
    LatentDiffusionRecommender — Generative DDPM (backend/diffusion_recommender.py)
    TwoTowerModel         — Dual-encoder retrieval (backend/two_tower.py)
    MMoERanker            — Multi-gate Mixture-of-Experts (backend/mmoe_ranker.py)
    ActorCriticPolicy     — A2C reinforcement learning (backend/rl_policy.py)

Note: Models live in the parent backend/ directory for backward compatibility.
This sub-package provides a logical grouping and documentation anchor.
"""

from backend.diffusion_recommender import LatentDiffusionRecommender
from backend.hyperbolic_recommender import HyperbolicRecommender
from backend.kan_ranker import KANRanker
from backend.lightgcn import LightGCN
from backend.mmoe_ranker import MMoERanker
from backend.neural_ode_recommender import QuantumFluidRecommender
from backend.rl_policy import ActorCriticPolicy
from backend.sasrec import SASRec
from backend.two_tower import TwoTowerModel

__all__ = [
    "LightGCN",
    "SASRec",
    "KANRanker",
    "QuantumFluidRecommender",
    "HyperbolicRecommender",
    "LatentDiffusionRecommender",
    "TwoTowerModel",
    "MMoERanker",
    "ActorCriticPolicy",
]
