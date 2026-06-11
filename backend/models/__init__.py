"""
ML model implementations for the APEX ensemble.

Models:
    LightGCN              — Graph collaborative filtering (backend/models/lightgcn.py)
    SASRec                — Sequential Transformer (backend/models/sasrec.py)
    KANRanker             — Kolmogorov-Arnold Network (backend/models/kan_ranker.py)
    QuantumFluidRecommender — Neural ODE + complex embeddings (backend/models/neural_ode_recommender.py)
    HyperbolicRecommender — Poincaré ball manifold (backend/models/hyperbolic_recommender.py)
    LatentDiffusionRecommender — Generative DDPM (backend/models/diffusion_recommender.py)
    TwoTowerModel         — Dual-encoder retrieval (backend/models/two_tower.py)
    MMoERanker            — Multi-gate Mixture-of-Experts (backend/models/mmoe_ranker.py)
    ActorCriticPolicy     — A2C reinforcement learning (backend/learning/rl_policy.py)

Training & optimization:
    ApexEnsembleEngine    — 6-model weighted ensemble engine (backend/models/ensemble_engine.py)
    ContextualWeightNetwork — Context-dependent ensemble weights (backend/models/neural_weight_optimizer.py)
    OnlineLearner         — LightGCN online BPR updates (backend/learning/online_learner.py)
"""

from backend.learning.online_learner import OnlineLearner
from backend.learning.rl_policy import ActorCriticPolicy, RLSafetyFilter
from backend.learning.rl_reward import RLRewardEngine
from backend.models.contextual_router import ContextualRouter, build_user_state
from backend.models.diffusion_recommender import LatentDiffusionRecommender
from backend.models.ensemble_engine import ApexEnsembleEngine, get_apex_engine
from backend.models.hyperbolic_recommender import HyperbolicRecommender
from backend.models.kan_ranker import KANRanker
from backend.models.lightgcn import LightGCN
from backend.models.mmoe_ranker import MMoERanker
from backend.models.neural_ode_recommender import QuantumFluidRecommender
from backend.models.neural_weight_optimizer import ContextualWeightNetwork, get_contextual_weights
from backend.models.sasrec import SASRec
from backend.models.two_tower import TwoTowerModel

__all__ = [
    # 6-model ensemble
    "LightGCN",
    "SASRec",
    "KANRanker",
    "QuantumFluidRecommender",
    "HyperbolicRecommender",
    "LatentDiffusionRecommender",
    "ContextualRouter",
    "build_user_state",
    # Retrieval + ranking
    "TwoTowerModel",
    "MMoERanker",
    # RL
    "ActorCriticPolicy",
    "RLSafetyFilter",
    "RLRewardEngine",
    # Ensemble engine
    "ApexEnsembleEngine",
    "get_apex_engine",
    # Contextual weights
    "ContextualWeightNetwork",
    "get_contextual_weights",
    # Online learning (base)
    "OnlineLearner",
]
