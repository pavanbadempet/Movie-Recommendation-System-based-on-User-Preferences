"""
Online learning and reinforcement learning sub-package for APEX.

Implements the closed online learning loop — all three highest-weighted
ensemble models receive incremental gradient updates from live events.

Online Learning (Tier1 only):
    OnlineLearner               — LightGCN BPR embedding updates
    SASRecOnlineLearner         — SASRec attention + item embedding fine-tuning
    KANOnlineLearner            — KAN Fourier coefficient updates
    OnlineLearningCoordinator   — Unified fan-out coordinator (all three above)

Reinforcement Learning:
    ActorCriticPolicy           — A2C policy network (backend.models)
    RLRewardEngine              — Conservative Q-Learning (CQL) reward shaping
    NeuralWeightOptimizer       — Context-dependent ensemble weight adaptation

Environment:
    NOVA_ONLINE_LEARNING_ENABLED=1   — Enable/disable online learning at runtime
"""

from backend.learning.kan_online_learner import KANOnlineLearner
from backend.learning.online_learner import OnlineLearner
from backend.learning.online_learning_coordinator import OnlineLearningCoordinator
from backend.learning.rl_policy import ActorCriticPolicy, RLSafetyFilter
from backend.learning.rl_reward import RLRewardEngine
from backend.learning.sasrec_online_learner import SASRecOnlineLearner

__all__ = [
    # Online learning
    "OnlineLearner",
    "SASRecOnlineLearner",
    "KANOnlineLearner",
    "OnlineLearningCoordinator",
    # RL
    "ActorCriticPolicy",
    "RLSafetyFilter",
    "RLRewardEngine",
]
