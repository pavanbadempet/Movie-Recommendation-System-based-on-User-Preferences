import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ActorCriticPolicy(nn.Module):
    """
    A2C (Advantage Actor-Critic) Neural Network for Long-Term Recommendation.
    - Actor: Decides which genre/cluster of movies to heavily weight (Action space).
    - Critic: Estimates the expected long-term Return (Value function) for the given state.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCriticPolicy, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head: Outputs a probability distribution over continuous action weights
        # We model the action as a shift vector to apply to the item embeddings
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head: Predicts the scalar Value V(s) - expected long-term return
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        features = self.shared(state)
        
        # Value estimation
        value = self.critic(features)
        
        # Action distribution
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        
        return action_mean, action_std, value
        
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Samples an action shift vector from the policy."""
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            return action_mean, value
            
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value

class RLSafetyFilter:
    """
    Ensures the RL agent does not violate critical product rules 
    (e.g. recommending explicitly disliked content).
    """
    @staticmethod
    def apply_hard_constraints(candidate_ids: list[int], user_dislikes: set[int]) -> list[int]:
        """Filters out items the user has previously rated < 2.5."""
        safe_candidates = [cid for cid in candidate_ids if cid not in user_dislikes]
        if not safe_candidates:
            logger.warning("RL Safety Filter blocked all candidates. Reverting to fallback.")
            return candidate_ids[:5] # Fallback to prevent empty states
        return safe_candidates
