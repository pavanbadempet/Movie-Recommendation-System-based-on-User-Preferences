"""
Multi-gate Mixture-of-Experts (MMoE) Ranker.

This architecture is heavily inspired by YouTube's production ranking systems.
Instead of a single dense layer trying to balance competing objectives (e.g., maximizing
clicks vs maximizing watch time vs maximizing 5-star ratings), MMoE routes inputs
through multiple "Expert" networks. Task-specific "Gates" then learn which experts
are most useful for their specific objective.

This mitigates the "seesaw effect" in multi-task learning where improving one metric
degrades another.

Features included:
- 4 Expert Networks (Deep MLPs)
- 3 Task Gates (Click, Watch Time, Satisfaction)
- Position Bias Shallow Tower (to correct for users blindly clicking the top result)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models")

class Expert(nn.Module):
    """A single expert network."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)

class MMoERanker(nn.Module):
    def __init__(
        self, 
        user_vocab_size: int, 
        item_vocab_size: int, 
        emb_dim: int = 16, 
        num_experts: int = 4, 
        expert_hidden_dim: int = 64,
        expert_out_dim: int = 32
    ):
        super().__init__()
        
        # Base embeddings
        self.user_emb = nn.Embedding(user_vocab_size, emb_dim)
        self.item_emb = nn.Embedding(item_vocab_size, emb_dim)
        
        # We will concatenate user_emb, item_emb, and a synthetic context vector 
        # (e.g., time of day, device type) -> For now, input_dim = emb_dim * 2
        input_dim = emb_dim * 2
        
        # 1. Experts
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim, expert_out_dim) 
            for _ in range(num_experts)
        ])
        
        # 2. Gates (One per task)
        # Task 1: CTR (Click-Through Rate)
        # Task 2: Watch Time
        # Task 3: Satisfaction (Rating >= 4.0)
        self.gate_ctr = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        self.gate_watch = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        self.gate_sat = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        # 3. Task Towers (Final specific predictions)
        self.tower_ctr = nn.Sequential(
            nn.Linear(expert_out_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Probability
        )
        
        self.tower_watch = nn.Sequential(
            nn.Linear(expert_out_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Continuous value (normalized)
        )
        
        self.tower_sat = nn.Sequential(
            nn.Linear(expert_out_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Probability
        )
        
        # 4. Position Bias Shallow Tower (Debiasing)
        # Learns the probability of a click based solely on position, ignoring item relevance
        self.position_bias = nn.Sequential(
            nn.Embedding(500, 4), # Max 500 positions
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids, position_ids=None):
        """
        Forward pass for MMoE.
        If position_ids is provided, it incorporates position bias (used in training).
        If position_ids is None, it disables position bias (used in serving).
        """
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        
        # Concatenate features
        x = torch.cat([u, i], dim=1)
        
        # Pass through all experts -> Shape: [batch_size, num_experts, expert_out_dim]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Calculate gate weights -> Shape: [batch_size, num_experts]
        ctr_weights = self.gate_ctr(x).unsqueeze(-1)
        watch_weights = self.gate_watch(x).unsqueeze(-1)
        sat_weights = self.gate_sat(x).unsqueeze(-1)
        
        # Blend expert outputs based on gate weights -> Shape: [batch_size, expert_out_dim]
        ctr_blend = (expert_outputs * ctr_weights).sum(dim=1)
        watch_blend = (expert_outputs * watch_weights).sum(dim=1)
        sat_blend = (expert_outputs * sat_weights).sum(dim=1)
        
        # Final Task Predictions
        pred_ctr = self.tower_ctr(ctr_blend).squeeze(-1)
        pred_watch = self.tower_watch(watch_blend).squeeze(-1)
        pred_sat = self.tower_sat(sat_blend).squeeze(-1)
        
        # Inject position bias during training for the CTR task
        if position_ids is not None:
            pos_bias = self.position_bias(position_ids).squeeze(-1)
            # The actual CTR is the relevance * position bias
            pred_ctr = pred_ctr * pos_bias
            
        return pred_ctr, pred_watch, pred_sat

# Singleton Instance
_mmoe_ranker = None

def get_mmoe_ranker(num_users: int = 1000, num_items: int = 10000) -> MMoERanker:
    global _mmoe_ranker
    if _mmoe_ranker is None:
        _mmoe_ranker = MMoERanker(user_vocab_size=num_users, item_vocab_size=num_items)
        
        # Load weights
        weight_path = MODELS_DIR / "mmoe_ranker.pth"
        if weight_path.exists():
            try:
                _mmoe_ranker.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
                logger.info("Loaded MMoE Multi-Task Ranker weights.")
            except Exception as e:
                logger.error(f"Failed to load MMoE weights: {e}")
                
        _mmoe_ranker.eval()
    return _mmoe_ranker
