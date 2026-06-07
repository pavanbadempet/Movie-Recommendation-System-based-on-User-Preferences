import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ContextualRouter(nn.Module):
    """
    Dynamic Contextual Router (Mixture of Experts) for recommendation models.
    Routes user recommendation queries dynamically to the top-k most suited models
    out of the 6 ensemble models based on user profile and context.
    """
    def __init__(self, emb_dim: int = 16, hidden_dims: List[int] = [64, 32], num_models: int = 6):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_models = num_models
        
        # Input state vector dimension: emb_dim + 4
        # (user embedding + 4 contextual metrics: interactions, session len, stability, energy)
        input_dim = emb_dim + 4
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, num_models))
        self.mlp = nn.Sequential(*layers)
        
        # Consistent ordering of the 6 models in the ensemble
        self.model_names = ["lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion"]
        
    def forward(self, user_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute routing logits for each model.
        
        Args:
            user_state: Tensor of shape [batch_size, emb_dim + 4] or [emb_dim + 4]
            
        Returns:
            Logits of shape [batch_size, num_models] or [num_models]
        """
        is_batched = user_state.dim() > 1
        if not is_batched:
            user_state = user_state.unsqueeze(0)
            
        logits = self.mlp(user_state)
        
        if not is_batched:
            logits = logits.squeeze(0)
        return logits
        
    def route(self, user_state: torch.Tensor, k: int = 2) -> Tuple[List[str], torch.Tensor]:
        """
        Dynamically routes user context to the top-k models.
        
        Args:
            user_state: Tensor of shape [emb_dim + 4]
            k: number of models to select (1 <= k <= 6)
            
        Returns:
            Tuple of:
            - list of selected model names (length k)
            - normalized routing weights for selected models (tensor of shape [k])
        """
        self.eval()
        k = min(max(k, 1), self.num_models)
        
        with torch.no_grad():
            logits = self.forward(user_state)
            
            # Apply softmax to get routing probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k indices and values
            top_probs, top_indices = torch.topk(probs, k=k, dim=-1)
            
            # Normalize top-k probabilities to sum to 1.0
            sum_probs = top_probs.sum()
            if sum_probs > 1e-6:
                normalized_weights = top_probs / sum_probs
            else:
                normalized_weights = torch.ones_like(top_probs) / k
                
            selected_models = [self.model_names[idx.item()] for idx in top_indices]
            
        return selected_models, normalized_weights

    def train_router_step(
        self, user_state: torch.Tensor, model_losses: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Performs a single training step to align routing probabilities with model losses.
        Uses Teacher-Student Loss Alignment where models with lower loss get higher probability.
        
        Args:
            user_state: Tensor of shape [batch_size, emb_dim + 4] or [emb_dim + 4]
            model_losses: Tensor of shape [batch_size, num_models] or [num_models] (loss of each model)
            optimizer: PyTorch optimizer
            
        Returns:
            Loss value (KL divergence)
        """
        self.train()
        
        # Handle 1D inputs by unsqueezing to add batch dimension
        if user_state.dim() == 1:
            user_state = user_state.unsqueeze(0)
        if model_losses.dim() == 1:
            model_losses = model_losses.unsqueeze(0)
            
        optimizer.zero_grad()
        
        logits = self.forward(user_state) # [batch_size, num_models]
        
        # Softmax over negative losses (lower loss -> higher probability)
        # Using temperature = 1.0 as default scaling
        targets = F.softmax(-model_losses, dim=-1)
        
        # KL Divergence loss
        router_log_probs = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(router_log_probs, targets, reduction='batchmean')
        
        loss.backward()
        optimizer.step()
        
        return loss.item()


def build_user_state(
    user_id: int,
    user_emb: torch.Tensor,
    session_seq: torch.Tensor,
    item_embeddings: torch.Tensor | None = None,
    interaction_count: int | None = None,
    inference_energy: float = 0.5,
) -> torch.Tensor:
    """
    Constructs the contextual user state vector of shape [emb_dim + 4]
    consisting of the user's base embedding and 4 profile metrics:
    1. Interaction count (normalized)
    2. Session length (normalized)
    3. Preference stability (average cosine similarity of consecutive active items)
    4. Active inference energy (current system/load indicator)
    """
    # 1. Interaction count (normalized using soft log scale)
    if interaction_count is None:
        interaction_count = 0
    norm_interaction_count = float(np.log1p(interaction_count) / 10.0)
    norm_interaction_count = min(max(norm_interaction_count, 0.0), 1.0)
    
    # 2. Session length (normalized by max sequence length, usually 50)
    seq = session_seq.squeeze()
    non_zero = int((seq > 0).sum().item())
    norm_session_length = float(non_zero / len(seq)) if len(seq) > 0 else 0.0
    
    # 3. Preference stability
    stability = 1.0
    if item_embeddings is not None and non_zero >= 2:
        active_items = seq[seq > 0]
        try:
            num_items = item_embeddings.shape[0]
            active_items = torch.clamp(active_items, 0, num_items - 1)
            embs = item_embeddings[active_items]
            
            # Compute cosine similarity of consecutive items
            embs_norm = F.normalize(embs, p=2, dim=-1)
            similarities = (embs_norm[:-1] * embs_norm[1:]).sum(dim=-1)
            mean_sim = float(similarities.mean().item())
            
            # Map similarity from [-1, 1] to [0, 1]
            stability = (mean_sim + 1.0) / 2.0
            if np.isnan(stability):
                stability = 1.0
        except Exception as exc:
            logger.debug("Failed to calculate preference stability: %s", exc)
            stability = 1.0
            
    # 4. Inference energy
    norm_inference_energy = min(max(inference_energy, 0.0), 1.0)
    
    # Pack metrics into tensor
    metrics = torch.tensor([
        norm_interaction_count,
        norm_session_length,
        stability,
        norm_inference_energy
    ], dtype=torch.float32, device=user_emb.device)
    
    return torch.cat([user_emb, metrics], dim=-1)
