"""
Two-Tower Candidate Generation Model

The foundation of every modern recommendation system (Netflix, YouTube, Amazon).
Two separate neural networks (towers) independently encode users and items into
a shared embedding space. At inference time, item embeddings are pre-computed
and indexed in FAISS for O(log n) retrieval. Only the user tower runs live.

Architecture:
    User Tower: user_features → MLP → 128d embedding
    Item Tower: item_features → MLP → 128d embedding
    Score: dot_product(user_emb, item_emb)

Training:
    Sampled softmax / InfoNCE contrastive loss with hard negatives.
    Positive pairs: (user, item) from ratings >= 3.5
    Hard negatives: random items + popularity-weighted sampling

References:
    - Covington et al. "Deep Neural Networks for YouTube Recommendations" (RecSys 2016)
    - Yi et al. "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (RecSys 2019)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UserTower(nn.Module):
    """
    Encodes user features into a dense embedding.

    Input features (concatenated):
      - ALS user embedding (16d from PySpark Gold layer)
      - User activity features: total_ratings, avg_rating (2d)
    Total input: 18d → MLP → 128d output
    """

    def __init__(self, input_dim: int = 18, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.net(x)
        # L2 normalize for cosine similarity via dot product
        return F.normalize(emb, p=2, dim=-1)


class ItemTower(nn.Module):
    """
    Encodes item features into a dense embedding.

    Input features (concatenated):
      - ALS item embedding (16d from PySpark Gold layer)
      - Item metadata: vote_average, vote_count_log, popularity_log, num_genres (4d)
    Total input: 20d → MLP → 128d output
    """

    def __init__(self, input_dim: int = 20, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.net(x)
        return F.normalize(emb, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Full Two-Tower model for candidate generation.

    Forward pass returns dot-product scores between user and item embeddings.
    During inference, item embeddings are pre-computed and stored in FAISS.
    Only the user tower runs live per request.
    """

    def __init__(
        self,
        user_input_dim: int = 18,
        item_input_dim: int = 20,
        embedding_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.user_tower = UserTower(user_input_dim, embedding_dim)
        self.item_tower = ItemTower(item_input_dim, embedding_dim)
        self.temperature = temperature
        self.embedding_dim = embedding_dim

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            user_features: [batch_size, user_input_dim]
            item_features: [batch_size, item_input_dim]

        Returns:
            scores: [batch_size] dot-product similarity scores
        """
        user_emb = self.user_tower(user_features)  # [B, D]
        item_emb = self.item_tower(item_features)  # [B, D]
        # Dot product similarity (cosine since both are L2-normalized)
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores

    def compute_contrastive_loss(
        self,
        user_features: torch.Tensor,
        pos_item_features: torch.Tensor,
        neg_item_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss (same as CLIP, SimCLR).

        For each user, we have 1 positive item and K negative items.
        The model must assign the highest score to the positive.

        Args:
            user_features:     [B, user_dim]
            pos_item_features: [B, item_dim]     (1 positive per user)
            neg_item_features: [B, K, item_dim]  (K negatives per user)
        """
        batch_size = user_features.shape[0]
        num_negatives = neg_item_features.shape[1]

        user_emb = self.user_tower(user_features)  # [B, D]
        pos_emb = self.item_tower(pos_item_features)  # [B, D]

        # Reshape negatives: [B*K, item_dim] → [B*K, D] → [B, K, D]
        neg_flat = neg_item_features.reshape(-1, neg_item_features.shape[-1])
        neg_emb = self.item_tower(neg_flat).reshape(batch_size, num_negatives, -1)

        # Positive scores: [B, 1]
        pos_scores = (user_emb * pos_emb).sum(dim=-1, keepdim=True) / self.temperature

        # Negative scores: [B, K]
        neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1) / self.temperature

        # InfoNCE: log_softmax over [positive, negatives]
        # Labels: positive is always at index 0
        logits = torch.cat([pos_scores, neg_scores], dim=-1)  # [B, 1+K]
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def encode_users(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user features into embeddings (for inference)."""
        self.eval()
        with torch.no_grad():
            return self.user_tower(user_features)

    def encode_items(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features into embeddings (for FAISS indexing)."""
        self.eval()
        with torch.no_grad():
            return self.item_tower(item_features)
