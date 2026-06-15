"""
Self-Attentive Sequential Recommendation (SASRec).

This module implements the absolute bleeding edge of recommendation retrieval.
While ALS captures static preferences and LightGCN captures multi-hop graphs,
they both ignore TIME. If a user watches Harry Potter 1 and then Harry Potter 2,
Matrix Factorization doesn't know what order they watched them in.

SASRec uses a pure Transformer architecture (Multi-Head Self-Attention)
exactly like ChatGPT, but instead of predicting the next word in a sentence,
it predicts the next movie in a user's chronological watch history.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SASRec(nn.Module):
    """
    Transformer-based Sequential Recommender.
    Models the user's exact chronological journey to predict their absolute next action.
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        # Item embeddings (index 0 is reserved for padding)
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        # Positional embeddings (injects the concept of chronological TIME)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # Build Transformer Blocks
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout_rate,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, log_seqs):
        """
        log_seqs: A tensor of shape [batch_size, max_seq_len] containing the user's historical movie interactions.
        """
        batch_size, seq_len = log_seqs.shape

        # 1. Look up embeddings for the movies in the sequence
        seqs = self.item_emb(log_seqs)

        # 2. Inject Positional Encodings (so the model knows which movie was watched recently vs long ago)
        positions = torch.arange(seq_len, dtype=torch.long, device=log_seqs.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        seqs = self.emb_dropout(seqs + self.pos_emb(positions))

        # Create causality mask so the Transformer cannot "look into the future" to predict the present
        attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=log_seqs.device, dtype=torch.bool), 1)

        # 3. Pass through Transformer Blocks
        for blk in self.blocks:
            seqs = blk(seqs, src_mask=attention_mask, is_causal=False)

        # 4. The final output tensor represents the user's "Sequential Intent" at the current moment
        return self.norm(seqs)

    def predict(self, log_seqs, candidate_items):
        """
        Calculates the probability score of candidate items being the absolute next interaction.
        """
        # Get the sequence embeddings
        seq_out = self.forward(log_seqs)

        # We only care about the very last state in the sequence (the absolute present)
        final_state = seq_out[:, -1, :]  # Shape: [batch_size, hidden_dim]

        # Get embeddings for the candidates we want to rank
        candidate_embs = self.item_emb(candidate_items)  # Shape: [batch_size, num_candidates, hidden_dim]

        # Fast vectorized Dot Product between current state and candidate movies
        scores = (final_state.unsqueeze(1) * candidate_embs).sum(dim=-1)
        return scores
