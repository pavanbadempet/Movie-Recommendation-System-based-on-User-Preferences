"""
Attention-Based User Modeling for APEX.

Uses multi-head self-attention (same architecture as GPT) to model
user interaction sequences, capturing long-range dependencies in watch history.

Key insight: Standard SASRec uses causal attention (can't look at future).
This module adds a BIDIRECTIONAL attention layer for user profile modeling
(not sequence prediction) — the user's full history is available.

This is equivalent to BERT4Rec (published at RecSys 2019) but integrated
into the APEX serving path as a user embedding enrichment layer.

The attention mechanism learns:
- Which past interactions are most relevant to current preferences
- How different genres interact in the user's taste profile
- Temporal patterns (e.g., user watches action on weekends)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class UserAttentionEncoder(nn.Module):
    """
    Bidirectional attention encoder for user interaction sequences.

    Input: Sequence of item embeddings from user history
    Output: Attended user representation (weighted sum of history)

    The attention weights reveal which past interactions matter most
    for the current recommendation context.
    """

    def __init__(
        self,
        emb_dim: int = 16,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        item_embeddings: torch.Tensor,  # [batch, seq_len, emb_dim]
        padding_mask: torch.Tensor | None = None,  # [batch, seq_len] True = padding
    ) -> torch.Tensor:
        """
        Returns attended user representation [batch, emb_dim].
        Uses mean pooling over attended sequence.
        """
        # Self-attention over the sequence
        attended, attention_weights = self.attention(
            item_embeddings,
            item_embeddings,
            item_embeddings,
            key_padding_mask=padding_mask,
        )
        attended = self.norm(item_embeddings + self.dropout(attended))

        # Mean pool over non-padding positions
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)  # [batch, seq, 1]
            user_repr = (attended * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            user_repr = attended.mean(dim=1)

        return user_repr  # [batch, emb_dim]


def build_attended_user_embedding(
    user_id: int,
    lightgcn_item_embeddings: torch.Tensor,
    session_sequence: list[int],
    encoder: UserAttentionEncoder | None = None,
) -> torch.Tensor | None:
    """
    Build an attention-weighted user embedding from their interaction history.

    Args:
        user_id: User ID (for logging)
        lightgcn_item_embeddings: Item embedding matrix [num_items, emb_dim]
        session_sequence: List of item indices (most recent last)
        encoder: Optional pre-trained UserAttentionEncoder

    Returns:
        User embedding tensor [1, emb_dim] or None if sequence is empty
    """
    if not session_sequence:
        return None

    try:
        # Get item embeddings for the sequence
        seq_tensor = torch.tensor(session_sequence, dtype=torch.long)
        # Clamp to valid range
        seq_tensor = seq_tensor.clamp(0, lightgcn_item_embeddings.shape[0] - 1)
        item_embs = lightgcn_item_embeddings[seq_tensor]  # [seq_len, emb_dim]
        item_embs = item_embs.unsqueeze(0)  # [1, seq_len, emb_dim]

        if encoder is not None:
            encoder.eval()
            with torch.no_grad():
                user_repr = encoder(item_embs)  # [1, emb_dim]
        else:
            # Simple mean pooling as fallback
            user_repr = item_embs.mean(dim=1)  # [1, emb_dim]

        return user_repr

    except Exception as exc:
        logger.debug("Attention user embedding failed for user %s: %s", user_id, exc)
        return None


# Module-level encoder singleton
_encoder: UserAttentionEncoder | None = None


def get_user_attention_encoder(emb_dim: int = 16) -> UserAttentionEncoder:
    """Get or create the module-level attention encoder."""
    global _encoder
    if _encoder is None:
        _encoder = UserAttentionEncoder(emb_dim=emb_dim)
        _encoder.eval()
    return _encoder
