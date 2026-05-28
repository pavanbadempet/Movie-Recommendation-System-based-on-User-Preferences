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

import math
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class PointWiseFeedForward(nn.Module):
    """A two-layer feed-forward network applied to each sequence position."""
    def __init__(self, hidden_dim: int, dropout_rate: float):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # inputs shape: [batch_size, seq_len, hidden_dim]
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        return outputs.transpose(-1, -2) + inputs # Residual connection

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
        dropout_rate: float = 0.2
    ):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        # Item embeddings (index 0 is reserved for padding)
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        # Positional embeddings (injects the concept of chronological TIME)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # Build Transformer Blocks
        self.attention_layernorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_blocks)])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
            for _ in range(num_blocks)
        ])
        
        self.forward_layernorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_blocks)])
        self.forward_layers = nn.ModuleList([
            PointWiseFeedForward(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])

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
        seqs += self.pos_emb(positions)
        
        seqs = self.emb_dropout(seqs)
        
        # Create causality mask so the Transformer cannot "look into the future" to predict the present
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(log_seqs.device)
        seqs *= ~timeline_mask.unsqueeze(-1) # Broadcast mask

        attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=log_seqs.device))

        # 3. Pass through Transformer Blocks
        for i in range(len(self.attention_layers)):
            # Self-Attention
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = seqs + mha_outputs # Residual connection
            
            # Point-wise Feed Forward
            seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            seqs *= ~timeline_mask.unsqueeze(-1)

        # 4. The final output tensor represents the user's "Sequential Intent" at the current moment
        return seqs

    def predict(self, log_seqs, candidate_items):
        """
        Calculates the probability score of candidate items being the absolute next interaction.
        """
        # Get the sequence embeddings
        seq_out = self.forward(log_seqs)
        
        # We only care about the very last state in the sequence (the absolute present)
        final_state = seq_out[:, -1, :] # Shape: [batch_size, hidden_dim]
        
        # Get embeddings for the candidates we want to rank
        candidate_embs = self.item_emb(candidate_items) # Shape: [batch_size, num_candidates, hidden_dim]
        
        # Fast vectorized Dot Product between current state and candidate movies
        scores = (final_state.unsqueeze(1) * candidate_embs).sum(dim=-1)
        return scores
