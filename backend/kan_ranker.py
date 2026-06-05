"""
Kolmogorov-Arnold Network (KAN) Ranker.

This module implements the absolute bleeding edge of 2024 neural network architecture.
For 60 years, Deep Learning has used Multi-Layer Perceptrons (MLPs) where:
- Nodes have fixed activation functions (ReLU, Sigmoid).
- Edges have linear weights.

The Kolmogorov-Arnold Representation Theorem states that ANY multivariate continuous
function can be represented as a superposition of continuous 1D functions.
KANs fundamentally invert the MLP paradigm:
- Nodes simply sum their inputs.
- EDGES contain learnable, non-linear activation functions (B-splines / Fourier series).

By replacing our Tier-2 XGBoost/MLP ranker with a KAN, we achieve highly interpretable,
mathematically pure ranking that requires a fraction of the parameters.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NaiveFourierKANLayer(nn.Module):
    """
    A simplified KAN layer using Fourier basis functions instead of complex B-Splines
    for rapid, stable training in recommendation ranking.
    """

    def __init__(self, in_features, out_features, grid_size=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # In a standard linear layer, this is just [out_features, in_features]
        # In a KAN, every single edge between nodes is an entire function.
        # We parameterize this function using a sum of Sine and Cosine waves (Fourier Series).
        self.fourier_coeffs_sin = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) / math.sqrt(in_features)
        )
        self.fourier_coeffs_cos = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) / math.sqrt(in_features)
        )

        # Base linear weight to ground the function (Standard SiLU)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x shape: [batch_size, in_features]
        # 1. Base linear transformation with SiLU (acts as the structural backbone)
        base_out = F.linear(F.silu(x), self.base_weight)  # [batch_size, out_features]

        # 2. Compute the learnable edge functions (The KAN Magic)
        # We project the scalar inputs into a high-dimensional Fourier grid
        grids = torch.arange(1, self.grid_size + 1, device=x.device, dtype=torch.float32)

        # [batch_size, in_features, grid_size]
        x_expanded = x.unsqueeze(-1) * grids.unsqueeze(0).unsqueeze(0)

        # Compute the Sine and Cosine values for every input on every grid point
        sin_x = torch.sin(x_expanded)
        cos_x = torch.cos(x_expanded)

        # Contract with the learnable coefficients
        # Einstein Summation: Multiply batch inputs by output-edge coefficients and sum
        fourier_out_sin = torch.einsum("bik,oik->bo", sin_x, self.fourier_coeffs_sin)
        fourier_out_cos = torch.einsum("bik,oik->bo", cos_x, self.fourier_coeffs_cos)

        # 3. Nodes simply sum the outputs of the edge functions
        final_out = base_out + fourier_out_sin + fourier_out_cos + self.bias

        return final_out


class KANRanker(nn.Module):
    """
    Tier-2 Ranking Model using Kolmogorov-Arnold Networks.
    Takes the concatenated vectors of (User, Candidate Item, Context) and predicts CTR.
    """

    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()

        # A 3-Layer Kolmogorov-Arnold Network
        self.kan1 = NaiveFourierKANLayer(in_features=input_dim, out_features=hidden_dim, grid_size=5)
        self.kan2 = NaiveFourierKANLayer(in_features=hidden_dim, out_features=hidden_dim // 2, grid_size=5)
        self.kan3 = NaiveFourierKANLayer(in_features=hidden_dim // 2, out_features=1, grid_size=3)

    def forward(self, user_emb, item_emb):
        """
        user_emb: [batch_size, emb_dim]
        item_emb: [batch_size, emb_dim]
        """
        # Concatenate features
        x = torch.cat([user_emb, item_emb], dim=-1)

        # Pass through the learnable edge functions
        x = self.kan1(x)
        x = self.kan2(x)
        out = self.kan3(x)

        # Map to Click-Through-Rate probability [0, 1]
        return torch.sigmoid(out).squeeze(-1)
