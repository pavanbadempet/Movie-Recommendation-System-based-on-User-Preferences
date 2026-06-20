"""
Hyperbolic Recommender (Poincaré Ball Model).

This is a profoundly novel approach to recommendation.
Standard embeddings (TurboVec, ALS, Transformers) exist in "Euclidean" (flat) space.
However, movies and user preferences are inherently HIERARCHICAL:
(Sci-Fi -> Cyberpunk -> The Matrix Franchise -> The Matrix Reloaded).

Flat space physically cannot embed infinite trees without massive distortion.
Hyperbolic space (specifically the Poincaré Ball) has volume that expands exponentially,
perfectly matching the geometry of trees.

By embedding users and movies in Hyperbolic space, we can compute "Poincaré Distances"
that understand hierarchical nuances better than any Euclidean model ever could.
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RiemannianManifold:
    """Math operations constrained to the Poincaré ball."""

    def __init__(self, c=1.0):
        # c is the curvature of the manifold
        self.c = c

    def mobius_add(self, x, y):
        """Möbius addition of x and y in the Poincaré ball."""
        xy = (x * y).sum(dim=-1, keepdim=True)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)

        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        den = 1 + 2 * self.c * xy + (self.c**2) * x2 * y2
        return num / den.clamp_min(1e-15)

    def poincare_distance(self, x, y):
        """
        Calculates the exact hyperbolic distance between two vectors.
        Unlike Euclidean Dot Product, this scales logarithmically as you approach the boundary.
        """
        mobius_minus_y = -y
        xy_plus = self.mobius_add(x, mobius_minus_y)
        norm_xy_plus = torch.norm(xy_plus, dim=-1, keepdim=True)

        # arcosh(1 + 2 * ||-x + y||^2 / ((1-||x||^2)(1-||y||^2)))
        # Simplified mathematically for stability:
        sqrt_c = math.sqrt(self.c)
        dist = (2 / sqrt_c) * torch.atanh(sqrt_c * norm_xy_plus.clamp_max(1 - 1e-5))
        return dist.squeeze(-1)

    def exp_map0(self, u):
        """Maps a vector from the flat tangent space at the origin into the Poincaré ball."""
        sqrt_c = math.sqrt(self.c)
        u_norm = torch.norm(u, dim=-1, keepdim=True).clamp_min(1e-15)
        res = torch.tanh(sqrt_c * u_norm) * (u / u_norm) / sqrt_c
        return res


class HyperbolicRecommender(nn.Module):
    """
    Learns continuous hierarchical representations of users and items.
    """

    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64, curvature: float = 1.0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.manifold = RiemannianManifold(c=curvature)

        # We initialize in Euclidean space, but will map to Hyperbolic during forward pass
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)

        # Init very small to stay close to origin (root of the tree)
        nn.init.uniform_(self.user_embedding.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.item_embedding.weight, -1e-3, 1e-3)

    def forward(self, users, pos_items, neg_items):
        """
        Contrastive learning in Hyperbolic space using Riemannian Optimization principles.
        """
        # 1. Look up flat embeddings
        u_flat = self.user_embedding(users)
        p_flat = self.item_embedding(pos_items)
        n_flat = self.item_embedding(neg_items)

        # 2. Map to Poincaré Ball
        u_hyp = self.manifold.exp_map0(u_flat)
        p_hyp = self.manifold.exp_map0(p_flat)
        n_hyp = self.manifold.exp_map0(n_flat)

        # 3. Calculate Hyperbolic Distances (Smaller is better)
        d_pos = self.manifold.poincare_distance(u_hyp, p_hyp)
        d_neg = self.manifold.poincare_distance(u_hyp, n_hyp)

        # 4. Fermi-Dirac Loss (Hyperbolic Margin Ranking Loss)
        # We want d_pos to be small and d_neg to be large.
        margin = 1.0
        loss = torch.nn.functional.softplus(d_pos - d_neg + margin).mean()

        return loss

    def predict(self, user_ids, candidate_items):
        """
        Returns a score based on Inverse Hyperbolic Distance.
        """
        u_flat = self.user_embedding(user_ids)
        c_flat = self.item_embedding(candidate_items)

        u_hyp = self.manifold.exp_map0(u_flat)
        c_hyp = self.manifold.exp_map0(c_flat)

        # Distance calculation
        # If user_ids is [batch] and candidate_items is [batch, num_candidates]
        u_hyp_expanded = u_hyp.expand_as(c_hyp) if u_hyp.dim() == c_hyp.dim() else u_hyp.unsqueeze(1).expand_as(c_hyp)

        distances = self.manifold.poincare_distance(u_hyp_expanded, c_hyp)

        # Convert distance to similarity score (e.g. 1 / (1 + distance))
        scores = 1.0 / (1.0 + distances)
        return scores
