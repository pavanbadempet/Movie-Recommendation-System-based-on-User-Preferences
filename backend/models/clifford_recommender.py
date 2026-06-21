"""
Clifford Geometric Algebra Recommender (G-CAR).

This architecture embeds users and items in a 2D Clifford Algebra Cl(1, 1) space.
Cl(1, 1) has basis vectors e1, e2 such that e1^2 = 1, e2^2 = -1, and e1*e2 = -e2*e1.

A multivector A in Cl(1, 1) is represented as:
    A = a0 + a1*e1 + a2*e2 + a3*e12
where a0 is scalar, a1/a2 are vectors, and a3 is a bivector.

The interaction between user multivector U and item multivector V is modeled via
the Clifford Sandwich Product:
    S = Real(U * V * ~U)
where ~U is the reversal of U (negating the bivector component).
"""

import torch
import torch.nn as nn


class CliffordRecommender(nn.Module):
    """
    Clifford Geometric Algebra Recommender in Cl(1, 1).
    Embeds entities in multivector spaces and scores via Clifford Sandwich Product.
    """

    def __init__(self, num_users: int, num_items: int, emb_dim: int = 16):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim

        # Ensure embedding dimension is a multiple of 4 (for 4 multivector components)
        if emb_dim % 4 != 0:
            raise ValueError(f"Embedding dimension {emb_dim} must be a multiple of 4 for Cl(1, 1).")

        self.num_multivectors = emb_dim // 4

        # Initialise user and item embeddings in Euclidean space
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)

        # Init with small values to ensure stable gradients
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)

    def _sandwich_product(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the Real part of the sandwich product: ScalarPart(U * V * ~U)
        u: [..., d, 4]
        v: [..., d, 4]
        Returns: [...]
        """
        # Split multivector components
        u0, u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
        v0, v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]

        # 1. Compute Geometric Product P = U * V
        # p0 (scalar): u0*v0 + u1*v1 - u2*v2 + u3*v3
        p0 = u0 * v0 + u1 * v1 - u2 * v2 + u3 * v3
        # p1 (e1 vector): u0*v1 + u1*v0 + u2*v3 - u3*v2
        p1 = u0 * v1 + u1 * v0 + u2 * v3 - u3 * v2
        # p2 (e2 vector): u0*v2 + u2*v0 + u1*v3 - u3*v1
        p2 = u0 * v2 + u2 * v0 + u1 * v3 - u3 * v1
        # p3 (e12 bivector): u0*v3 + u3*v0 + u1*v2 - u2*v1
        p3 = u0 * v3 + u3 * v0 + u1 * v2 - u2 * v1

        # 2. Compute ScalarPart(P * ~U) where ~U = [u0, u1, u2, -u3]
        # Scalar part of product of P and ~U is: p0*u0 + p1*u1 - p2*u2 - p3*u3
        sandwich_scalar = p0 * u0 + p1 * u1 - p2 * u2 - p3 * u3

        # Sum over the independent multivectors
        return sandwich_scalar.sum(dim=-1)

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor) -> torch.Tensor:
        """
        Computes margin loss for BPR training.
        """
        # Look up embeddings
        u_emb = self.user_embedding(users)  # [batch, emb_dim]
        pos_emb = self.item_embedding(pos_items)  # [batch, emb_dim]
        neg_emb = self.item_embedding(neg_items)  # [batch, emb_dim]

        # Reshape to multivectors [batch, d, 4]
        u = u_emb.view(-1, self.num_multivectors, 4)
        pos = pos_emb.view(-1, self.num_multivectors, 4)
        neg = neg_emb.view(-1, self.num_multivectors, 4)

        # Compute sandwich scores
        pos_scores = self._sandwich_product(u, pos)
        neg_scores = self._sandwich_product(u, neg)

        # Margin loss
        loss = torch.clamp(1.0 - pos_scores + neg_scores, min=0.0).mean()
        return loss

    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Predicts interaction scores for given user and candidate items.
        Supports both 1D item_ids and 2D candidate item batches.
        """
        u_emb = self.user_embedding(user_ids)  # [batch, emb_dim] or [1, emb_dim]
        v_emb = self.item_embedding(item_ids)  # [batch, emb_dim] or [batch, candidates, emb_dim]

        if u_emb.dim() == 1:
            u_emb = u_emb.unsqueeze(0)
        if v_emb.dim() == 1:
            v_emb = v_emb.unsqueeze(0)

        # Broaden user embedding to match item layout
        if u_emb.dim() != v_emb.dim():
            # u_emb is [batch, emb_dim], v_emb is [batch, candidates, emb_dim]
            u_emb = u_emb.unsqueeze(1).expand_as(v_emb)

        # Reshape to [..., d, 4]
        u = u_emb.reshape(*u_emb.shape[:-1], self.num_multivectors, 4)
        v = v_emb.reshape(*v_emb.shape[:-1], self.num_multivectors, 4)

        scores = self._sandwich_product(u, v)
        return scores
