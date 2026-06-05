"""
LightGCN (Graph Convolutional Network) for Collaborative Filtering.

This module implements Tier 1C of the retrieval funnel.
Unlike ALS (Matrix Factorization) which only learns direct User-Item interactions,
LightGCN learns high-order, multi-hop connectivity (e.g., User A likes Movie B,
Movie B is liked by User C, User C likes Movie D -> Recommend D to A).

This is a PyTorch implementation designed for offline Kaggle training,
exporting final node embeddings to the Feature Store.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LightGCN(nn.Module):
    """
    State-of-the-art Graph Collaborative Filtering Model.
    Removes non-linear activation functions and feature transformations
    for massive scalability on bipartite User-Item graphs.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Initial embeddings (E^0)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Xavier Initialization for stable gradients
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def computer(self, adj_matrix):
        """
        Propagates embeddings over the graph structure.
        adj_matrix: A normalized sparse SciPy or Torch adjacency matrix representing user-item edges.
        """
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight

        # Concatenate user and item embeddings into a single node matrix E^0
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        # Message Passing
        for _layer in range(self.num_layers):
            if isinstance(adj_matrix, torch.sparse.FloatTensor):
                all_emb = torch.sparse.mm(adj_matrix, all_emb)
            else:
                # Fallback for dense or custom sparse mult
                all_emb = torch.matmul(adj_matrix, all_emb)
            embs.append(all_emb)

        # Combine multi-hop embeddings (mean pooling)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def forward(self, users, pos_items, neg_items, adj_matrix):
        """
        Calculates BPR (Bayesian Personalized Ranking) Loss.
        Optimizes for: Score(user, pos_item) > Score(user, neg_item)
        """
        all_users, all_items = self.computer(adj_matrix)

        user_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        # Dot product scoring
        pos_scores = torch.mul(user_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(user_emb, neg_emb).sum(dim=1)

        # BPR Loss
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        # L2 Regularization to prevent overfitting on sparse users
        reg_loss = (
            (1 / 2) * (user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / float(len(users))
        )

        return loss + (1e-4 * reg_loss)

    def export_to_feature_store(self, user_mapping: dict, item_mapping: dict, output_dir: str):
        """
        Exports the fully propagated embeddings (E_final) to Parquet files
        so the FastAPI backend can load them without needing PyTorch.
        """
        from pathlib import Path

        import pandas as pd

        logger.info("Exporting LightGCN embeddings for Feature Store...")

        # Get final embeddings without gradients
        with torch.no_grad():
            final_user_emb = self.user_embedding.weight.cpu().numpy()
            final_item_emb = self.item_embedding.weight.cpu().numpy()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        user_records = [
            {"user_id": uid, "embedding": final_user_emb[idx].tolist()} for uid, idx in user_mapping.items()
        ]
        pd.DataFrame(user_records).to_parquet(out_dir / "lightgcn_user_factors.parquet")

        item_records = [
            {"movie_id": mid, "embedding": final_item_emb[idx].tolist()} for mid, idx in item_mapping.items()
        ]
        pd.DataFrame(item_records).to_parquet(out_dir / "lightgcn_item_factors.parquet")

        logger.info("LightGCN export complete.")
