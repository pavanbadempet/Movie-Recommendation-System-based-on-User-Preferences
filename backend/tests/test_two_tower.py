"""
Tests for the Two-Tower Candidate Generation Model.
Validates architecture, embedding quality, and FAISS retrieval.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from backend.two_tower import TwoTowerModel, UserTower, ItemTower


class TestTwoTowerArchitecture:
    """Test the model architecture produces correct shapes and valid outputs."""

    def test_user_tower_output_shape(self):
        tower = UserTower(input_dim=18, embedding_dim=128)
        x = torch.randn(32, 18)
        out = tower(x)
        assert out.shape == (32, 128)

    def test_item_tower_output_shape(self):
        tower = ItemTower(input_dim=20, embedding_dim=128)
        x = torch.randn(32, 20)
        out = tower(x)
        assert out.shape == (32, 128)

    def test_embeddings_are_l2_normalized(self):
        model = TwoTowerModel(user_input_dim=18, item_input_dim=20, embedding_dim=128)
        user_emb = model.user_tower(torch.randn(10, 18))
        item_emb = model.item_tower(torch.randn(10, 20))

        user_norms = torch.norm(user_emb, p=2, dim=-1)
        item_norms = torch.norm(item_emb, p=2, dim=-1)

        assert torch.allclose(user_norms, torch.ones(10), atol=1e-5)
        assert torch.allclose(item_norms, torch.ones(10), atol=1e-5)

    def test_forward_produces_scores(self):
        model = TwoTowerModel()
        scores = model(torch.randn(16, 18), torch.randn(16, 20))
        assert scores.shape == (16,)
        assert not torch.isnan(scores).any()

    def test_scores_bounded_by_cosine_range(self):
        """Cosine similarity of L2-normalized vectors is in [-1, 1]."""
        model = TwoTowerModel()
        scores = model(torch.randn(100, 18), torch.randn(100, 20))
        assert scores.min() >= -1.01
        assert scores.max() <= 1.01

    def test_contrastive_loss_runs(self):
        model = TwoTowerModel()
        loss = model.compute_contrastive_loss(
            user_features=torch.randn(8, 18),
            pos_item_features=torch.randn(8, 20),
            neg_item_features=torch.randn(8, 5, 20),
        )
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_contrastive_loss_decreases_with_training(self):
        """Verify the model can actually learn (loss goes down)."""
        model = TwoTowerModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed batch for deterministic test
        user_feat = torch.randn(32, 18)
        pos_feat = torch.randn(32, 20)
        neg_feat = torch.randn(32, 5, 20)

        initial_loss = model.compute_contrastive_loss(user_feat, pos_feat, neg_feat).item()

        for _ in range(20):
            loss = model.compute_contrastive_loss(user_feat, pos_feat, neg_feat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = model.compute_contrastive_loss(user_feat, pos_feat, neg_feat).item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"


class TestTrainedModel:
    """Test that the trained model weights exist and produce valid outputs."""

    MODELS_DIR = Path(__file__).parent.parent.parent / "models"

    def test_trained_weights_exist(self):
        assert (self.MODELS_DIR / "two_tower.pth").exists(), "Trained weights not found"

    def test_faiss_index_exists(self):
        assert (self.MODELS_DIR / "two_tower_faiss.index").exists(), "FAISS index not found"

    def test_id_map_exists(self):
        assert (self.MODELS_DIR / "two_tower_item_ids.npy").exists(), "ID map not found"

    def test_trained_weights_differ_from_random(self):
        """Trained weights should be different from a freshly initialized model."""
        trained = TwoTowerModel()
        trained.load_state_dict(torch.load(self.MODELS_DIR / "two_tower.pth", weights_only=True))

        random_model = TwoTowerModel()

        # Compare first layer weights — they should differ
        trained_w = trained.user_tower.net[0].weight.data
        random_w = random_model.user_tower.net[0].weight.data

        assert not torch.allclose(trained_w, random_w, atol=1e-3), "Trained weights match random init!"

    def test_faiss_retrieval_returns_results(self):
        import faiss

        index = faiss.read_index(str(self.MODELS_DIR / "two_tower_faiss.index"))
        item_ids = np.load(str(self.MODELS_DIR / "two_tower_item_ids.npy"))

        assert index.ntotal > 0
        assert len(item_ids) == index.ntotal

        # Query with a random vector
        query = np.random.randn(1, 128).astype(np.float32)
        faiss.normalize_L2(query)
        distances, indices = index.search(query, 10)

        assert indices.shape == (1, 10)
        assert all(idx >= 0 for idx in indices[0])
