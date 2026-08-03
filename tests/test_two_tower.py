"""Tests for SOTA Two-Tower PyTorch Neural Model."""

import pytest
import torch
from backend.models.two_tower import TwoTowerModel, UserTower, ItemTower


def test_user_and_item_towers():
    user_tower = UserTower(num_users=100, feature_dim=16, embedding_dim=32)
    item_tower = ItemTower(num_items=500, feature_dim=16, embedding_dim=32)

    user_ids = torch.tensor([1, 5, 12], dtype=torch.long)
    item_ids = torch.tensor([10, 50, 120], dtype=torch.long)

    user_embeds = user_tower(user_ids)
    item_embeds = item_tower(item_ids)

    assert user_embeds.shape == (3, 32)
    assert item_embeds.shape == (3, 32)

    # Check L2 normalization (norm should be ~1.0)
    user_norms = torch.norm(user_embeds, p=2, dim=1)
    assert torch.allclose(user_norms, torch.ones_like(user_norms), atol=1e-4)


def test_two_tower_infonce_loss():
    model = TwoTowerModel(num_users=100, num_items=500, embedding_dim=32)
    user_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    item_ids = torch.tensor([10, 20, 30, 40], dtype=torch.long)

    user_embeds, item_embeds = model(user_ids, item_ids)
    loss = model.compute_infonce_loss(user_embeds, item_embeds)

    assert loss is not None
    assert loss.item() > 0.0
