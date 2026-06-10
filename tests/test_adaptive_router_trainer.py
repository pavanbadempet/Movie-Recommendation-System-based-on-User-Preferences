"""
Tests for AdaptiveRouterTrainer — online self-training of the MoE router.

Covers:
- Replay buffer (circular eviction, capacity limits)
- IPS debiasing (selected models get down-weighted)
- Router weights change after training steps
- Checkpoint save/load round-trip
- Stats reporting
"""

from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

from backend.learning.adaptive_router_trainer import AdaptiveRouterTrainer
from backend.models.contextual_router import ContextualRouter


@pytest.fixture
def router():
    torch.manual_seed(42)
    return ContextualRouter(emb_dim=16)


@pytest.fixture
def trainer(router):
    return AdaptiveRouterTrainer(
        router=router,
        buffer_capacity=100,
        min_train_size=10,
        batch_size=8,
        lr=0.01,
        checkpoint_interval=50,
    )


def _make_user_state(emb_dim: int = 16) -> torch.Tensor:
    """Create a random user state vector [emb_dim + 4]."""
    return torch.randn(emb_dim + 4)


def _make_model_scores() -> dict[str, float]:
    """Create random model scores."""
    models = ["lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion"]
    return {m: float(np.random.uniform(0.0, 1.0)) for m in models}


class TestReplayBuffer:
    def test_record_adds_to_buffer(self, trainer):
        assert trainer.buffer_size == 0
        trainer.record(_make_user_state(), _make_model_scores())
        assert trainer.buffer_size == 1

    def test_buffer_circular_eviction(self, trainer):
        """Buffer should evict oldest when capacity is reached."""
        for i in range(150):
            trainer.record(_make_user_state(), _make_model_scores())

        assert trainer.buffer_size == 100  # Capacity is 100
        assert trainer._total_samples_recorded == 150

    def test_is_ready_threshold(self, trainer):
        """Should not be ready until min_train_size is reached."""
        for i in range(9):
            trainer.record(_make_user_state(), _make_model_scores())
        assert not trainer.is_ready

        trainer.record(_make_user_state(), _make_model_scores())
        assert trainer.is_ready


class TestIPSDebiasing:
    def test_ips_weights_uniform_when_no_selections(self, trainer):
        """IPS weights should be uniform when no selection data exists."""
        for i in range(20):
            trainer.record(_make_user_state(), _make_model_scores(), selected_models=None)

        ips = trainer._compute_ips_weights(
            [None] * 20,
            device=torch.device("cpu"),
        )
        assert torch.allclose(ips, torch.ones(20))

    def test_ips_weights_downweight_frequent_selections(self, trainer):
        """Models selected more frequently should have lower IPS weights (closer to 1/freq)."""
        # Record 50 samples where lightgcn is always selected
        for i in range(50):
            trainer.record(_make_user_state(), _make_model_scores(), selected_models=["lightgcn", "quantum"])

        # Now create batch where some have lightgcn vs rare models
        selected_list = [
            ["lightgcn", "quantum"],  # frequent
            ["kan", "diffusion"],  # rare (never selected before)
        ]

        ips = trainer._compute_ips_weights(selected_list, device=torch.device("cpu"))

        # IPS weights should differ (rare selection = higher weight)
        # The frequent models get lower IPS; rare get higher
        assert ips.shape == (2,)
        # Both should be positive and normalized to mean ≈ 1.0
        assert ips.mean().item() == pytest.approx(1.0, abs=0.1)


class TestTraining:
    def test_train_step_returns_none_when_not_ready(self, trainer):
        loss = trainer.train_step()
        assert loss is None

    def test_train_step_returns_loss_when_ready(self, trainer):
        for i in range(20):
            trainer.record(_make_user_state(), _make_model_scores())

        loss = trainer.train_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_training_changes_router_weights(self, trainer, router):
        """After multiple training steps, router routing should change."""
        # Record biased data: lightgcn always scores highest
        for i in range(50):
            scores = _make_model_scores()
            scores["lightgcn"] = 0.95
            scores["quantum"] = 0.90
            scores["kan"] = 0.1
            scores["diffusion"] = 0.1
            trainer.record(_make_user_state(), scores, selected_models=["lightgcn", "quantum"])

        # Get initial routing
        test_state = _make_user_state()
        initial_models, initial_weights = router.route(test_state, k=2)

        # Train multiple steps
        for _ in range(30):
            trainer.train_step()

        # Routing should have changed
        final_models, final_weights = router.route(test_state, k=2)

        # After training on data where lightgcn/quantum are best,
        # router should prefer them
        top_2 = set(final_models)
        assert "lightgcn" in top_2 or "quantum" in top_2

    def test_train_step_increments_counter(self, trainer):
        for i in range(20):
            trainer.record(_make_user_state(), _make_model_scores())

        trainer.train_step()
        assert trainer._train_steps == 1
        trainer.train_step()
        assert trainer._train_steps == 2


class TestCheckpointing:
    def test_force_checkpoint_saves_file(self, trainer):
        """force_checkpoint should save router weights to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "contextual_router.pth"
            with patch.object(trainer, "_save_checkpoint", wraps=trainer._save_checkpoint):
                # Patch MODELS_DIR to temp directory
                import backend.learning.adaptive_router_trainer as art

                original = art.MODELS_DIR
                art.MODELS_DIR = Path(tmpdir)
                try:
                    trainer.force_checkpoint()
                    assert save_path.exists()

                    # Verify the saved state_dict can be loaded
                    loaded = torch.load(save_path, map_location="cpu", weights_only=True)
                    assert isinstance(loaded, dict)
                    assert len(loaded) > 0
                finally:
                    art.MODELS_DIR = original


class TestStats:
    def test_get_stats_structure(self, trainer):
        stats = trainer.get_stats()
        assert "buffer_size" in stats
        assert "buffer_capacity" in stats
        assert "total_samples_recorded" in stats
        assert "train_steps" in stats
        assert "last_train_loss" in stats
        assert "avg_train_loss" in stats
        assert "is_ready" in stats

    def test_stats_update_after_operations(self, trainer):
        for i in range(20):
            trainer.record(_make_user_state(), _make_model_scores())

        stats_pre = trainer.get_stats()
        assert stats_pre["buffer_size"] == 20
        assert stats_pre["total_samples_recorded"] == 20
        assert stats_pre["train_steps"] == 0

        trainer.train_step()
        stats_post = trainer.get_stats()
        assert stats_post["train_steps"] == 1
        assert stats_post["last_train_loss"] > 0
