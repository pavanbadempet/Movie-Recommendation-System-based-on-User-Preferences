
import numpy as np
import torch

from backend.models.neural_weight_optimizer import (
    WEIGHT_KEYS,
    ContextualWeightNetwork,
    get_contextual_weights,
    train_contextual_weight_network,
)


def test_contextual_weight_network_forward():
    """Verify ContextualWeightNetwork outputs correct dimensions and normalized weights."""
    net = ContextualWeightNetwork(context_dim=20, n_models=7)
    net.eval()

    # Batch size 3, 20-dim context
    context = torch.randn(3, 20)
    with torch.no_grad():
        weights = net(context)

    assert weights.shape == (3, 7)
    for i in range(3):
        w_sum = weights[i].sum().item()
        assert abs(w_sum - 1.0) < 1e-5
        assert torch.all(weights[i] >= 0.0)


def test_get_contextual_weights_fallback(tmp_path):
    """Verify get_contextual_weights falls back to static weights if file does not exist."""
    non_existent_path = tmp_path / "non_existent.pth"
    weights = get_contextual_weights(
        behavior_profile={"total_ratings": 5, "avg_rating": 4.0},
        als_user_embedding=np.random.randn(16),
        model_path=non_existent_path,
    )

    assert len(weights) == len(WEIGHT_KEYS)
    for key in WEIGHT_KEYS:
        assert key in weights
        assert isinstance(weights[key], float)


def test_train_contextual_weight_network_runs(monkeypatch, tmp_path):
    """Verify contextual network training executes and saves weights successfully."""
    # Mock Models directory
    dummy_model_path = tmp_path / "contextual_weight_net.pth"
    monkeypatch.setattr("backend.models.neural_weight_optimizer.MODELS_DIR", tmp_path)

    # Mock optimize_ensemble_weights functions
    dummy_users = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

    def mock_load_interaction_data():
        return {uid: [{"event_ts": "2026-06-20T00:00:00Z", "movie_id": 42}] for uid in dummy_users}

    def mock_build_validation_split(user_events):
        train_hist = {uid: [42] for uid in dummy_users}
        val_gt = {uid: {42} for uid in dummy_users}
        return train_hist, val_gt

    def mock_precompute_per_model_scores(engine, train_history, val_ground_truth, rng):
        # returns user_id -> {item_id: [7 scores]}
        return {uid: {42: [0.60, 0.20, 0.10, 0.0, 0.05, 0.0, 0.05]} for uid in dummy_users}

    def mock_iter_events():
        # Yield dummy events
        for uid in dummy_users:
            yield {"user_id": uid, "event_type": "rating", "rating": 5.0, "movie_id": 42}
            yield {"user_id": uid, "event_type": "click", "movie_id": 42}

    monkeypatch.setattr("scripts.optimize_ensemble_weights._load_interaction_data", mock_load_interaction_data)
    monkeypatch.setattr("scripts.optimize_ensemble_weights._build_validation_split", mock_build_validation_split)
    monkeypatch.setattr("scripts.optimize_ensemble_weights._precompute_per_model_scores", mock_precompute_per_model_scores)
    monkeypatch.setattr("backend.events.iter_events", mock_iter_events)

    # Train for a small number of epochs
    train_contextual_weight_network(epochs=5, lr=1e-3)

    assert dummy_model_path.exists()

    # Verify we can load and run it
    weights = get_contextual_weights(
        behavior_profile={"total_ratings": 2, "avg_rating": 3.0, "click_count": 1, "view_count": 4},
        als_user_embedding=np.random.randn(16),
        model_path=dummy_model_path,
    )
    assert len(weights) == 7
    assert abs(sum(weights.values()) - 1.0) < 1e-4
