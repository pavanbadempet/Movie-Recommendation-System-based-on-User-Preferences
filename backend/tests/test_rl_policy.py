import torch

from backend.rl_policy import ActorCriticPolicy, RLSafetyFilter
from backend.rl_reward import RLRewardEngine


def test_actor_critic_forward_pass():
    """Ensure the neural network outputs valid continuous action distributions and scalar values."""
    state_dim = 771
    action_dim = 768

    policy = ActorCriticPolicy(state_dim=state_dim, action_dim=action_dim)

    # Create mock batch of 4 states
    mock_states = torch.randn(4, state_dim)

    action_mean, action_std, value = policy(mock_states)

    # Check shapes
    assert action_mean.shape == (4, action_dim)
    assert action_std.shape == (4, action_dim)
    assert value.shape == (4, 1)

    # Check bounds
    assert not torch.isnan(action_mean).any()
    assert (action_std > 0).all()  # Standard deviation must be strictly positive


def test_rl_safety_filter():
    """Ensure the safety filter correctly blocks disliked movies and falls back gracefully."""
    candidates = [10, 20, 30, 40, 50, 60]
    user_dislikes = {20, 40}

    safe = RLSafetyFilter.apply_hard_constraints(candidates, user_dislikes)

    assert 20 not in safe
    assert 40 not in safe
    assert safe == [10, 30, 50, 60]

    # Test fallback scenario when all are blocked
    all_disliked = {10, 20, 30, 40, 50, 60}
    safe_fallback = RLSafetyFilter.apply_hard_constraints(candidates, all_disliked)
    assert safe_fallback == candidates[:5]  # Should return top 5 as emergency fallback


def test_reward_engine_logic():
    """Ensure long-term retention mathematically dominates short-term clicks."""
    engine = RLRewardEngine()

    # Pure click
    click_reward = engine.calculate_reward({"event_type": "click", "timestamp": 100}, [])
    assert click_reward == 0.1

    # Click + 3-day retention (within 1-7 days)
    retention_reward = engine.calculate_reward(
        {"event_type": "click", "timestamp": 100}, [{"event_type": "view", "timestamp": 100 + (86400 * 3)}]
    )
    # 0.1 (click) + 1.0 (retention)
    assert abs(retention_reward - 1.1) < 1e-4

    # Hate rating
    hate_reward = engine.calculate_reward({"event_type": "rating", "rating": 1.0, "timestamp": 100}, [])
    assert hate_reward == -2.0
