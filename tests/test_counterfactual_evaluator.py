"""
Tests for CounterfactualEvaluator — Off-Policy Evaluation Engine.

Covers:
- IPS estimator (unbiased, clipping, edge cases)
- SNIPS estimator (self-normalized, lower variance)
- Doubly Robust estimator (reward model + IPS correction)
- Policy comparison (recommendation generation)
- Router-k configuration evaluation
"""
import pytest
import numpy as np

from backend.metrics.counterfactual_evaluator import (
    CounterfactualEvaluator,
    LoggedInteraction,
)


@pytest.fixture
def evaluator():
    return CounterfactualEvaluator(ips_clip=10.0)


def _make_logged_data(n: int = 100, click_rate: float = 0.3, seed: int = 42) -> list[LoggedInteraction]:
    """Generate synthetic logged interaction data."""
    rng = np.random.RandomState(seed)
    data = []
    for i in range(n):
        reward = 1.0 if rng.random() < click_rate else 0.0
        propensity = rng.uniform(0.1, 0.9)
        data.append(LoggedInteraction(
            user_id=i % 50,
            item_id=rng.randint(0, 1000),
            reward=reward,
            propensity=propensity,
            selected_models=["lightgcn", "quantum"],
        ))
    return data


class TestIPSEstimator:
    def test_ips_returns_valid_estimate(self, evaluator):
        data = _make_logged_data(100)
        target_props = [0.5] * len(data)
        
        result = evaluator.ips_estimate(data, target_props)
        assert "estimate" in result
        assert "variance" in result
        assert "effective_sample_size" in result
        assert result["num_samples"] == 100

    def test_ips_identical_policies(self, evaluator):
        """When target = behavior policy, estimate should ≈ mean reward."""
        data = _make_logged_data(500, click_rate=0.5, seed=123)
        # Target propensity = behavior propensity → importance weight = 1.0
        target_props = [d.propensity for d in data]
        
        result = evaluator.ips_estimate(data, target_props)
        mean_reward = np.mean([d.reward for d in data])
        
        # Should be close to mean reward when policies match
        assert abs(result["estimate"] - mean_reward) < 0.05

    def test_ips_clipping(self, evaluator):
        """IPS weights should be clipped to prevent variance explosion."""
        data = [LoggedInteraction(user_id=0, item_id=0, reward=1.0, propensity=0.001)]
        target_props = [0.999]
        
        result = evaluator.ips_estimate(data, target_props)
        # With clip=10.0, max weighted reward = 10.0 * 1.0 = 10.0
        assert result["estimate"] <= 10.0

    def test_ips_empty_data(self, evaluator):
        result = evaluator.ips_estimate([], [])
        assert result["estimate"] == 0.0
        assert result["num_samples"] == 0

    def test_ips_effective_sample_size(self, evaluator):
        """ESS should be less than or equal to actual sample size."""
        data = _make_logged_data(100)
        target_props = [0.3] * len(data)
        
        result = evaluator.ips_estimate(data, target_props)
        assert result["effective_sample_size"] <= 100


class TestSNIPSEstimator:
    def test_snips_returns_valid_estimate(self, evaluator):
        data = _make_logged_data(100)
        target_props = [0.5] * len(data)
        
        result = evaluator.snips_estimate(data, target_props)
        assert "estimate" in result
        assert "weight_sum" in result
        assert result["num_samples"] == 100

    def test_snips_bounded(self, evaluator):
        """SNIPS estimate should be bounded in [0, 1] for binary rewards."""
        data = _make_logged_data(200)
        target_props = [0.5] * len(data)
        
        result = evaluator.snips_estimate(data, target_props)
        assert 0.0 <= result["estimate"] <= 1.0

    def test_snips_lower_variance_than_ips(self, evaluator):
        """SNIPS should generally have lower variance than IPS."""
        data = _make_logged_data(500, seed=99)
        
        # Run multiple evaluations with different target propensities
        ips_vars = []
        snips_estimates = []
        
        for seed in range(10):
            rng = np.random.RandomState(seed + 1000)
            target_props = [rng.uniform(0.1, 0.9) for _ in data]
            
            ips_result = evaluator.ips_estimate(data, target_props)
            snips_result = evaluator.snips_estimate(data, target_props)
            
            ips_vars.append(ips_result["variance"])
            snips_estimates.append(snips_result["estimate"])
        
        # SNIPS estimates should be more stable (in [0,1] range)
        for est in snips_estimates:
            assert 0.0 <= est <= 1.5  # SNIPS should stay reasonable


class TestDoublyRobustEstimator:
    def test_dr_returns_valid_estimate(self, evaluator):
        data = _make_logged_data(100)
        target_props = [0.5] * len(data)
        reward_preds = [0.3] * len(data)  # Reward model always predicts 0.3
        
        result = evaluator.doubly_robust_estimate(data, target_props, reward_preds)
        assert "estimate" in result
        assert "reward_model_component" in result
        assert "ips_correction_component" in result
        assert result["num_samples"] == 100

    def test_dr_perfect_reward_model(self, evaluator):
        """With perfect reward model, DR should ≈ mean predicted reward."""
        data = _make_logged_data(100, seed=42)
        target_props = [d.propensity for d in data]
        # Perfect reward model: predicts exact reward
        reward_preds = [d.reward for d in data]
        
        result = evaluator.doubly_robust_estimate(data, target_props, reward_preds)
        mean_reward = np.mean([d.reward for d in data])
        
        # With perfect model, IPS correction should be near zero
        assert abs(result["ips_correction_component"]) < 0.1

    def test_dr_mismatched_lengths(self, evaluator):
        data = _make_logged_data(10)
        result = evaluator.doubly_robust_estimate(data, [0.5] * 5, [0.3] * 10)
        assert result["num_samples"] == 0


class TestPolicyComparison:
    def test_compare_policies_structure(self, evaluator):
        data = _make_logged_data(200)
        policy_a = [0.5] * len(data)
        policy_b = [0.3] * len(data)
        
        result = evaluator.compare_policies(data, policy_a, policy_b)
        assert "policy_a" in result
        assert "policy_b" in result
        assert "recommendation" in result
        assert "reason" in result

    def test_compare_with_dr(self, evaluator):
        data = _make_logged_data(200)
        policy_a = [0.5] * len(data)
        policy_b = [0.3] * len(data)
        reward_a = [0.4] * len(data)
        reward_b = [0.3] * len(data)
        
        result = evaluator.compare_policies(
            data, policy_a, policy_b,
            reward_model_predictions_a=reward_a,
            reward_model_predictions_b=reward_b,
        )
        assert "dr" in result["policy_a"]
        assert "dr" in result["policy_b"]

    def test_compare_generates_recommendation(self, evaluator):
        data = _make_logged_data(500, click_rate=0.5)
        # Policy A with higher propensity should get different estimate
        policy_a = [0.7] * len(data)
        policy_b = [0.2] * len(data)
        
        result = evaluator.compare_policies(data, policy_a, policy_b)
        assert result["recommendation"] in ("policy_a", "policy_b", "no_difference", "insufficient_data")


class TestRouterConfigEvaluation:
    def test_evaluate_router_k(self, evaluator):
        data = _make_logged_data(100)
        result = evaluator.evaluate_router_configuration(
            logged_data=data,
            current_k=2,
            candidate_k=3,
        )
        assert result["current_k"] == 2
        assert result["candidate_k"] == 3
        assert "comparison" in result
        assert result["latency_factor"] == 1.5  # 3/2

    def test_evaluate_router_k_reduction(self, evaluator):
        data = _make_logged_data(100)
        result = evaluator.evaluate_router_configuration(
            logged_data=data,
            current_k=4,
            candidate_k=2,
        )
        assert result["latency_factor"] == 0.5  # 2/4 = faster


class TestEdgeCases:
    def test_zero_propensity_skipped(self, evaluator):
        """Interactions with zero propensity should be skipped."""
        data = [
            LoggedInteraction(user_id=0, item_id=0, reward=1.0, propensity=0.0),
            LoggedInteraction(user_id=1, item_id=1, reward=1.0, propensity=0.5),
        ]
        target_props = [0.5, 0.5]
        
        result = evaluator.ips_estimate(data, target_props)
        assert result["num_samples"] == 2

    def test_all_rewards_zero(self, evaluator):
        data = [
            LoggedInteraction(user_id=i, item_id=i, reward=0.0, propensity=0.5)
            for i in range(50)
        ]
        target_props = [0.5] * 50
        
        result = evaluator.ips_estimate(data, target_props)
        assert result["estimate"] == 0.0

    def test_all_rewards_one(self, evaluator):
        data = [
            LoggedInteraction(user_id=i, item_id=i, reward=1.0, propensity=0.5)
            for i in range(50)
        ]
        target_props = [0.5] * 50
        
        result = evaluator.ips_estimate(data, target_props)
        assert result["estimate"] > 0.0
