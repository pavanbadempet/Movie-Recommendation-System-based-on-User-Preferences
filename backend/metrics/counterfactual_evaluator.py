"""
Counterfactual Off-Policy Evaluation (OPE) Engine.

Estimates how a new recommendation policy would perform using historical
logged data — without requiring live A/B tests.

Implements three industry-standard OPE estimators:
1. **IPS (Inverse Propensity Scoring)**: Unbiased but high variance
2. **SNIPS (Self-Normalized IPS)**: Lower variance, slightly biased
3. **DR (Doubly Robust)**: Best of both — combines a reward model with IPS

Why this matters:
- A/B tests are expensive and slow (weeks of traffic)
- OPE lets you evaluate dozens of policy candidates offline in seconds
- Google, Netflix, and Spotify all use OPE for policy selection

References:
  - Dudík et al. "Doubly Robust Policy Evaluation" (ICML 2011)
  - Swaminathan & Joachims "The Self-Normalized Estimator" (NeurIPS 2015)
  - Gilotte et al. "Offline A/B testing" (WSDM 2018)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoggedInteraction:
    """
    A single logged interaction from the production system.

    Captures the full context needed for off-policy evaluation:
    - What was the user context
    - What action (model selection / recommendation) was taken
    - What reward was observed
    - What was the propensity (probability) of this action under the logging policy
    """
    user_id: int
    item_id: int
    reward: float  # 1.0 for click/engagement, 0.0 for no interaction
    propensity: float  # P(action | context) under the logging policy
    context_features: dict[str, float] = field(default_factory=dict)
    selected_models: list[str] = field(default_factory=list)
    timestamp: float = 0.0


class CounterfactualEvaluator:
    """
    Off-Policy Evaluation engine for recommendation policies.

    Evaluates how a new target policy would perform using logged data
    from the current production (behavior) policy, without live deployment.

    Thread-safe: all computations are stateless.
    """

    def __init__(self, ips_clip: float = 10.0):
        """
        Args:
            ips_clip: Maximum importance weight to prevent variance explosion.
        """
        self.ips_clip = ips_clip

    def ips_estimate(
        self,
        logged_data: list[LoggedInteraction],
        target_propensities: list[float],
    ) -> dict[str, float]:
        """
        Inverse Propensity Scoring (IPS) estimator.

        Reweights observed rewards by the importance ratio:
        w(x,a) = π_target(a|x) / π_behavior(a|x)

        Unbiased under correct propensity specification, but can have
        high variance when the policies differ significantly.

        Args:
            logged_data: Historical interactions from the behavior policy.
            target_propensities: P(action | context) under the target policy,
                aligned with logged_data.

        Returns:
            Dict with 'estimate', 'variance', 'effective_sample_size', 'num_samples'.
        """
        if not logged_data or len(logged_data) != len(target_propensities):
            return {"estimate": 0.0, "variance": 0.0, "effective_sample_size": 0, "num_samples": 0}

        n = len(logged_data)
        weighted_rewards = []

        for interaction, target_p in zip(logged_data, target_propensities):
            if interaction.propensity < 1e-8:
                continue  # Skip if behavior policy had near-zero propensity

            # Importance weight: target / behavior
            w = min(target_p / interaction.propensity, self.ips_clip)
            weighted_rewards.append(w * interaction.reward)

        if not weighted_rewards:
            return {"estimate": 0.0, "variance": 0.0, "effective_sample_size": 0, "num_samples": 0}

        arr = np.array(weighted_rewards)
        estimate = float(arr.mean())
        variance = float(arr.var())

        # Effective sample size: measures how much the importance weights
        # reduce the effective number of samples (Kish's ESS)
        weights = []
        for interaction, target_p in zip(logged_data, target_propensities):
            if interaction.propensity >= 1e-8:
                weights.append(min(target_p / interaction.propensity, self.ips_clip))

        w_arr = np.array(weights) if weights else np.array([1.0])
        ess = float((w_arr.sum() ** 2) / (w_arr ** 2).sum()) if w_arr.sum() > 0 else 0.0

        return {
            "estimate": round(estimate, 6),
            "variance": round(variance, 6),
            "effective_sample_size": round(ess, 1),
            "num_samples": n,
        }

    def snips_estimate(
        self,
        logged_data: list[LoggedInteraction],
        target_propensities: list[float],
    ) -> dict[str, float]:
        """
        Self-Normalized Inverse Propensity Scoring (SNIPS) estimator.

        Divides the IPS sum by the sum of importance weights, reducing
        variance at the cost of introducing a small bias. In practice,
        SNIPS almost always outperforms vanilla IPS.

        Args:
            logged_data: Historical interactions from the behavior policy.
            target_propensities: P(action | context) under the target policy.

        Returns:
            Dict with 'estimate', 'weight_sum', 'effective_sample_size', 'num_samples'.
        """
        if not logged_data or len(logged_data) != len(target_propensities):
            return {"estimate": 0.0, "weight_sum": 0.0, "effective_sample_size": 0, "num_samples": 0}

        numerator = 0.0
        denominator = 0.0
        weights = []

        for interaction, target_p in zip(logged_data, target_propensities):
            if interaction.propensity < 1e-8:
                continue

            w = min(target_p / interaction.propensity, self.ips_clip)
            numerator += w * interaction.reward
            denominator += w
            weights.append(w)

        if denominator < 1e-8:
            return {"estimate": 0.0, "weight_sum": 0.0, "effective_sample_size": 0, "num_samples": len(logged_data)}

        estimate = numerator / denominator

        # Kish's ESS
        w_arr = np.array(weights)
        ess = float((w_arr.sum() ** 2) / (w_arr ** 2).sum()) if len(weights) > 0 else 0.0

        return {
            "estimate": round(estimate, 6),
            "weight_sum": round(denominator, 4),
            "effective_sample_size": round(ess, 1),
            "num_samples": len(logged_data),
        }

    def doubly_robust_estimate(
        self,
        logged_data: list[LoggedInteraction],
        target_propensities: list[float],
        reward_model_predictions: list[float],
    ) -> dict[str, float]:
        """
        Doubly Robust (DR) estimator — the gold standard for OPE.

        Combines a reward model (imputation) with IPS correction:
        DR = E[reward_model(x,a)] + IPS_correction

        Doubly robust because it is consistent (converges to truth) if
        EITHER the propensity model OR the reward model is correct.

        Args:
            logged_data: Historical interactions from the behavior policy.
            target_propensities: P(action | context) under the target policy.
            reward_model_predictions: Predicted reward for each (user, item) pair
                under the target policy.

        Returns:
            Dict with 'estimate', 'variance', 'reward_model_component',
                'ips_correction_component', 'effective_sample_size', 'num_samples'.
        """
        n = len(logged_data)
        if not logged_data or len(target_propensities) != n or len(reward_model_predictions) != n:
            return {
                "estimate": 0.0, "variance": 0.0,
                "reward_model_component": 0.0, "ips_correction_component": 0.0,
                "effective_sample_size": 0, "num_samples": 0,
            }

        dr_terms = []
        reward_model_sum = 0.0
        ips_correction_sum = 0.0
        weights = []

        for interaction, target_p, pred_reward in zip(
            logged_data, target_propensities, reward_model_predictions
        ):
            # Reward model component
            reward_model_sum += pred_reward

            if interaction.propensity < 1e-8:
                dr_terms.append(pred_reward)
                continue

            # IPS correction: correct the reward model's error using importance weighting
            w = min(target_p / interaction.propensity, self.ips_clip)
            correction = w * (interaction.reward - pred_reward)

            ips_correction_sum += correction
            dr_terms.append(pred_reward + correction)
            weights.append(w)

        arr = np.array(dr_terms)
        estimate = float(arr.mean())
        variance = float(arr.var())

        # Kish's ESS
        w_arr = np.array(weights) if weights else np.array([1.0])
        ess = float((w_arr.sum() ** 2) / (w_arr ** 2).sum()) if w_arr.sum() > 0 else 0.0

        return {
            "estimate": round(estimate, 6),
            "variance": round(variance, 6),
            "reward_model_component": round(reward_model_sum / max(n, 1), 6),
            "ips_correction_component": round(ips_correction_sum / max(n, 1), 6),
            "effective_sample_size": round(ess, 1),
            "num_samples": n,
        }

    def compare_policies(
        self,
        logged_data: list[LoggedInteraction],
        policy_a_propensities: list[float],
        policy_b_propensities: list[float],
        reward_model_predictions_a: list[float] | None = None,
        reward_model_predictions_b: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Compare two candidate policies using all three OPE estimators.

        Returns a comprehensive comparison report with a recommendation
        on which policy to deploy.

        Args:
            logged_data: Historical interactions from the behavior policy.
            policy_a_propensities: Target propensities for Policy A.
            policy_b_propensities: Target propensities for Policy B.
            reward_model_predictions_a: Optional reward model predictions for Policy A.
            reward_model_predictions_b: Optional reward model predictions for Policy B.

        Returns:
            Dict with per-policy results and a deployment recommendation.
        """
        result = {"policy_a": {}, "policy_b": {}, "recommendation": ""}

        # IPS estimates
        result["policy_a"]["ips"] = self.ips_estimate(logged_data, policy_a_propensities)
        result["policy_b"]["ips"] = self.ips_estimate(logged_data, policy_b_propensities)

        # SNIPS estimates
        result["policy_a"]["snips"] = self.snips_estimate(logged_data, policy_a_propensities)
        result["policy_b"]["snips"] = self.snips_estimate(logged_data, policy_b_propensities)

        # DR estimates (if reward model predictions available)
        if reward_model_predictions_a is not None and reward_model_predictions_b is not None:
            result["policy_a"]["dr"] = self.doubly_robust_estimate(
                logged_data, policy_a_propensities, reward_model_predictions_a
            )
            result["policy_b"]["dr"] = self.doubly_robust_estimate(
                logged_data, policy_b_propensities, reward_model_predictions_b
            )

        # Generate recommendation based on SNIPS (most reliable in practice)
        a_est = result["policy_a"]["snips"]["estimate"]
        b_est = result["policy_b"]["snips"]["estimate"]
        a_ess = result["policy_a"]["snips"]["effective_sample_size"]
        b_ess = result["policy_b"]["snips"]["effective_sample_size"]

        min_ess_threshold = 50.0  # Need at least 50 effective samples

        if a_ess < min_ess_threshold and b_ess < min_ess_threshold:
            result["recommendation"] = "insufficient_data"
            result["reason"] = (
                f"Both policies have low effective sample sizes "
                f"(A: {a_ess:.0f}, B: {b_ess:.0f}). Need more logged data."
            )
        elif a_est > b_est:
            lift = ((a_est - b_est) / max(abs(b_est), 1e-8)) * 100
            result["recommendation"] = "policy_a"
            result["reason"] = f"Policy A outperforms B by {lift:.1f}% (SNIPS: {a_est:.4f} vs {b_est:.4f})"
        elif b_est > a_est:
            lift = ((b_est - a_est) / max(abs(a_est), 1e-8)) * 100
            result["recommendation"] = "policy_b"
            result["reason"] = f"Policy B outperforms A by {lift:.1f}% (SNIPS: {b_est:.4f} vs {a_est:.4f})"
        else:
            result["recommendation"] = "no_difference"
            result["reason"] = "Both policies have equivalent expected reward"

        return result

    def evaluate_router_configuration(
        self,
        logged_data: list[LoggedInteraction],
        current_k: int,
        candidate_k: int,
        model_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a different router-k configuration against the current one.

        Simulates what would happen if we changed the number of active models
        in the MoE routing, using logged interaction data.

        Args:
            logged_data: Historical interactions with model selection info.
            current_k: Current number of active models (e.g. 2).
            candidate_k: Proposed number of active models (e.g. 3).
            model_names: Names of the ensemble models.

        Returns:
            Dict with estimated impact of changing router-k.
        """
        if model_names is None:
            model_names = ["lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion"]

        n = len(model_names)
        current_selection_prob = current_k / n  # Simplified uniform selection
        candidate_selection_prob = candidate_k / n

        # Generate propensities
        current_props = []
        candidate_props = []

        for interaction in logged_data:
            # Current: uniform probability of selecting k out of n models
            current_props.append(max(current_selection_prob, 0.01))
            candidate_props.append(max(candidate_selection_prob, 0.01))

        comparison = self.compare_policies(
            logged_data,
            policy_a_propensities=current_props,
            policy_b_propensities=candidate_props,
        )

        return {
            "current_k": current_k,
            "candidate_k": candidate_k,
            "comparison": comparison,
            "latency_factor": round(candidate_k / max(current_k, 1), 2),
        }
