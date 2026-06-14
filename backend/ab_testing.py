"""
A/B Testing Framework for Recommendation Policies

This module provides:
- Multi-armed bandit strategies (epsilon-greedy, Thompson sampling)
- A/B test configuration and management
- Statistical significance testing
- Traffic partitioning
- Experiment tracking and analysis
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
from scipy import stats
import hashlib

logger = logging.getLogger(__name__)


class BanditStrategy(Enum):
    """Bandit strategy types."""
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB = "ucb"  # Upper Confidence Bound
    RANDOM = "random"


@dataclass
class RecommendationPolicy:
    """
    A recommendation policy to test.
    
    Attributes:
        name: Policy identifier
        description: Human-readable description
        ensemble_weights: Model ensemble weights for this policy
        reranking_config: Reranking configuration
        metadata: Additional policy metadata
    """
    name: str
    description: str
    ensemble_weights: Dict[str, float]
    reranking_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate ensemble weights sum to 1.0."""
        total = sum(self.ensemble_weights.values())
        if not np.isclose(total, 1.0, atol=0.01):
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total}")


@dataclass
class ExperimentMetrics:
    """Metrics collected for an experiment."""
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    ctr: float = 0.0  # Click-through rate
    conversion_rate: float = 0.0
    
    def update(self, clicked: bool = False, converted: bool = False, reward: float = 0.0) -> None:
        """Update metrics with a new interaction."""
        self.impressions += 1
        if clicked:
            self.clicks += 1
        if converted:
            self.conversions += 1
        self.total_reward += reward
        
        # Recalculate averages
        if self.impressions > 0:
            self.avg_reward = self.total_reward / self.impressions
            self.ctr = self.clicks / self.impressions
            self.conversion_rate = self.conversions / self.impressions


class MultiArmedBandit:
    """
    Multi-armed bandit for policy selection.
    
    Supports multiple exploration-exploitation strategies:
    - Epsilon-greedy: Random exploration with probability epsilon
    - Thompson sampling: Bayesian exploration using Beta distributions
    - UCB: Optimism in the face of uncertainty
    """
    
    def __init__(
        self,
        policies: List[RecommendationPolicy],
        strategy: BanditStrategy = BanditStrategy.EPSILON_GREEDY,
        epsilon: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize bandit.
        
        Args:
            policies: List of recommendation policies
            strategy: Bandit strategy to use
            epsilon: Exploration rate for epsilon-greedy
            temperature: Temperature for softmax selection
        """
        self.policies = policies
        self.strategy = strategy
        self.epsilon = epsilon
        self.temperature = temperature
        
        # Policy metrics
        self.metrics: Dict[str, ExperimentMetrics] = {
            policy.name: ExperimentMetrics() for policy in policies
        }
        
        # Thompson sampling parameters (Beta distribution)
        self.alpha: Dict[str, float] = {policy.name: 1.0 for policy in policies}
        self.beta: Dict[str, float] = {policy.name: 1.0 for policy in policies}
        
        # UCB parameters
        self.total_rounds: int = 0
        
        logger.info(f"Initialized {strategy.value} bandit with {len(policies)} policies")
    
    def select_policy(self, user_id: Optional[str] = None) -> RecommendationPolicy:
        """
        Select a policy using the configured strategy.
        
        Args:
            user_id: Optional user ID for consistent assignment
            
        Returns:
            Selected recommendation policy
        """
        if self.strategy == BanditStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy_select()
        elif self.strategy == BanditStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling_select()
        elif self.strategy == BanditStrategy.UCB:
            return self._ucb_select()
        elif self.strategy == BanditStrategy.RANDOM:
            return self._random_select()
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using random")
            return self._random_select()
    
    def _epsilon_greedy_select(self) -> RecommendationPolicy:
        """Epsilon-greedy policy selection."""
        if np.random.random() < self.epsilon:
            # Explore: random policy
            return np.random.choice(self.policies)
        else:
            # Exploit: best policy by average reward
            best_policy_name = max(
                self.metrics.keys(),
                key=lambda k: self.metrics[k].avg_reward
            )
            return next(p for p in self.policies if p.name == best_policy_name)
    
    def _thompson_sampling_select(self) -> RecommendationPolicy:
        """Thompson sampling policy selection."""
        samples = []
        for policy in self.policies:
            # Sample from Beta distribution
            sample = np.random.beta(self.alpha[policy.name], self.beta[policy.name])
            samples.append(sample)
        
        best_idx = np.argmax(samples)
        return self.policies[best_idx]
    
    def _ucb_select(self) -> RecommendationPolicy:
        """Upper Confidence Bound policy selection."""
        ucb_values = []
        for policy in self.policies:
            metrics = self.metrics[policy.name]
            if metrics.impressions == 0:
                ucb = float('inf')
            else:
                # UCB formula: avg_reward + sqrt(2 * ln(total_rounds) / n)
                exploration = np.sqrt(2 * np.log(self.total_rounds + 1) / metrics.impressions)
                ucb = metrics.avg_reward + exploration
            ucb_values.append(ucb)
        
        best_idx = np.argmax(ucb_values)
        return self.policies[best_idx]
    
    def _random_select(self) -> RecommendationPolicy:
        """Random policy selection."""
        return np.random.choice(self.policies)
    
    def update_policy(
        self,
        policy_name: str,
        clicked: bool = False,
        converted: bool = False,
        reward: float = 0.0
    ) -> None:
        """
        Update policy metrics with interaction feedback.
        
        Args:
            policy_name: Name of the policy used
            clicked: Whether the user clicked
            converted: Whether the user converted
            reward: Reward value (e.g., rating, watch time)
        """
        if policy_name not in self.metrics:
            logger.warning(f"Unknown policy: {policy_name}")
            return
        
        # Update metrics
        self.metrics[policy_name].update(clicked, converted, reward)
        
        # Update Thompson sampling parameters
        if self.strategy == BanditStrategy.THOMPSON_SAMPLING:
            if clicked:
                self.alpha[policy_name] += 1
            else:
                self.beta[policy_name] += 1
        
        # Update UCB counter
        if self.strategy == BanditStrategy.UCB:
            self.total_rounds += 1
        
        logger.debug(f"Updated policy {policy_name}: reward={reward}, clicked={clicked}")
    
    def get_policy_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all policies."""
        stats = {}
        for policy_name, metrics in self.metrics.items():
            stats[policy_name] = {
                "impressions": metrics.impressions,
                "clicks": metrics.clicks,
                "conversions": metrics.conversions,
                "avg_reward": metrics.avg_reward,
                "ctr": metrics.ctr,
                "conversion_rate": metrics.conversion_rate
            }
        return stats
    
    def reset_policy(self, policy_name: str) -> None:
        """Reset metrics for a specific policy."""
        if policy_name in self.metrics:
            self.metrics[policy_name] = ExperimentMetrics()
            self.alpha[policy_name] = 1.0
            self.beta[policy_name] = 1.0
            logger.info(f"Reset policy metrics: {policy_name}")


class ABTestManager:
    """
    A/B test manager for recommendation policies.
    
    Manages:
    - Test configuration and lifecycle
    - Traffic partitioning
    - Statistical significance testing
    - Experiment results analysis
    """
    
    def __init__(self, test_name: str, policies: List[RecommendationPolicy]):
        """
        Initialize A/B test manager.
        
        Args:
            test_name: Name of the A/B test
            policies: Policies to test
        """
        self.test_name = test_name
        self.policies = policies
        self.metrics: Dict[str, ExperimentMetrics] = {
            policy.name: ExperimentMetrics() for policy in policies
        }
        self.start_time = datetime.now()
        self.is_active = True
        
        logger.info(f"Initialized A/B test: {test_name} with {len(policies)} policies")
    
    def assign_policy(self, user_id: str) -> RecommendationPolicy:
        """
        Assign a policy to a user using consistent hashing.
        
        Args:
            user_id: User identifier
            
        Returns:
            Assigned recommendation policy
        """
        # Consistent hashing for stable assignment
        hash_value = int(hashlib.md5(f"{self.test_name}:{user_id}".encode()).hexdigest(), 16)
        policy_idx = hash_value % len(self.policies)
        
        return self.policies[policy_idx]
    
    def record_interaction(
        self,
        policy_name: str,
        clicked: bool = False,
        converted: bool = False,
        reward: float = 0.0
    ) -> None:
        """Record an interaction for a policy."""
        if policy_name not in self.metrics:
            logger.warning(f"Unknown policy: {policy_name}")
            return
        
        self.metrics[policy_name].update(clicked, converted, reward)
    
    def calculate_significance(
        self,
        metric: str = "ctr",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance between policies.
        
        Args:
            metric: Metric to compare (ctr, conversion_rate, avg_reward)
            alpha: Significance level
            
        Returns:
            Dictionary with significance results
        """
        if len(self.policies) < 2:
            return {"error": "Need at least 2 policies for comparison"}
        
        policy_names = list(self.metrics.keys())
        control = policy_names[0]
        treatment = policy_names[1]
        
        control_metrics = self.metrics[control]
        treatment_metrics = self.metrics[treatment]
        
        # Get metric values
        if metric == "ctr":
            control_clicks = control_metrics.clicks
            control_total = control_metrics.impressions
            treatment_clicks = treatment_metrics.clicks
            treatment_total = treatment_metrics.impressions
        elif metric == "conversion_rate":
            control_clicks = control_metrics.conversions
            control_total = control_metrics.impressions
            treatment_clicks = treatment_metrics.conversions
            treatment_total = treatment_metrics.impressions
        else:
            # For avg_reward, use t-test
            control_values = [control_metrics.avg_reward] * control_metrics.impressions
            treatment_values = [treatment_metrics.avg_reward] * treatment_metrics.impressions
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            
            return {
                "metric": metric,
                "control": control,
                "treatment": treatment,
                "control_value": control_metrics.avg_reward,
                "treatment_value": treatment_metrics.avg_reward,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < alpha,
                "alpha": alpha
            }
        
        # For proportions, use z-test
        if control_total == 0 or treatment_total == 0:
            return {"error": "Insufficient data for significance test"}
        
        control_rate = control_clicks / control_total
        treatment_rate = treatment_clicks / treatment_total
        
        # Pooled proportion
        pooled_prop = (control_clicks + treatment_clicks) / (control_total + treatment_total)
        
        # Standard error
        se = np.sqrt(pooled_prop * (1 - pooled_prop) * (1/control_total + 1/treatment_total))
        
        # Z-score
        z_score = (treatment_rate - control_rate) / se if se > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            "metric": metric,
            "control": control,
            "treatment": treatment,
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "lift": (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0,
            "z_score": z_score,
            "p_value": p_value,
            "significant": p_value < alpha,
            "alpha": alpha,
            "control_impressions": control_total,
            "treatment_impressions": treatment_total
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get complete A/B test results."""
        return {
            "test_name": self.test_name,
            "start_time": self.start_time.isoformat(),
            "duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "is_active": self.is_active,
            "policies": [
                {
                    "name": policy.name,
                    "description": policy.description,
                    "metrics": {
                        "impressions": self.metrics[policy.name].impressions,
                        "clicks": self.metrics[policy.name].clicks,
                        "conversions": self.metrics[policy.name].conversions,
                        "avg_reward": self.metrics[policy.name].avg_reward,
                        "ctr": self.metrics[policy.name].ctr,
                        "conversion_rate": self.metrics[policy.name].conversion_rate
                    }
                }
                for policy in self.policies
            ],
            "significance": self.calculate_significance()
        }
    
    def end_test(self) -> Dict[str, Any]:
        """End the A/B test and return final results."""
        self.is_active = False
        return self.get_results()


def create_default_policies() -> List[RecommendationPolicy]:
    """Create default recommendation policies for testing."""
    return [
        RecommendationPolicy(
            name="control",
            description="Current production ensemble weights",
            ensemble_weights={
                "sasrec": 0.659,
                "kan": 0.298,
                "lightgcn": 0.005,
                "quantum": 0.010,
                "hyperbolic": 0.004,
                "diffusion": 0.024
            }
        ),
        RecommendationPolicy(
            name="treatment_high_sasrec",
            description="Increased SASRec weight for sequential signals",
            ensemble_weights={
                "sasrec": 0.750,
                "kan": 0.200,
                "lightgcn": 0.010,
                "quantum": 0.015,
                "hyperbolic": 0.005,
                "diffusion": 0.020
            }
        ),
        RecommendationPolicy(
            name="treatment_diverse",
            description="Balanced weights for diversity",
            ensemble_weights={
                "sasrec": 0.400,
                "kan": 0.250,
                "lightgcn": 0.150,
                "quantum": 0.100,
                "hyperbolic": 0.050,
                "diffusion": 0.050
            }
        )
    ]
