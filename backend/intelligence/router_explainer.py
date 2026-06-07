"""
Router Explainability Module — SHAP-Style Feature Attribution for MoE Decisions.

Explains WHY the Contextual Router selected specific models for a given user:
- Which user features drove the routing decision
- What the marginal contribution of each feature dimension was
- Human-readable explanations for debugging and transparency

Uses permutation-based feature attribution (analogous to SHAP) computed
without external dependencies — just PyTorch forward passes with masked inputs.

This is critical for:
1. **Debugging**: Understanding why a model was unexpectedly selected/excluded
2. **Fairness**: Ensuring routing doesn't discriminate based on sensitive attributes
3. **Trust**: Operators can verify the router is making sensible decisions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import numpy as np

if TYPE_CHECKING:
    from backend.models.contextual_router import ContextualRouter

logger = logging.getLogger(__name__)

# Human-readable names for the 4 contextual metrics appended to user embedding
METRIC_NAMES = [
    "interaction_count",
    "session_length",
    "preference_stability",
    "inference_energy",
]


@dataclass
class RoutingExplanation:
    """
    Complete explanation of a routing decision.

    Contains per-model routing probabilities, selected models,
    and feature-level attribution for the routing decision.
    """
    selected_models: list[str]
    routing_weights: list[float]
    all_model_probabilities: dict[str, float]
    feature_attributions: dict[str, float]
    top_positive_features: list[tuple[str, float]]
    top_negative_features: list[tuple[str, float]]
    explanation_text: str
    user_state_summary: dict[str, float] = field(default_factory=dict)


class RouterExplainer:
    """
    Explains routing decisions using permutation-based feature attribution.

    For each feature dimension, we measure how much the routing probability
    changes when that feature is replaced with a baseline (zero). The magnitude
    of the change is the feature's "importance" for the routing decision.

    This is a model-agnostic explanation method (like SHAP) that works with
    any router architecture.
    """

    def __init__(
        self,
        router: "ContextualRouter",
        emb_dim: int = 16,
        num_permutations: int = 10,
    ):
        """
        Args:
            router: The ContextualRouter to explain.
            emb_dim: Embedding dimension (first emb_dim features are embedding dims).
            num_permutations: Number of random baseline samples for attribution.
        """
        self.router = router
        self.emb_dim = emb_dim
        self.num_permutations = num_permutations

    def explain(
        self,
        user_state: torch.Tensor,
        k: int = 2,
    ) -> RoutingExplanation:
        """
        Generate a complete explanation for a routing decision.

        Args:
            user_state: User state vector [emb_dim + 4].
            k: Number of models selected by the router.

        Returns:
            RoutingExplanation with feature attributions and human-readable text.
        """
        self.router.eval()
        user_state = user_state.detach()

        # 1. Get the actual routing decision
        selected_models, routing_weights = self.router.route(user_state, k=k)

        # 2. Get full probability distribution
        with torch.no_grad():
            logits = self.router(user_state.unsqueeze(0) if user_state.dim() == 1 else user_state)
            probs = F.softmax(logits.squeeze(), dim=-1)

        all_probs = {
            name: round(probs[i].item(), 4)
            for i, name in enumerate(self.router.model_names)
        }

        # 3. Compute feature attributions
        attributions = self._compute_attributions(user_state, selected_models)

        # 4. Extract user state summary
        state_summary = self._extract_state_summary(user_state)

        # 5. Sort attributions
        sorted_attrs = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [(name, val) for name, val in sorted_attrs if val > 0.001][:5]
        top_negative = [(name, val) for name, val in sorted_attrs if val < -0.001][:5]

        # 6. Generate human-readable explanation
        explanation_text = self._generate_text(
            selected_models, routing_weights, top_positive, top_negative, state_summary
        )

        return RoutingExplanation(
            selected_models=selected_models,
            routing_weights=[round(w.item(), 4) for w in routing_weights],
            all_model_probabilities=all_probs,
            feature_attributions=attributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            explanation_text=explanation_text,
            user_state_summary=state_summary,
        )

    def _compute_attributions(
        self,
        user_state: torch.Tensor,
        selected_models: list[str],
    ) -> dict[str, float]:
        """
        Compute per-feature attribution using permutation importance.

        For each feature, we zero it out and measure the change in routing
        probability for the selected models. Larger change = more important.
        """
        self.router.eval()
        state = user_state.clone()
        total_dim = state.shape[-1]

        # Baseline: routing probabilities with original state
        with torch.no_grad():
            base_logits = self.router(state.unsqueeze(0)).squeeze()
            base_probs = F.softmax(base_logits, dim=-1)

        # Get indices of selected models
        selected_indices = [
            self.router.model_names.index(m)
            for m in selected_models
            if m in self.router.model_names
        ]

        # Sum of probabilities for selected models (our "target score")
        base_score = sum(base_probs[idx].item() for idx in selected_indices)

        # Group features: embedding dimensions are aggregated, metrics are individual
        feature_groups: dict[str, list[int]] = {}

        # Group embedding dimensions into a single "user_embedding" feature
        feature_groups["user_embedding"] = list(range(self.emb_dim))

        # Individual metric features
        for i, name in enumerate(METRIC_NAMES):
            dim_idx = self.emb_dim + i
            if dim_idx < total_dim:
                feature_groups[name] = [dim_idx]

        attributions: dict[str, float] = {}

        for feature_name, dim_indices in feature_groups.items():
            # Zero out this feature group
            perturbed = state.clone()
            for idx in dim_indices:
                perturbed[idx] = 0.0

            with torch.no_grad():
                perturbed_logits = self.router(perturbed.unsqueeze(0)).squeeze()
                perturbed_probs = F.softmax(perturbed_logits, dim=-1)

            perturbed_score = sum(perturbed_probs[idx].item() for idx in selected_indices)

            # Attribution = how much the selection probability drops when feature is removed
            attribution = base_score - perturbed_score
            attributions[feature_name] = round(attribution, 4)

        return attributions

    def _extract_state_summary(self, user_state: torch.Tensor) -> dict[str, float]:
        """Extract human-readable summary from user state vector."""
        summary = {}

        # Embedding norm
        emb = user_state[:self.emb_dim]
        summary["embedding_norm"] = round(torch.norm(emb).item(), 4)

        # Individual metrics
        for i, name in enumerate(METRIC_NAMES):
            idx = self.emb_dim + i
            if idx < user_state.shape[-1]:
                summary[name] = round(user_state[idx].item(), 4)

        return summary

    def _generate_text(
        self,
        selected_models: list[str],
        routing_weights: torch.Tensor,
        top_positive: list[tuple[str, float]],
        top_negative: list[tuple[str, float]],
        state_summary: dict[str, float],
    ) -> str:
        """Generate a human-readable explanation of the routing decision."""
        lines = []

        # Model selection
        model_str = ", ".join(
            f"{m} ({w.item():.0%})" for m, w in zip(selected_models, routing_weights)
        )
        lines.append(f"Router selected: {model_str}")

        # User context
        interaction_count = state_summary.get("interaction_count", 0)
        session_length = state_summary.get("session_length", 0)
        stability = state_summary.get("preference_stability", 0)

        if interaction_count > 0.5:
            lines.append("User has high interaction history → sequential models favored")
        elif interaction_count < 0.1:
            lines.append("User is cold-start → content-based models favored")

        if stability > 0.7:
            lines.append("User has stable preferences → collaborative filtering reliable")
        elif stability < 0.3:
            lines.append("User preferences are volatile → diverse models needed")

        # Feature attribution
        if top_positive:
            top_feat, top_val = top_positive[0]
            lines.append(f"Top driver: '{top_feat}' (attribution: +{top_val:.3f})")

        if top_negative:
            neg_feat, neg_val = top_negative[0]
            lines.append(f"Suppression: '{neg_feat}' (attribution: {neg_val:.3f})")

        return " | ".join(lines)

    def batch_explain(
        self,
        user_states: list[torch.Tensor],
        k: int = 2,
    ) -> list[RoutingExplanation]:
        """Explain routing decisions for multiple users."""
        return [self.explain(state, k=k) for state in user_states]
