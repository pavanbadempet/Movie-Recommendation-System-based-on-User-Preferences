"""
Tests for RouterExplainer — SHAP-Style Feature Attribution for MoE Decisions.

Covers:
- Feature attribution computation (permutation-based)
- Explanation structure and content
- Human-readable text generation
- Batch explanation
- Engine integration (explain_routing API)
"""
import os

import pytest
import torch

from backend.models.contextual_router import ContextualRouter
from backend.intelligence.router_explainer import RouterExplainer, RoutingExplanation


@pytest.fixture
def router():
    torch.manual_seed(42)
    return ContextualRouter(emb_dim=16)


@pytest.fixture
def explainer(router):
    return RouterExplainer(router=router, emb_dim=16)


def _make_user_state(emb_dim: int = 16) -> torch.Tensor:
    """Create a random user state vector [emb_dim + 4]."""
    return torch.randn(emb_dim + 4)


class TestFeatureAttribution:
    def test_attributions_cover_all_features(self, explainer):
        """Attribution dict should contain user_embedding and all 4 metrics."""
        state = _make_user_state()
        explanation = explainer.explain(state, k=2)
        
        attrs = explanation.feature_attributions
        assert "user_embedding" in attrs
        assert "interaction_count" in attrs
        assert "session_length" in attrs
        assert "preference_stability" in attrs
        assert "inference_energy" in attrs

    def test_attributions_are_finite(self, explainer):
        """All attribution values should be finite floats."""
        state = _make_user_state()
        explanation = explainer.explain(state, k=2)
        
        for name, val in explanation.feature_attributions.items():
            assert isinstance(val, float)
            assert not torch.tensor(val).isnan()
            assert not torch.tensor(val).isinf()

    def test_important_feature_has_nonzero_attribution(self, explainer):
        """At least one feature should have non-zero attribution."""
        torch.manual_seed(99)
        state = torch.randn(20) * 5.0  # Amplify signal
        explanation = explainer.explain(state, k=2)
        
        total_importance = sum(abs(v) for v in explanation.feature_attributions.values())
        assert total_importance > 0.0

    def test_zero_user_state_produces_attributions(self, explainer):
        """Even zero input should not crash — attributions may be zero."""
        state = torch.zeros(20)
        explanation = explainer.explain(state, k=2)
        assert len(explanation.feature_attributions) == 5


class TestExplanationStructure:
    def test_explanation_dataclass_fields(self, explainer):
        state = _make_user_state()
        explanation = explainer.explain(state, k=2)
        
        assert isinstance(explanation, RoutingExplanation)
        assert len(explanation.selected_models) == 2
        assert len(explanation.routing_weights) == 2
        assert len(explanation.all_model_probabilities) == 6
        assert isinstance(explanation.explanation_text, str)
        assert len(explanation.explanation_text) > 0

    def test_routing_weights_sum_to_one(self, explainer):
        state = _make_user_state()
        explanation = explainer.explain(state, k=2)
        
        weight_sum = sum(explanation.routing_weights)
        assert abs(weight_sum - 1.0) < 0.01

    def test_all_model_probabilities_sum_to_one(self, explainer):
        state = _make_user_state()
        explanation = explainer.explain(state, k=3)
        
        prob_sum = sum(explanation.all_model_probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01

    def test_selected_models_are_valid(self, explainer):
        state = _make_user_state()
        explanation = explainer.explain(state, k=2)
        
        valid_models = {"lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion"}
        for model in explanation.selected_models:
            assert model in valid_models

    def test_user_state_summary(self, explainer):
        state = _make_user_state()
        explanation = explainer.explain(state, k=2)
        
        summary = explanation.user_state_summary
        assert "embedding_norm" in summary
        assert "interaction_count" in summary
        assert "session_length" in summary
        assert "preference_stability" in summary
        assert "inference_energy" in summary


class TestExplanationText:
    def test_text_contains_model_names(self, explainer):
        state = _make_user_state()
        explanation = explainer.explain(state, k=2)
        
        text = explanation.explanation_text
        # Should mention at least one selected model
        assert any(m in text for m in explanation.selected_models)

    def test_text_contains_router_selected(self, explainer):
        state = _make_user_state()
        explanation = explainer.explain(state, k=2)
        
        assert "Router selected" in explanation.explanation_text


class TestTopFeatures:
    def test_top_positive_features_sorted(self, explainer):
        state = _make_user_state()
        explanation = explainer.explain(state, k=2)
        
        # Positive features should be sorted by magnitude (descending)
        if len(explanation.top_positive_features) >= 2:
            for i in range(len(explanation.top_positive_features) - 1):
                assert abs(explanation.top_positive_features[i][1]) >= abs(explanation.top_positive_features[i+1][1])

    def test_top_negative_features_are_negative(self, explainer):
        state = _make_user_state()
        explanation = explainer.explain(state, k=2)
        
        for name, val in explanation.top_negative_features:
            assert val < 0


class TestBatchExplain:
    def test_batch_returns_list(self, explainer):
        states = [_make_user_state() for _ in range(5)]
        explanations = explainer.batch_explain(states, k=2)
        
        assert len(explanations) == 5
        assert all(isinstance(e, RoutingExplanation) for e in explanations)

    def test_batch_empty(self, explainer):
        explanations = explainer.batch_explain([], k=2)
        assert explanations == []


class TestDifferentK:
    def test_k1_selects_one_model(self, explainer):
        state = _make_user_state()
        explanation = explainer.explain(state, k=1)
        
        assert len(explanation.selected_models) == 1
        assert len(explanation.routing_weights) == 1

    def test_k6_selects_all_models(self, explainer):
        state = _make_user_state()
        explanation = explainer.explain(state, k=6)
        
        assert len(explanation.selected_models) == 6
        assert abs(sum(explanation.routing_weights) - 1.0) < 0.01


class TestEngineIntegration:
    def test_engine_explain_routing(self):
        """Verify the engine's explain_routing method works."""
        os.environ["NOVA_DISABLE_MODEL_DOWNLOADS"] = "1"
        os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key-for-ci-only"
        
        from backend.models.ensemble_engine import get_apex_engine
        engine = get_apex_engine()
        
        result = engine.explain_routing(user_id=42, k=2)
        
        assert "selected_models" in result
        assert "routing_weights" in result
        assert "feature_attributions" in result
        assert "explanation_text" in result
        assert len(result["selected_models"]) == 2
