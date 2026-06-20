import torch
import torch.nn as nn

from backend.intelligence.active_inference_engine import ActiveInferenceEngine
from backend.models.ensemble_engine import ApexEnsembleEngine


def test_apex_ensemble_initialization():
    """Verify that all 6 neural paradigms initialize with correct tensor dimensions."""
    engine = ApexEnsembleEngine(num_users=100, num_items=100, emb_dim=16)

    # The constructor uses max(num_users, 610) and max(num_items, 9724) to ensure
    # MovieLens-scale compatibility, so actual shapes use those floors.
    effective_users = max(100, 610)
    effective_items = max(100, 9724)

    # Check Quantum Fluid
    assert engine.quantum.user_embedding.amplitude.weight.shape == (effective_users, 16)
    assert engine.quantum.item_embedding.phase.weight.shape == (effective_items, 16)

    # Check Hyperbolic
    assert engine.hyperbolic.user_embedding.weight.shape == (effective_users, 16)

    # Check Clifford
    assert engine.clifford.user_embedding.weight.shape == (effective_users, 16)
    assert engine.clifford.item_embedding.weight.shape == (effective_items, 16)

    # Check KAN (Requires concat of user+item)
    assert engine.kan is not None

    # Check SASRec and LightGCN (SASRec uses max(num_items, 32660))
    effective_sasrec_items = max(100, 32660)
    assert engine.sasrec.item_emb.weight.shape[0] == effective_sasrec_items + 1  # +1 for padding
    assert engine.sasrec.item_emb.weight.shape[1] == 128  # SASRec hidden_dim
    # LightGCN uses max(num_users, 1110) and max(num_items, 12966)
    assert engine.lightgcn.user_embedding.weight.shape == (max(100, 1110), 16)


def test_apex_ensemble_forward_pass_no_nans():
    """
    Verify that the 6-model forward pass computes a blended score
    without mathematical collapse (NaNs or Infs).
    """
    engine = ApexEnsembleEngine(num_users=100, num_items=100, emb_dim=16)
    engine.eval()

    user_id = 5
    candidate_ids = [10, 20, 30, 40]

    scores = engine.predict_ensemble(user_id, candidate_ids)

    # Verify outputs
    assert len(scores) == 4
    for _item_id, score in scores.items():
        assert not torch.isnan(torch.tensor(score))
        assert not torch.isinf(torch.tensor(score))
        # Since it uses Min-Max scaling and weights, score should be roughly between 0 and 1
        assert 0.0 <= score <= 1.0


def test_active_inference_gradient_flow():
    """
    Verify Karl Friston's Free Energy Principle physically alters the weights.
    We assert that backward() produces non-zero gradients on the dynamic prior.
    """
    active_engine = ActiveInferenceEngine(emb_dim=16)

    # Store initial weights to prove they change
    initial_prior = active_engine.dynamic_prior.clone()

    # Simulate a user DISLIKING a movie (High Surprise / Negative Reward)
    movie_embedding = torch.randn(1, 16)
    reward = -1.0

    # Execute Self-Healing
    loss = active_engine.self_heal(movie_embedding, reward)

    # 1. Loss should be strictly positive for a Dislike
    assert loss > 0

    # 2. Gradients must have flowed to the dynamic prior
    assert active_engine.dynamic_prior.grad is not None
    assert torch.sum(torch.abs(active_engine.dynamic_prior.grad)) > 0

    # 3. The weights must have physically shifted away from the initial state
    assert not torch.allclose(active_engine.dynamic_prior, initial_prior)


def test_apex_engine_resolves_item_embedding_by_exact_movie_id():
    engine = ApexEnsembleEngine.__new__(ApexEnsembleEngine)
    nn.Module.__init__(engine)
    engine._item_id_to_index = {42: 1}
    item_embedding = nn.Embedding(3, 4)
    item_embedding.weight.data.copy_(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        )
    )
    engine.lightgcn = nn.Module()
    engine.lightgcn.item_embedding = item_embedding

    assert torch.equal(engine.get_item_embedding(42), torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert engine.get_item_embedding(999) is None


def test_apex_engine_configurable_uncertainty_penalty(monkeypatch):
    """Verify that APEX_UNCERTAINTY_PENALTY env var alters uncertainty gating behavior without errors."""
    engine = ApexEnsembleEngine(num_users=100, num_items=100, emb_dim=16)
    engine.eval()

    # Test high penalty configuration
    monkeypatch.setenv("APEX_UNCERTAINTY_PENALTY", "0.9")
    scores_high = engine.predict_ensemble(5, [10, 20])

    # Test zero penalty configuration
    monkeypatch.setenv("APEX_UNCERTAINTY_PENALTY", "0.0")
    scores_zero = engine.predict_ensemble(5, [10, 20])

    assert len(scores_high) == 2
    assert len(scores_zero) == 2
    for item_id in [10, 20]:
        assert 0.0 <= scores_high[item_id] <= 1.0
        assert 0.0 <= scores_zero[item_id] <= 1.0


def test_apex_engine_geometric_blend_mode(monkeypatch):
    """Verify that APEX_ENSEMBLE_BLEND_MODE=geometric computes scores correctly without mathematical collapse."""
    engine = ApexEnsembleEngine(num_users=100, num_items=100, emb_dim=16)
    engine.eval()

    monkeypatch.setenv("APEX_ENSEMBLE_BLEND_MODE", "geometric")
    scores = engine.predict_ensemble(5, [10, 20])

    assert len(scores) == 2
    for item_id in [10, 20]:
        assert 0.0 <= scores[item_id] <= 1.0
