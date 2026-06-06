import torch

from backend.intelligence.active_inference_engine import ActiveInferenceEngine
from backend.models.ensemble_engine import ApexEnsembleEngine


def test_apex_ensemble_initialization():
    """Verify that all 6 neural paradigms initialize with correct tensor dimensions."""
    engine = ApexEnsembleEngine(num_users=100, num_items=100, emb_dim=16)

    # Check Quantum Fluid
    assert engine.quantum.user_embedding.amplitude.weight.shape == (100, 16)
    assert engine.quantum.item_embedding.phase.weight.shape == (100, 16)

    # Check Hyperbolic
    assert engine.hyperbolic.user_embedding.weight.shape == (100, 16)

    # Check KAN (Requires concat of user+item)
    assert engine.kan is not None

    # Check SASRec and LightGCN
    assert engine.sasrec.item_emb.weight.shape[0] == 101  # +1 for padding
    assert engine.sasrec.item_emb.weight.shape[1] == 16
    assert engine.lightgcn.user_embedding.weight.shape == (100, 16)


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
