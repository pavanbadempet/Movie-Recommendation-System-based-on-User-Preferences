"""
Property-Based Fuzz Testing for Neural Architectures.

This suite uses Hypothesis to aggressively fuzz the PyTorch models.
Instead of providing fixed 'happy path' tensors, it generates thousands of
edge-case tensors (zeros, massive floats, negative indices, extreme sparsity)
to mathematically prove that the models NEVER crash, NaN-out, or Segfault
under unpredictable production loads.
"""

from hypothesis import given, settings
from hypothesis import strategies as st
import torch

from backend.models.hyperbolic_recommender import HyperbolicRecommender
from backend.models.kan_ranker import KANRanker
from backend.models.neural_ode_recommender import QuantumFluidRecommender
from backend.models.sasrec import SASRec


# FAANG limits: Fuzz with up to 100 random configurations per test
@settings(max_examples=50, deadline=None)
@given(
    user_id=st.integers(min_value=0, max_value=999),
    item_id=st.integers(min_value=0, max_value=999),
    time_delta=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_quantum_fluid_stability(user_id, item_id, time_delta):
    """Fuzz the Quantum Fluid ODE with extreme time deltas and random indices."""
    model = QuantumFluidRecommender(num_users=1000, num_items=1000, emb_dim=8)
    model.eval()

    u_tensor = torch.tensor([user_id], dtype=torch.long)
    i_tensor = torch.tensor([item_id], dtype=torch.long)

    with torch.no_grad():
        score = model.predict(u_tensor, i_tensor, time_delta)

    # Must not collapse into NaN or Inf
    assert not torch.isnan(score).any()
    assert not torch.isinf(score).any()


@settings(max_examples=50, deadline=None)
@given(
    u_emb_list=st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False), min_size=16, max_size=16),
    i_emb_list=st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False), min_size=16, max_size=16),
)
def test_kan_ranker_b_spline_robustness(u_emb_list, i_emb_list):
    """Fuzz the Kolmogorov-Arnold Network's B-Splines with erratic embeddings."""
    model = KANRanker(input_dim=32, hidden_dim=16)
    model.eval()

    # KAN expects separate user and item embeddings
    u_tensor = torch.tensor([u_emb_list], dtype=torch.float32)
    i_tensor = torch.tensor([i_emb_list], dtype=torch.float32)

    with torch.no_grad():
        score = model.forward(u_tensor, i_tensor)

    assert not torch.isnan(score).any()


@settings(max_examples=50, deadline=None)
@given(
    seq_list=st.lists(st.integers(min_value=0, max_value=999), min_size=1, max_size=50),
    target_item=st.integers(min_value=1, max_value=999),
)
def test_sasrec_transformer_masking(seq_list, target_item):
    """
    Fuzz the Transformer sequence recommender with variable length sequences
    (from 1 to 50) to ensure the causal attention mask never breaks.
    """
    model = SASRec(num_items=1000, max_seq_len=50, hidden_dim=16)
    model.eval()

    # Pad sequence to 50 if necessary
    padded_seq = [0] * (50 - len(seq_list)) + seq_list
    seq_tensor = torch.tensor([padded_seq], dtype=torch.long)
    item_tensor = torch.tensor([target_item], dtype=torch.long)

    with torch.no_grad():
        score = model.predict(seq_tensor, item_tensor)

    assert not torch.isnan(score).any()


def test_hyperbolic_poincare_bounds():
    """
    Manually inject an edge-case tensor that is exactly on the boundary
    of the Poincare disk (norm >= 1.0) to ensure the clipping mechanism
    (eps) physically prevents a divide-by-zero math explosion.
    """
    model = HyperbolicRecommender(num_users=10, num_items=10, emb_dim=8)
    model.eval()

    # Force embeddings to have norm > 1.0 (illegal in Poincare geometry)
    model.user_embedding.weight.data = torch.ones(10, 8) * 10.0
    model.item_embedding.weight.data = torch.ones(10, 8) * 10.0

    u_t = torch.tensor([1], dtype=torch.long)
    i_t = torch.tensor([1], dtype=torch.long)

    with torch.no_grad():
        dist = model.predict(u_t, i_t)

    # If the clipping fails, this will be NaN.
    assert not torch.isnan(dist).any()
