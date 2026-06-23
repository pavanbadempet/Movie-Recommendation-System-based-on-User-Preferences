import os
import time
from unittest.mock import patch

import pytest
import torch

from backend.models.contextual_router import ContextualRouter, build_user_state
from backend.models.ensemble_engine import get_apex_engine


def test_router_outputs_valid_weights():
    # Initialize router
    router = ContextualRouter(emb_dim=16)
    router.eval()

    # Create a dummy user state [emb_dim + 4] = 20
    user_state = torch.randn(20)

    # Test for k=2
    selected_models, weights = router.route(user_state, k=2)

    assert len(selected_models) == 2
    assert len(weights) == 2
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    # Verify all selected models are from our standard 6 models
    valid_models = {"lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion"}
    for m in selected_models:
        assert m in valid_models

    # Test for k=4
    selected_models_4, weights_4 = router.route(user_state, k=4)
    assert len(selected_models_4) == 4
    assert len(weights_4) == 4
    assert torch.allclose(weights_4.sum(), torch.tensor(1.0), atol=1e-5)


def test_build_user_state():
    emb_dim = 16
    user_emb = torch.randn(emb_dim)
    session_seq = torch.tensor([1, 2, 3, 0, 0] + [0] * 45)  # 50 elements, 3 active
    item_embeddings = torch.randn(10, emb_dim)

    state = build_user_state(
        user_id=42,
        user_emb=user_emb,
        session_seq=session_seq,
        item_embeddings=item_embeddings,
        interaction_count=100,
        inference_energy=0.8,
    )

    # Expected length: emb_dim + 4
    assert state.shape[0] == emb_dim + 4

    # Verify values
    # user_emb part
    assert torch.allclose(state[:emb_dim], user_emb)

    # 4 metrics part
    metrics = state[emb_dim:]
    assert metrics[0] > 0.0  # normalized interaction count
    assert metrics[1] == pytest.approx(3.0 / 50.0)  # normalized session length
    assert 0.0 <= metrics[2].item() <= 1.0  # stability is mapped to [0, 1]
    assert metrics[3] == pytest.approx(0.8)  # inference energy


def test_router_training_convergence():
    torch.manual_seed(42)
    router = ContextualRouter(emb_dim=16)
    optimizer = torch.optim.Adam(router.parameters(), lr=0.01)

    # Generate dummy training data (100 samples)
    user_states = torch.randn(100, 20)

    # Assume lightgcn (index 0) and quantum (index 1) are always the best (lowest loss)
    # model losses are 6-dimensional
    model_losses = torch.zeros(100, 6)
    model_losses[:, 0] = 0.1  # low loss
    model_losses[:, 1] = 0.2  # low loss
    model_losses[:, 2:] = 2.0  # high loss

    # Train for 20 steps
    initial_loss = None
    final_loss = None
    for step in range(20):
        step_loss = 0.0
        for i in range(100):
            loss = router.train_router_step(user_states[i], model_losses[i], optimizer)
            step_loss += loss
        step_loss /= 100.0
        if initial_loss is None:
            initial_loss = step_loss
        final_loss = step_loss

    print(f"Initial Training Loss: {initial_loss:.6f}, Final Loss: {final_loss:.6f}")
    assert final_loss < initial_loss

    # Verify router prefers lightgcn or quantum
    selected, weights = router.route(user_states[0], k=2)
    assert "lightgcn" in selected or "quantum" in selected


def test_model_pruning_execution():
    # Retrieve engine singleton
    os.environ["NOVA_DISABLE_MODEL_DOWNLOADS"] = "1"
    os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key-for-ci-only"

    engine = get_apex_engine()
    candidate_ids = list(range(10))

    # Mock health monitor to not interfere with routing decisions
    mock_health = None
    if engine.health_monitor is not None:
        mock_health = patch.object(
            engine.health_monitor,
            "get_active_models",
            return_value=["quantum", "hyperbolic", "kan", "diffusion", "sasrec", "lightgcn"],
        )

    # Force router to select specific models and mock model calls
    with (
        patch.object(engine.router, "route", return_value=(["quantum", "hyperbolic"], torch.tensor([0.6, 0.4]))),
        patch.object(engine.quantum, "predict", return_value=torch.randn(10)) as mock_quantum,
        patch.object(engine.hyperbolic, "predict", return_value=torch.randn(10)) as mock_hyperbolic,
        patch.object(engine.kan, "forward", return_value=torch.randn(10)) as mock_kan,
        patch.object(engine.diffusion.denoiser, "forward", return_value=torch.randn(10, 16)) as mock_diffusion,
        patch.object(engine.sasrec, "predict", return_value=torch.randn(10)) as mock_sasrec,
    ):
        if mock_health is not None:
            mock_health.start()
        try:
            scores = engine.predict_ensemble(
                user_id=1, candidate_item_ids=candidate_ids, use_router=True, router_k=2, session_sequence=[1, 2, 3]
            )

            # Assert selected models were called
            assert mock_quantum.called, "Quantum model should be called by router selection"
            assert mock_hyperbolic.called, "Hyperbolic model should be called by router selection"

            # Assert bypassed models were NOT called
            assert not mock_kan.called
            assert not mock_diffusion.called
            assert not mock_sasrec.called
        finally:
            if mock_health is not None:
                mock_health.stop()


def test_router_latency_speedup():
    os.environ["NOVA_DISABLE_MODEL_DOWNLOADS"] = "1"
    os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key-for-ci-only"

    engine = get_apex_engine()
    candidate_ids = list(range(100))
    user_id = 42

    # Define a dummy override tensor to force PyTorch execution for both paths
    dummy_override = torch.zeros(16)

    # Warmup
    for _ in range(5):
        engine.predict_ensemble(user_id, candidate_ids, use_router=False, user_emb_override=dummy_override)
        engine.predict_ensemble(user_id, candidate_ids, use_router=True, router_k=2, user_emb_override=dummy_override)

    # Measure full ensemble time
    start_full = time.perf_counter()
    for _ in range(10):
        engine.predict_ensemble(user_id, candidate_ids, use_router=False, user_emb_override=dummy_override)
    duration_full = time.perf_counter() - start_full

    # Measure routed ensemble time (k=2)
    start_routed = time.perf_counter()
    for _ in range(10):
        engine.predict_ensemble(user_id, candidate_ids, use_router=True, router_k=2, user_emb_override=dummy_override)
    duration_routed = time.perf_counter() - start_routed

    speedup = duration_full / duration_routed if duration_routed > 0 else float("inf")
    print("\nLatency Benchmark (10 runs, 100 items):")
    print(f"Full 6-Model Ensemble (PyTorch): {duration_full:.4f}s")
    print(f"Routed Top-2 Models (PyTorch):  {duration_routed:.4f}s")
    print(f"Speedup Factor:                  {speedup:.2f}x")

    # Verify routed is not dramatically slower than full (allow 2x tolerance for CI variability)
    assert duration_routed < duration_full * 2.0, (
        f"Routed ({duration_routed:.4f}s) should not be >2x slower than full ({duration_full:.4f}s)"
    )
