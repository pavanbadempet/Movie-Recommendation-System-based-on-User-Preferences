"""
Tests for the Rényi Differential Privacy (RDP) Privacy Budget Accountant.
"""

from __future__ import annotations

import os
import json
import tempfile
import datetime
import pytest
import torch
import numpy as np

from backend.privacy.privacy_preserving_ml import PrivacyBudgetAccountant
from backend.models.ensemble_engine import ApexEnsembleEngine


class TestPrivacyBudgetAccountant:
    """Unit tests for PrivacyBudgetAccountant class in isolation."""

    @pytest.fixture
    def temp_storage(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_initial_budget_status(self, temp_storage):
        accountant = PrivacyBudgetAccountant(storage_path=temp_storage, epsilon_max=10.0)
        status = accountant.get_user_budget_status(user_id=42)
        
        assert status["user_id"] == 42
        assert status["current_epsilon"] == 0.0
        assert status["remaining_epsilon"] == 10.0
        assert not status["is_exhausted"]
        assert len(status["rdp_spent"]) == len(accountant.orders)
        assert all(val == 0.0 for val in status["rdp_spent"].values())

    def test_query_rdp_increments(self, temp_storage):
        accountant = PrivacyBudgetAccountant(storage_path=temp_storage)
        
        # Test zero epsilon request
        zeros = accountant.compute_query_rdp(request_epsilon=0.0)
        assert all(val == 0.0 for val in zeros.values())

        # Test Gaussian mechanism increments
        g_increments = accountant.compute_query_rdp(request_epsilon=1.0, request_delta=1e-5, mechanism="gaussian")
        assert len(g_increments) == len(accountant.orders)
        for alpha, rdp in g_increments.items():
            assert rdp > 0.0
            # Under Gaussian, RDP(alpha) = alpha / (2 * sigma^2)
            # Higher alpha must have higher RDP cost
            assert g_increments[alpha] == pytest.approx(alpha * (g_increments[2.0] / 2.0))


        # Test Laplace mechanism increments
        l_increments = accountant.compute_query_rdp(request_epsilon=1.0, mechanism="laplace")
        assert len(l_increments) == len(accountant.orders)
        for alpha, rdp in l_increments.items():
            assert rdp > 0.0

    def test_composition_versus_naive(self, temp_storage):
        # With RDP composition, multiple small queries compose to a much smaller epsilon than naive addition.
        accountant = PrivacyBudgetAccountant(storage_path=temp_storage, epsilon_max=20.0)
        
        user_id = 100
        # Make 10 queries of eps=1.0
        for _ in range(10):
            allowed, remaining = accountant.check_and_deduct_budget(
                user_id=user_id,
                request_epsilon=1.0,
                request_delta=1e-5,
                mechanism="gaussian"
            )
            assert allowed
            
        status = accountant.get_user_budget_status(user_id=100)
        cumulative_eps = status["current_epsilon"]
        
        # Naive composition would be 10 * 1.0 = 10.0
        # RDP composition is tighter
        assert cumulative_eps < 6.0
        assert cumulative_eps > 1.0

    def test_budget_exhaustion(self, temp_storage):
        accountant = PrivacyBudgetAccountant(storage_path=temp_storage, epsilon_max=3.0)
        
        # Deduct a large budget request
        allowed, remaining = accountant.check_and_deduct_budget(user_id=77, request_epsilon=2.5)
        assert allowed
        assert remaining < 1.0

        # Try to deduct another cost that pushes it over the limit
        allowed_again, remaining_again = accountant.check_and_deduct_budget(user_id=77, request_epsilon=2.5)
        assert not allowed_again
        # Remaining budget should be the same as before the second attempt
        assert remaining_again == pytest.approx(remaining)

    def test_persistence(self, temp_storage):
        accountant1 = PrivacyBudgetAccountant(storage_path=temp_storage, epsilon_max=5.0)
        allowed, remaining = accountant1.check_and_deduct_budget(user_id=99, request_epsilon=2.0)
        assert allowed

        # Create a new instance pointing to the same file
        accountant2 = PrivacyBudgetAccountant(storage_path=temp_storage, epsilon_max=5.0)
        status = accountant2.get_user_budget_status(user_id=99)
        assert status["current_epsilon"] == pytest.approx(5.0 - remaining)
        assert status["remaining_epsilon"] == pytest.approx(remaining)

    def test_decay_budgets(self, temp_storage):
        accountant = PrivacyBudgetAccountant(storage_path=temp_storage, epsilon_max=10.0)
        allowed, remaining = accountant.check_and_deduct_budget(user_id=200, request_epsilon=5.0)
        assert allowed

        pre_decay_eps = accountant.get_user_budget_status(user_id=200)["current_epsilon"]
        assert pre_decay_eps > 0.0

        # Decay by 50%
        accountant.decay_budgets(factor=0.5)

        post_decay_status = accountant.get_user_budget_status(user_id=200)
        # RDP spent values decayed by 50% should lead to lower cumulative epsilon
        assert post_decay_status["current_epsilon"] < pre_decay_eps
        assert post_decay_status["remaining_epsilon"] > remaining

    def test_daily_reset(self, temp_storage):
        accountant = PrivacyBudgetAccountant(storage_path=temp_storage, epsilon_max=10.0)
        allowed, remaining = accountant.check_and_deduct_budget(user_id=300, request_epsilon=4.0)
        assert allowed

        status_before = accountant.get_user_budget_status(user_id=300)
        assert status_before["current_epsilon"] > 0.0

        # Manually alter the JSON file to simulate an old update date (yesterday)
        with open(temp_storage, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        yesterday_str = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        data["300"]["last_update"] = yesterday_str
        
        with open(temp_storage, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Get status again, which triggers auto-reset
        status_after = accountant.get_user_budget_status(user_id=300)
        assert status_after["current_epsilon"] == 0.0
        assert status_after["remaining_epsilon"] == 10.0
        assert status_after["last_update"] == datetime.date.today().isoformat()


class TestEnsemblePrivacyBudgetIntegration:
    """Integration tests verifying PrivacyBudgetAccountant inside ApexEnsembleEngine."""

    @pytest.fixture
    def engine(self):
        # We can construct a small engine
        return ApexEnsembleEngine(num_users=10, num_items=50, emb_dim=8)

    def test_engine_deducts_budget_and_falls_back(self, engine):
        assert engine.privacy_accountant is not None
        
        user_id = 5
        # Reset budget for user_id to ensure clean test state
        engine.privacy_accountant.reset_budget(user_id)
        
        # Override the accountant to use a lower maximum budget for easier test exhaustion
        engine.privacy_accountant.epsilon_max = 2.0
        
        # Generate a mock user embedding override
        user_emb = torch.randn(8)
        user_emb = user_emb / torch.norm(user_emb)
        candidate_ids = [2, 7, 12, 19]

        # First query: should be allowed and return normal scores
        # Epsilon = 1.0, under epsilon_max = 2.0
        os.environ["APEX_DP_EPSILON"] = "1.0"
        scores_1 = engine.predict_ensemble(
            user_id=user_id,
            candidate_item_ids=candidate_ids,
            user_emb_override=user_emb
        )
        assert len(scores_1) == len(candidate_ids)
        status = engine.privacy_accountant.get_user_budget_status(user_id)
        assert not status["is_exhausted"]
        assert status["current_epsilon"] > 0.0

        # Run multiple queries to exhaust budget
        for _ in range(5):
            engine.predict_ensemble(
                user_id=user_id,
                candidate_item_ids=candidate_ids,
                user_emb_override=user_emb
            )

        status_after = engine.privacy_accountant.get_user_budget_status(user_id)
        assert status_after["is_exhausted"]

        # Request again under exhausted budget
        # Verify that it falls back to user 0 / zero embedding fallback
        # Let's get predictions under fallback
        scores_fallback = engine.predict_ensemble(
            user_id=user_id,
            candidate_item_ids=candidate_ids,
            user_emb_override=user_emb,
            session_sequence=[0]*50
        )

        # Get predictions directly for user 0 / zero override to verify equivalence
        scores_user0_zero_override = engine.predict_ensemble(
            user_id=0,
            candidate_item_ids=candidate_ids,
            user_emb_override=torch.zeros(8),
            session_sequence=[0]*50
        )

        for cid in candidate_ids:
            # Fallback output should align with the zero embedding/dummy user predictions
            assert scores_fallback[cid] == pytest.approx(scores_user0_zero_override[cid], abs=1e-5)

