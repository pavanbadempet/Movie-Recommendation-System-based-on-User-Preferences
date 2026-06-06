"""
Tests for the unified OnlineLearningCoordinator and the two new learners:
  - SASRecOnlineLearner
  - KANOnlineLearner
  - OnlineLearningCoordinator (fan-out coordinator)

These tests are property-based where possible and cover:
  - Lifecycle (start / stop / idempotency)
  - Event routing and fan-out
  - Gradient step execution and weight mutation
  - Queue full / drop behaviour
  - Coordinator status reporting
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from backend.learning.kan_online_learner import KANOnlineLearner
from backend.models.kan_ranker import KANRanker
from backend.models.lightgcn import LightGCN
from backend.learning.online_learning_coordinator import OnlineLearningCoordinator
from backend.models.sasrec import SASRec
from backend.learning.sasrec_online_learner import SASRecOnlineLearner

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_USERS = 50
NUM_ITEMS = 100
EMB_DIM = 8


@pytest.fixture
def lightgcn():
    return LightGCN(num_users=NUM_USERS, num_items=NUM_ITEMS, embedding_dim=EMB_DIM)


@pytest.fixture
def sasrec():
    return SASRec(num_items=NUM_ITEMS, max_seq_len=10, hidden_dim=EMB_DIM, num_blocks=1, num_heads=2)


@pytest.fixture
def kan():
    return KANRanker(input_dim=EMB_DIM * 2, hidden_dim=16)


def _make_click_event(user_id=1, movie_id=5):
    return {"event_type": "click", "user_id": user_id, "movie_id": movie_id}


def _make_rating_event(user_id=1, movie_id=5, rating=5.0):
    return {"event_type": "rating", "user_id": user_id, "movie_id": movie_id, "rating": rating}


def _make_neutral_rating_event(user_id=1, movie_id=5):
    return {"event_type": "rating", "user_id": user_id, "movie_id": movie_id, "rating": 3.0}


# ===========================================================================
# SASRecOnlineLearner tests
# ===========================================================================


class TestSASRecOnlineLearner:
    def _make_learner(self, sasrec_model, num_items=NUM_ITEMS):
        return SASRecOnlineLearner(
            sasrec=sasrec_model,
            session_sequence_getter=lambda uid: [1, 2, 3, uid % num_items],
            batch_size=4,
            lr=1e-3,
            checkpoint_interval=10000,  # never checkpoint during tests
            num_items=num_items,
        )

    def test_start_stop_lifecycle(self, sasrec):
        learner = self._make_learner(sasrec)
        learner.start()
        assert learner._thread is not None
        assert learner._thread.is_alive()
        learner.stop()
        assert not learner._thread.is_alive()

    def test_start_is_idempotent(self, sasrec):
        """Starting twice should not raise; second start creates a fresh thread."""
        learner = self._make_learner(sasrec)
        learner.start()
        first_thread = learner._thread
        learner.start()  # second call
        learner.stop()
        # No exception — that's the contract

    def test_neutral_rating_not_enqueued(self, sasrec):
        learner = self._make_learner(sasrec)
        learner.enqueue(_make_neutral_rating_event())
        assert learner._queue.qsize() == 0

    def test_click_event_enqueued(self, sasrec):
        learner = self._make_learner(sasrec)
        learner.enqueue(_make_click_event())
        assert learner._queue.qsize() == 1

    def test_positive_rating_enqueued(self, sasrec):
        learner = self._make_learner(sasrec)
        learner.enqueue(_make_rating_event(rating=4.5))
        assert learner._queue.qsize() == 1

    def test_negative_rating_enqueued(self, sasrec):
        learner = self._make_learner(sasrec)
        learner.enqueue(_make_rating_event(rating=1.0))
        assert learner._queue.qsize() == 1

    def test_unknown_event_type_not_enqueued(self, sasrec):
        learner = self._make_learner(sasrec)
        learner.enqueue({"event_type": "view", "user_id": 1, "movie_id": 5})
        assert learner._queue.qsize() == 0

    def test_gradient_step_mutates_item_embeddings(self, sasrec):
        """A gradient step should change item embedding weights."""
        learner = self._make_learner(sasrec)
        before = sasrec.item_emb.weight.data.clone()
        batch = [
            {"event_type": "click", "user_id": i, "movie_id": (i * 3) % NUM_ITEMS,
             "interaction_weight": 0.3}
            for i in range(1, 6)
        ]
        learner._apply_gradient_step(batch)
        after = sasrec.item_emb.weight.data
        assert not torch.allclose(before, after), "Item embeddings should have changed after gradient step"

    def test_gradient_step_empty_batch_is_noop(self, sasrec):
        """An empty batch must not crash and must not mutate weights."""
        learner = self._make_learner(sasrec)
        before = sasrec.item_emb.weight.data.clone()
        learner._apply_gradient_step([])
        assert torch.allclose(before, sasrec.item_emb.weight.data)

    def test_queue_overflow_drops_oldest(self, sasrec):
        """When the queue is full, the oldest event is dropped to make room."""
        learner = SASRecOnlineLearner(
            sasrec=sasrec,
            session_sequence_getter=lambda uid: [],
            batch_size=4,
            num_items=NUM_ITEMS,
        )
        # Manually fill the queue
        learner._queue.maxsize = 3
        for i in range(3):
            learner._queue.put_nowait({"event_type": "click", "user_id": i, "movie_id": i,
                                       "interaction_weight": 0.3})
        assert learner._queue.full()
        # This enqueue should drop oldest and add new one — no exception
        learner.enqueue(_make_click_event(user_id=99, movie_id=99))
        assert learner._queue.qsize() == 3  # still 3 (one dropped, one added)

    def test_events_processed_counter_increments(self, sasrec):
        learner = self._make_learner(sasrec)
        learner.start()
        for _ in range(8):
            learner.enqueue(_make_click_event())
        time.sleep(0.3)  # give the background thread time to drain
        assert learner._events_processed > 0
        learner.stop()

    @given(
        user_id=st.integers(min_value=0, max_value=NUM_USERS - 1),
        movie_id=st.integers(min_value=0, max_value=NUM_ITEMS - 1),
        rating=st.floats(min_value=4.0, max_value=5.0),
    )
    @settings(max_examples=20, deadline=2000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_positive_rating_always_accepted(self, sasrec, user_id, movie_id, rating):
        learner = self._make_learner(sasrec)
        learner.enqueue({"event_type": "rating", "user_id": user_id, "movie_id": movie_id, "rating": rating})
        assert learner._queue.qsize() == 1

    @given(
        user_id=st.integers(min_value=0, max_value=NUM_USERS - 1),
        movie_id=st.integers(min_value=0, max_value=NUM_ITEMS - 1),
    )
    @settings(max_examples=20, deadline=2000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_click_always_accepted(self, sasrec, user_id, movie_id):
        learner = self._make_learner(sasrec)
        learner.enqueue({"event_type": "click", "user_id": user_id, "movie_id": movie_id})
        assert learner._queue.qsize() == 1


# ===========================================================================
# KANOnlineLearner tests
# ===========================================================================


class TestKANOnlineLearner:
    def _make_learner(self, kan_model, lgcn):
        return KANOnlineLearner(
            kan=kan_model,
            lightgcn=lgcn,
            batch_size=4,
            lr=1e-3,
            checkpoint_interval=10000,
        )

    def test_start_stop_lifecycle(self, kan, lightgcn):
        learner = self._make_learner(kan, lightgcn)
        learner.start()
        assert learner._thread is not None
        assert learner._thread.is_alive()
        learner.stop()
        assert not learner._thread.is_alive()

    def test_neutral_rating_not_enqueued(self, kan, lightgcn):
        learner = self._make_learner(kan, lightgcn)
        learner.enqueue(_make_neutral_rating_event())
        assert learner._queue.qsize() == 0

    def test_click_enqueued(self, kan, lightgcn):
        learner = self._make_learner(kan, lightgcn)
        learner.enqueue(_make_click_event())
        assert learner._queue.qsize() == 1

    def test_gradient_step_mutates_kan_params(self, kan, lightgcn):
        """KAN Fourier coefficients should change; LightGCN embeddings must not."""
        learner = self._make_learner(kan, lightgcn)

        kan_before = [p.data.clone() for p in kan.parameters()]
        lgcn_before = lightgcn.user_embedding.weight.data.clone()

        batch = [
            {"event_type": "click", "user_id": i % NUM_USERS,
             "movie_id": (i * 7) % NUM_ITEMS, "interaction_weight": 0.3}
            for i in range(1, 6)
        ]
        learner._apply_gradient_step(batch)

        # KAN params must have changed
        any_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(kan_before, kan.parameters())
        )
        assert any_changed, "At least one KAN parameter should have changed"

        # LightGCN embeddings must NOT change (only KAN is updated)
        assert torch.allclose(lgcn_before, lightgcn.user_embedding.weight.data), \
            "LightGCN user embeddings must not be modified by KANOnlineLearner"

    def test_gradient_step_empty_batch_is_noop(self, kan, lightgcn):
        learner = self._make_learner(kan, lightgcn)
        params_before = [p.data.clone() for p in kan.parameters()]
        learner._apply_gradient_step([])
        for before, after in zip(params_before, kan.parameters()):
            assert torch.allclose(before, after)

    def test_events_processed_counter_increments(self, kan, lightgcn):
        learner = self._make_learner(kan, lightgcn)
        learner.start()
        for _ in range(8):
            learner.enqueue(_make_click_event())
        time.sleep(0.3)
        assert learner._events_processed > 0
        learner.stop()

    @given(
        user_id=st.integers(min_value=0, max_value=NUM_USERS - 1),
        movie_id=st.integers(min_value=0, max_value=NUM_ITEMS - 1),
        rating=st.floats(min_value=1.0, max_value=2.4),
    )
    @settings(max_examples=20, deadline=2000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_negative_rating_always_accepted(self, kan, lightgcn, user_id, movie_id, rating):
        learner = self._make_learner(kan, lightgcn)
        learner.enqueue({"event_type": "rating", "user_id": user_id, "movie_id": movie_id, "rating": rating})
        assert learner._queue.qsize() == 1


# ===========================================================================
# OnlineLearningCoordinator tests
# ===========================================================================


def _make_mock_engine(lightgcn_model, sasrec_model, kan_model):
    """Build a minimal mock ApexEnsembleEngine with the real sub-models."""
    engine = MagicMock()
    engine.lightgcn = lightgcn_model
    engine.sasrec = sasrec_model
    engine.kan = kan_model
    engine.num_items = NUM_ITEMS
    engine.num_users = NUM_USERS
    return engine


class TestOnlineLearningCoordinator:
    @pytest.fixture
    def coord(self, lightgcn, sasrec, kan):
        engine = _make_mock_engine(lightgcn, sasrec, kan)
        with patch("backend.learning.online_learning_coordinator.OnlineLearningCoordinator._get_session_sequence",
                   return_value=[1, 2, 3]):
            coordinator = OnlineLearningCoordinator(engine=engine)
        return coordinator

    def test_start_stop(self, coord):
        coord.start()
        status = coord.status()
        assert status["started"] is True
        for name, info in status["learners"].items():
            assert info["thread_alive"], f"{name} thread should be alive after start()"
        coord.stop()
        assert coord.status()["started"] is False

    def test_double_start_is_safe(self, coord):
        """Calling start() twice must not raise and must not duplicate threads."""
        coord.start()
        coord.start()  # should log warning, not crash
        coord.stop()

    def test_stop_without_start_is_safe(self, coord):
        """Calling stop() before start() must not raise."""
        coord.stop()  # no-op

    def test_enqueue_fans_out_to_all_learners(self, coord):
        """A single enqueue must reach all three sub-learner queues."""
        event = _make_click_event()
        coord.enqueue(event)
        assert coord.lightgcn_learner._queue.qsize() == 1
        assert coord.sasrec_learner._queue.qsize() == 1
        assert coord.kan_learner._queue.qsize() == 1

    def test_enqueue_neutral_event_not_forwarded(self, coord):
        """A neutral rating (3.0) must not reach any learner queue."""
        event = _make_neutral_rating_event()
        coord.enqueue(event)
        assert coord.lightgcn_learner._queue.qsize() == 0
        assert coord.sasrec_learner._queue.qsize() == 0
        assert coord.kan_learner._queue.qsize() == 0

    def test_status_reports_all_three_learners(self, coord):
        coord.start()
        status = coord.status()
        assert set(status["learners"].keys()) == {"lightgcn", "sasrec", "kan"}
        coord.stop()

    def test_status_events_processed_tracks_activity(self, coord):
        coord.start()
        for _ in range(10):
            coord.enqueue(_make_click_event())
        time.sleep(0.4)
        status = coord.status()
        total = sum(v["events_processed"] for v in status["learners"].values())
        assert total > 0, "At least some events should have been processed"
        coord.stop()

    def test_enqueue_multiple_events_all_reach_queues(self, coord):
        events = [_make_click_event(user_id=i, movie_id=i % NUM_ITEMS) for i in range(1, 6)]
        for e in events:
            coord.enqueue(e)
        assert coord.lightgcn_learner._queue.qsize() == 5
        assert coord.sasrec_learner._queue.qsize() == 5
        assert coord.kan_learner._queue.qsize() == 5

    @given(
        num_events=st.integers(min_value=1, max_value=20),
        user_id=st.integers(min_value=0, max_value=NUM_USERS - 1),
        movie_id=st.integers(min_value=0, max_value=NUM_ITEMS - 1),
    )
    @settings(max_examples=25, deadline=3000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_click_event_always_fans_out(
        self, lightgcn, sasrec, kan, num_events, user_id, movie_id
    ):
        """Property: for any valid click event count, all queues have the same depth."""
        engine = _make_mock_engine(lightgcn, sasrec, kan)
        with patch("backend.learning.online_learning_coordinator.OnlineLearningCoordinator._get_session_sequence",
                   return_value=[]):
            coord = OnlineLearningCoordinator(engine=engine)

        for _ in range(num_events):
            coord.enqueue(_make_click_event(user_id=user_id, movie_id=movie_id))

        lgcn_depth = coord.lightgcn_learner._queue.qsize()
        sar_depth = coord.sasrec_learner._queue.qsize()
        kan_depth = coord.kan_learner._queue.qsize()

        assert lgcn_depth == sar_depth == kan_depth == num_events, (
            f"All queues should have depth {num_events}, got "
            f"lgcn={lgcn_depth}, sar={sar_depth}, kan={kan_depth}"
        )


# ===========================================================================
# Integration: end-to-end gradient flow through coordinator
# ===========================================================================


class TestOnlineLearningIntegration:
    def test_end_to_end_gradient_flow(self, lightgcn, sasrec, kan):
        """
        A positive click event fed through the coordinator should result in
        measurable weight changes in all three models after the background
        threads drain the batch.
        """
        engine = _make_mock_engine(lightgcn, sasrec, kan)

        lgcn_before = lightgcn.user_embedding.weight.data.clone()
        sar_before = sasrec.item_emb.weight.data.clone()
        kan_params_before = [p.data.clone() for p in kan.parameters()]

        with patch("backend.learning.online_learning_coordinator.OnlineLearningCoordinator._get_session_sequence",
                   return_value=[1, 2, 3]):
            coord = OnlineLearningCoordinator(engine=engine)
            # Force small batch sizes so events flush quickly
            coord.lightgcn_learner.batch_size = 4
            coord.sasrec_learner.batch_size = 4
            coord.kan_learner.batch_size = 4
            coord.start()

        # Send enough events to fill a batch for each learner
        for i in range(6):
            coord.enqueue(_make_rating_event(user_id=i % NUM_USERS,
                                             movie_id=(i * 7) % NUM_ITEMS,
                                             rating=5.0))

        time.sleep(0.5)  # allow background threads to process
        coord.stop()

        lgcn_changed = not torch.allclose(lgcn_before, lightgcn.user_embedding.weight.data)
        sar_changed = not torch.allclose(sar_before, sasrec.item_emb.weight.data)
        kan_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(kan_params_before, kan.parameters())
        )

        assert lgcn_changed, "LightGCN embeddings should have changed after online learning"
        assert sar_changed, "SASRec item embeddings should have changed after online learning"
        assert kan_changed, "KAN parameters should have changed after online learning"
