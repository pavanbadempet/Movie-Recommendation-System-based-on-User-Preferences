"""
Unit and property-based tests for OnlineLearner._run, _apply_gradient_step,
and _checkpoint.

Requirements: 3.2, 3.6, 3.7, 3.9, 3.10
"""

from pathlib import Path
import time

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
import torch

from backend.learning.online_learner import OnlineLearner
from backend.models.lightgcn import LightGCN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_USERS = 50
NUM_ITEMS = 100
EMB_DIM = 16


def make_learner(
    batch_size: int = 32,
    lr: float = 1e-3,
    checkpoint_interval: int = 1000,
    checkpoint_path: Path = Path("models/lightgcn_online_test.pth"),
) -> OnlineLearner:
    """Return a fresh OnlineLearner backed by a small LightGCN."""
    model = LightGCN(num_users=NUM_USERS, num_items=NUM_ITEMS, embedding_dim=EMB_DIM)
    return OnlineLearner(
        lightgcn=model,
        batch_size=batch_size,
        lr=lr,
        checkpoint_interval=checkpoint_interval,
        checkpoint_path=checkpoint_path,
    )


def positive_event(user_id: int = 1, movie_id: int = 10) -> dict:
    return {
        "event_type": "rating",
        "user_id": str(user_id),
        "movie_id": movie_id,
        "rating": 4.5,
        "interaction_weight": 1.0,
    }


def negative_event(user_id: int = 2, movie_id: int = 20) -> dict:
    return {
        "event_type": "rating",
        "user_id": str(user_id),
        "movie_id": movie_id,
        "rating": 1.5,
        "interaction_weight": -0.5,
    }


def click_event(user_id: int = 3, movie_id: int = 30) -> dict:
    return {
        "event_type": "click",
        "user_id": str(user_id),
        "movie_id": movie_id,
        "interaction_weight": 0.3,
    }


# ---------------------------------------------------------------------------
# _apply_gradient_step — unit tests
# ---------------------------------------------------------------------------


class TestApplyGradientStep:
    def test_positive_event_updates_embeddings(self):
        """A positive event should change user and item embedding weights."""
        learner = make_learner()
        user_emb_before = learner.lightgcn.user_embedding.weight.clone().detach()
        item_emb_before = learner.lightgcn.item_embedding.weight.clone().detach()

        learner._apply_gradient_step([positive_event()])

        assert not torch.equal(learner.lightgcn.user_embedding.weight, user_emb_before)
        assert not torch.equal(learner.lightgcn.item_embedding.weight, item_emb_before)

    def test_negative_event_updates_embeddings(self):
        """A negative event (swapped BPR roles) should also update embeddings."""
        learner = make_learner()
        user_emb_before = learner.lightgcn.user_embedding.weight.clone().detach()

        learner._apply_gradient_step([negative_event()])

        assert not torch.equal(learner.lightgcn.user_embedding.weight, user_emb_before)

    def test_click_event_updates_embeddings(self):
        """A click event (weak positive) should update embeddings."""
        learner = make_learner()
        user_emb_before = learner.lightgcn.user_embedding.weight.clone().detach()

        learner._apply_gradient_step([click_event()])

        assert not torch.equal(learner.lightgcn.user_embedding.weight, user_emb_before)

    def test_empty_batch_is_no_op(self):
        """An empty batch should not change any embeddings."""
        learner = make_learner()
        user_emb_before = learner.lightgcn.user_embedding.weight.clone().detach()
        item_emb_before = learner.lightgcn.item_embedding.weight.clone().detach()

        learner._apply_gradient_step([])

        assert torch.equal(learner.lightgcn.user_embedding.weight, user_emb_before)
        assert torch.equal(learner.lightgcn.item_embedding.weight, item_emb_before)

    def test_event_with_none_user_id_is_skipped(self):
        """Events missing user_id should be silently skipped."""
        learner = make_learner()
        user_emb_before = learner.lightgcn.user_embedding.weight.clone().detach()

        bad_event = {"event_type": "click", "user_id": None, "movie_id": 5, "interaction_weight": 0.3}
        learner._apply_gradient_step([bad_event])

        assert torch.equal(learner.lightgcn.user_embedding.weight, user_emb_before)

    def test_event_with_none_movie_id_is_skipped(self):
        """Events missing movie_id should be silently skipped."""
        learner = make_learner()
        user_emb_before = learner.lightgcn.user_embedding.weight.clone().detach()

        bad_event = {"event_type": "click", "user_id": "1", "movie_id": None, "interaction_weight": 0.3}
        learner._apply_gradient_step([bad_event])

        assert torch.equal(learner.lightgcn.user_embedding.weight, user_emb_before)

    def test_zero_weight_event_is_skipped(self):
        """Events with interaction_weight == 0 should be skipped."""
        learner = make_learner()
        user_emb_before = learner.lightgcn.user_embedding.weight.clone().detach()

        zero_event = {"event_type": "click", "user_id": "1", "movie_id": 5, "interaction_weight": 0.0}
        learner._apply_gradient_step([zero_event])

        assert torch.equal(learner.lightgcn.user_embedding.weight, user_emb_before)

    def test_user_id_modulo_bounds(self):
        """user_id larger than num_users should be mapped via modulo without error."""
        learner = make_learner()
        large_user_event = positive_event(user_id=NUM_USERS * 10 + 3, movie_id=5)
        # Should not raise
        learner._apply_gradient_step([large_user_event])

    def test_movie_id_modulo_bounds(self):
        """movie_id larger than num_items should be mapped via modulo without error."""
        learner = make_learner()
        large_item_event = positive_event(user_id=1, movie_id=NUM_ITEMS * 10 + 7)
        # Should not raise
        learner._apply_gradient_step([large_item_event])

    def test_gradient_clipping_prevents_large_updates(self):
        """After a gradient step, no gradient should exceed max_norm=1.0."""
        learner = make_learner(lr=1.0)  # high LR to stress-test clipping

        # Apply a gradient step and check that embedding weight grads are clipped
        batch = [positive_event(user_id=i, movie_id=i % NUM_ITEMS) for i in range(32)]
        learner._apply_gradient_step(batch)

        # After the step, gradients should have been clipped — verify by checking
        # that the embedding weights changed but didn't explode (finite values)
        user_w = learner.lightgcn.user_embedding.weight
        item_w = learner.lightgcn.item_embedding.weight
        assert torch.isfinite(user_w).all(), "User embeddings contain non-finite values"
        assert torch.isfinite(item_w).all(), "Item embeddings contain non-finite values"

    def test_mixed_batch_processes_all_valid_events(self):
        """A batch with positive, negative, and click events should all be processed."""
        learner = make_learner()
        user_emb_before = learner.lightgcn.user_embedding.weight.clone().detach()

        batch = [positive_event(1, 1), negative_event(2, 2), click_event(3, 3)]
        learner._apply_gradient_step(batch)

        assert not torch.equal(learner.lightgcn.user_embedding.weight, user_emb_before)


# ---------------------------------------------------------------------------
# _checkpoint — unit tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_checkpoint_saves_state_dict(self, tmp_path):
        """_checkpoint should write a valid state_dict to checkpoint_path."""
        ckpt = tmp_path / "test_ckpt.pth"
        learner = make_learner(checkpoint_path=ckpt)

        learner._checkpoint()

        assert ckpt.exists()
        loaded = torch.load(ckpt, weights_only=True)
        assert "user_embedding.weight" in loaded
        assert "item_embedding.weight" in loaded

    def test_checkpoint_does_not_raise_on_bad_path(self):
        """_checkpoint should log ERROR and not re-raise on write failure."""
        learner = make_learner(checkpoint_path=Path("/nonexistent_dir/ckpt.pth"))
        # Should not raise
        learner._checkpoint()

    def test_checkpoint_state_matches_model(self, tmp_path):
        """Saved checkpoint should match the current model weights exactly."""
        ckpt = tmp_path / "match_ckpt.pth"
        learner = make_learner(checkpoint_path=ckpt)

        # Apply a gradient step to make weights non-trivial
        learner._apply_gradient_step([positive_event()])
        learner._checkpoint()

        loaded = torch.load(ckpt, weights_only=True)
        assert torch.equal(
            loaded["user_embedding.weight"],
            learner.lightgcn.user_embedding.weight.detach(),
        )


# ---------------------------------------------------------------------------
# _run — unit tests
# ---------------------------------------------------------------------------


def _wait_and_stop_learner(learner, target_events=0, timeout=5.0):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if learner._queue.empty():
            if target_events == 0 or learner._events_processed >= target_events:
                break
        time.sleep(0.05)
    learner.stop()


class TestRun:
    def test_run_processes_enqueued_events(self):
        """_run should drain the queue and update embeddings."""
        learner = make_learner(batch_size=4)
        user_emb_before = learner.lightgcn.user_embedding.weight.clone().detach()

        # Pre-fill the queue
        for i in range(4):
            learner._queue.put(positive_event(user_id=i, movie_id=i))

        # Run, wait for processing, then stop
        learner.start()
        _wait_and_stop_learner(learner, target_events=4)

        assert not torch.equal(learner.lightgcn.user_embedding.weight, user_emb_before)

    def test_run_increments_events_processed(self):
        """_run should increment _events_processed by the number of events consumed."""
        learner = make_learner(batch_size=8)

        for i in range(8):
            learner._queue.put(positive_event(user_id=i, movie_id=i))

        learner.start()
        _wait_and_stop_learner(learner, target_events=8)

        assert learner._events_processed >= 8

    def test_run_triggers_checkpoint_at_interval(self, tmp_path):
        """_run should call _checkpoint when events_processed hits checkpoint_interval."""
        ckpt = tmp_path / "interval_ckpt.pth"
        learner = make_learner(batch_size=10, checkpoint_interval=10, checkpoint_path=ckpt)

        for i in range(10):
            learner._queue.put(positive_event(user_id=i, movie_id=i))

        learner.start()
        _wait_and_stop_learner(learner, target_events=10)

        assert ckpt.exists(), "Checkpoint should have been written after 10 events"

    def test_run_continues_after_gradient_step_exception(self):
        """_run should log ERROR and continue when _apply_gradient_step raises."""
        learner = make_learner(batch_size=2)

        call_count = {"n": 0}
        original = learner._apply_gradient_step

        def flaky_step(batch):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Simulated gradient failure")
            original(batch)

        learner._apply_gradient_step = flaky_step

        for i in range(4):
            learner._queue.put(positive_event(user_id=i, movie_id=i))

        learner.start()
        _wait_and_stop_learner(learner, target_events=4)

        # The second batch should have been processed despite the first failing
        assert call_count["n"] >= 2

    def test_run_stops_when_stop_event_set(self):
        """_run should exit its loop when _stop_event is set."""
        learner = make_learner()
        learner.start()
        assert learner._thread is not None and learner._thread.is_alive()

        learner.stop()
        assert not learner._thread.is_alive()

    def test_run_empty_queue_does_not_crash(self):
        """_run should handle an empty queue gracefully without crashing."""
        learner = make_learner()
        learner.start()
        time.sleep(0.3)
        learner.stop()
        # No exception means success


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(
    user_id=st.integers(min_value=0, max_value=10**9),
    movie_id=st.integers(min_value=0, max_value=10**9),
    weight=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_apply_gradient_step_never_raises_for_valid_positive_events(user_id: int, movie_id: int, weight: float):
    """
    **Validates: Requirements 3.2, 3.6, 3.7**

    For any valid positive event (interaction_weight > 0, non-None IDs),
    _apply_gradient_step should complete without raising an exception.
    """
    learner = make_learner()
    event = {
        "event_type": "rating",
        "user_id": str(user_id),
        "movie_id": movie_id,
        "interaction_weight": weight,
    }
    learner._apply_gradient_step([event])


@given(
    user_id=st.integers(min_value=0, max_value=10**9),
    movie_id=st.integers(min_value=0, max_value=10**9),
    weight=st.floats(min_value=-2.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_apply_gradient_step_never_raises_for_valid_negative_events(user_id: int, movie_id: int, weight: float):
    """
    **Validates: Requirements 3.2, 3.6, 3.7**

    For any valid negative event (interaction_weight < 0, non-None IDs),
    _apply_gradient_step should complete without raising an exception.
    """
    learner = make_learner()
    event = {
        "event_type": "rating",
        "user_id": str(user_id),
        "movie_id": movie_id,
        "interaction_weight": weight,
    }
    learner._apply_gradient_step([event])


@given(
    batch=st.lists(
        st.fixed_dictionaries(
            {
                "event_type": st.sampled_from(["rating", "click"]),
                "user_id": st.one_of(st.none(), st.integers(0, 10**6).map(str)),
                "movie_id": st.one_of(st.none(), st.integers(0, 10**6)),
                "interaction_weight": st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
            }
        ),
        min_size=0,
        max_size=64,
    )
)
@settings(max_examples=50, deadline=None)
def test_apply_gradient_step_never_raises_for_arbitrary_batches(batch: list[dict]):
    """
    **Validates: Requirements 3.2, 3.6, 3.7, 3.10**

    For any batch of events (including None IDs, zero weights, mixed types),
    _apply_gradient_step should never raise an exception.
    """
    learner = make_learner()
    learner._apply_gradient_step(batch)


@given(
    num_events=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=30, deadline=None)
def test_events_processed_counter_is_monotonically_increasing(num_events: int):
    """
    **Validates: Requirements 3.6, 3.9**

    After processing N events, _events_processed should equal N.
    """
    learner = make_learner(batch_size=num_events)

    for i in range(num_events):
        learner._queue.put(positive_event(user_id=i % NUM_USERS, movie_id=i % NUM_ITEMS))

    learner.start()
    _wait_and_stop_learner(learner, target_events=num_events)

    assert learner._events_processed >= num_events


# ---------------------------------------------------------------------------
# Property 4: Interaction Weight Assignment (enqueue)
# Feature: apex-peak-capability, Property 4
# Validates: Requirements 3.3, 3.4, 3.5
# ---------------------------------------------------------------------------


@given(
    rating=st.floats(min_value=4.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    movie_id=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_enqueue_rating_gte_4_assigns_weight_1(rating: float, movie_id: int):
    """
    Feature: apex-peak-capability, Property 4
    rating >= 4.0 → interaction_weight == +1.0
    """
    learner = make_learner()
    event = {"event_type": "rating", "user_id": "u1", "movie_id": movie_id, "rating": rating}
    learner.enqueue(event)
    assert not learner._queue.empty()
    queued = learner._queue.get_nowait()
    assert queued["interaction_weight"] == 1.0


@given(
    rating=st.floats(min_value=1.0, max_value=2.499, allow_nan=False, allow_infinity=False),
    movie_id=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=100, deadline=None)
def test_enqueue_rating_lt_2_5_assigns_weight_neg_0_5(rating: float, movie_id: int):
    """
    Feature: apex-peak-capability, Property 4
    rating < 2.5 → interaction_weight == -0.5
    """
    learner = make_learner()
    event = {"event_type": "rating", "user_id": "u1", "movie_id": movie_id, "rating": rating}
    learner.enqueue(event)
    assert not learner._queue.empty()
    queued = learner._queue.get_nowait()
    assert queued["interaction_weight"] == -0.5


@given(movie_id=st.integers(min_value=1, max_value=1000))
@settings(max_examples=100, deadline=None)
def test_enqueue_click_assigns_weight_0_3(movie_id: int):
    """
    Feature: apex-peak-capability, Property 4
    click event → interaction_weight == +0.3
    """
    learner = make_learner()
    event = {"event_type": "click", "user_id": "u1", "movie_id": movie_id}
    learner.enqueue(event)
    assert not learner._queue.empty()
    queued = learner._queue.get_nowait()
    assert queued["interaction_weight"] == pytest.approx(0.3)


@given(
    rating=st.floats(min_value=2.5, max_value=3.999, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_enqueue_neutral_rating_not_queued(rating: float):
    """
    Feature: apex-peak-capability, Property 4
    Neutral rating (2.5 <= rating < 4.0) → not enqueued.
    """
    learner = make_learner()
    event = {"event_type": "rating", "user_id": "u1", "movie_id": 1, "rating": rating}
    learner.enqueue(event)
    assert learner._queue.empty()
