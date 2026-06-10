"""
Unified Online Learning Coordinator.

Coordinates incremental learning across all three online-capable models in
the ensemble:

  1. LightGCN   — via OnlineLearner      (DR weight: 0.005, graph CF)
  2. SASRec      — via SASRecOnlineLearner (DR weight: 0.659, sequential Transformer)
  3. KAN         — via KANOnlineLearner   (DR weight: 0.298, learnable edge functions)

A single ``enqueue(event)`` call fans out to all three learners.
All lifecycle operations (start / stop) are coordinated in one place.

Usage in lifespan (main.py):
    from backend.learning.online_learning_coordinator import OnlineLearningCoordinator
    coordinator = OnlineLearningCoordinator(engine=apex_engine)
    coordinator.start()
    ...
    coordinator.stop()

Enqueueing from serving path (recommendation_routes.py):
    coordinator.enqueue(event_dict)
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class OnlineLearningCoordinator:
    """
    Fan-out coordinator that routes live events to all online learners.

    Wraps:
      - ``OnlineLearner``         → LightGCN embedding updates
      - ``SASRecOnlineLearner``   → SASRec attention + item embedding updates
      - ``KANOnlineLearner``      → KAN Fourier coefficient updates

    All three operate in independent daemon threads so that one slow learner
    cannot back-pressure the others or the serving path.

    Thread-safety: ``enqueue`` is safe to call from any request handler thread.
    ``start`` and ``stop`` are intended to be called from the lifespan manager.
    """

    def __init__(self, engine) -> None:
        """
        Initialise from a live ApexEnsembleEngine instance.

        Args:
            engine: ``ApexEnsembleEngine`` — provides references to all models
                    and the session sequence cache.
        """
        from backend.learning.kan_online_learner import KANOnlineLearner
        from backend.learning.online_learner import OnlineLearner
        from backend.learning.sasrec_online_learner import SASRecOnlineLearner

        self._engine = engine
        self._lock = threading.Lock()
        self._started = False

        # 1. LightGCN learner (pre-existing)
        self.lightgcn_learner = OnlineLearner(lightgcn=engine.lightgcn)

        # 2. SASRec learner — session sequences come from the engine's live cache
        self.sasrec_learner = SASRecOnlineLearner(
            sasrec=engine.sasrec,
            session_sequence_getter=self._get_session_sequence,
            num_items=engine.num_items,
        )

        # 3. KAN learner — embeddings sourced from LightGCN (shared repr)
        self.kan_learner = KANOnlineLearner(
            kan=engine.kan,
            lightgcn=engine.lightgcn,
        )

        logger.info("OnlineLearningCoordinator initialised: LightGCN + SASRec + KAN online learners ready.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all three background learner threads."""
        with self._lock:
            if self._started:
                logger.warning("OnlineLearningCoordinator already started — ignoring.")
                return

            self.lightgcn_learner.start()
            self.sasrec_learner.start()
            self.kan_learner.start()
            self._started = True

        logger.info("OnlineLearningCoordinator: all learner threads started.")

    def stop(self) -> None:
        """Gracefully stop all background learner threads."""
        with self._lock:
            if not self._started:
                return

            self.lightgcn_learner.stop()
            self.sasrec_learner.stop()
            self.kan_learner.stop()
            self._started = False

        logger.info("OnlineLearningCoordinator: all learner threads stopped.")

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    def enqueue(self, event: dict) -> None:
        """
        Fan out a live event to all three learners simultaneously.

        Each learner independently decides whether the event is actionable
        (based on event_type and rating thresholds). This means a click event
        will be accepted by all three, while a neutral 3.0-star rating will be
        silently dropped by all three.

        This method is non-blocking and safe to call from request handlers.
        """
        self.lightgcn_learner.enqueue(event)
        self.sasrec_learner.enqueue(event)
        self.kan_learner.enqueue(event)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, object]:
        """Return a compact health snapshot for the /v1/platform/slo endpoint."""

        def _thread_alive(learner) -> bool:
            t = getattr(learner, "_thread", None)
            return t is not None and t.is_alive()

        def _queue_depth(learner) -> int:
            q = getattr(learner, "_queue", None)
            return q.qsize() if q is not None else -1

        return {
            "started": self._started,
            "learners": {
                "lightgcn": {
                    "thread_alive": _thread_alive(self.lightgcn_learner),
                    "events_processed": self.lightgcn_learner._events_processed,
                    "queue_depth": _queue_depth(self.lightgcn_learner),
                },
                "sasrec": {
                    "thread_alive": _thread_alive(self.sasrec_learner),
                    "events_processed": self.sasrec_learner._events_processed,
                    "queue_depth": _queue_depth(self.sasrec_learner),
                },
                "kan": {
                    "thread_alive": _thread_alive(self.kan_learner),
                    "events_processed": self.kan_learner._events_processed,
                    "queue_depth": _queue_depth(self.kan_learner),
                },
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_session_sequence(self, user_id: int) -> list[int]:
        """
        Retrieve the user's current session sequence from the engine's live cache.

        Uses the real-time feature updater first (sub-millisecond), falls back to
        the engine's background index, then returns an empty list for cold starts.
        """
        try:
            from backend.serving.realtime_feature_updater import get_user_session_sequence

            seq = get_user_session_sequence(user_id, max_len=self._engine.sasrec.max_seq_len)
            if seq:
                return seq
        except Exception as exc:
            logger.debug("Real-time session lookup failed for user %s: %s", user_id, exc)

        # Fall back to the engine's background event index
        try:
            from backend.models.ensemble_engine import _get_user_event_index

            index = _get_user_event_index()
            interactions = index.get(str(user_id), [])
            seq_len = self._engine.sasrec.max_seq_len
            recent = sorted(interactions, key=lambda x: x[0])[-seq_len:]
            return [item_id for _, item_id in recent]
        except Exception as exc:
            logger.debug("Background event index lookup failed for user %s: %s", user_id, exc)

        return []
