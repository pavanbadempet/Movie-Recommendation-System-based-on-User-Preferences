"""
Online Learner for incremental LightGCN embedding updates.

Consumes live click and rating events from an in-process queue and applies
incremental gradient updates to LightGCN user/item embeddings without a full
retraining cycle.

Requirements: 3.1, 3.3, 3.4, 3.5, 3.8
"""

import logging
from pathlib import Path
import queue
import random
import threading

import torch
import torch.nn.functional as F

from backend.models.lightgcn import LightGCN

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


class OnlineLearner:
    """
    Background learner that incrementally updates LightGCN embeddings from
    live click and rating events.

    Events are enqueued from the serving path and processed in batches by a
    daemon thread, applying BPR-style gradient steps without blocking inference.
    """

    def __init__(
        self,
        lightgcn: LightGCN,
        batch_size: int = 32,
        lr: float = 1e-4,
        checkpoint_interval: int = 1000,
        checkpoint_path: Path = MODELS_DIR / "lightgcn_online.pth",
    ) -> None:
        """
        Initialise the OnlineLearner.

        Args:
            lightgcn: The LightGCN model whose embeddings will be updated.
            batch_size: Number of interactions to accumulate before a gradient step.
            lr: Learning rate for the incremental optimizer.
            checkpoint_interval: Number of processed events between checkpoints.
            checkpoint_path: Path to persist updated LightGCN weights.
        """
        self.lightgcn = lightgcn
        self.batch_size = batch_size
        self.lr = lr
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path

        # Internal event queue — bounded to prevent unbounded memory growth
        self._queue: queue.Queue = queue.Queue(maxsize=10000)

        # Stop signal for the background thread
        self._stop_event: threading.Event = threading.Event()

        # Counter for processed events (used to trigger checkpoints)
        self._events_processed: int = 0

        # Background thread reference (set by start())
        self._thread: threading.Thread | None = None

        # Persistent Adam optimizer — created once so momentum state (m, v,
        # step count) accumulates across gradient steps. Creating a new
        # optimizer per batch discards momentum and degrades to plain SGD.
        self._optimizer: torch.optim.Adam | None = None

    def enqueue(self, event: dict) -> None:
        """
        Compute the interaction weight for an event and push it onto the queue.

        Weight mapping:
          - rating >= 4.0  → +1.0  (strong positive)
          - rating < 2.5   → -0.5  (negative)
          - click           → +0.3  (weak positive)
          - other types     → skipped (not enqueued)

        If the queue is full, the oldest event is dropped to make room and a
        WARNING is logged.

        Args:
            event: A normalized event dict (must contain at least 'event_type').
        """
        event_type = event.get("event_type")

        if event_type == "rating":
            rating = event.get("rating", 0.0)
            if rating >= 4.0:
                interaction_weight = 1.0
            elif rating < 2.5:
                interaction_weight = -0.5
            else:
                # Neutral rating — not actionable, skip
                return
        elif event_type == "click":
            interaction_weight = 0.3
        else:
            # Unknown event type — skip without enqueuing
            return

        # Build an augmented copy so the original dict is not mutated
        enriched_event = dict(event)
        enriched_event["interaction_weight"] = interaction_weight

        try:
            self._queue.put_nowait(enriched_event)
        except queue.Full:
            # Drop the oldest event to make room, then enqueue the new one
            try:
                dropped = self._queue.get_nowait()
                logger.warning(
                    "OnlineLearner queue full — dropped oldest event "
                    "(event_type=%s, user_id=%s, movie_id=%s) to make room.",
                    dropped.get("event_type"),
                    dropped.get("user_id"),
                    dropped.get("movie_id"),
                )
            except queue.Empty:
                pass  # Race condition: queue drained between Full and get
            self._queue.put_nowait(enriched_event)

    def start(self) -> None:
        """
        Start the background daemon thread that processes the event queue.

        The thread is a daemon so it will not prevent the process from exiting.
        """
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="OnlineLearner",
            daemon=True,
        )
        self._thread.start()
        logger.info("OnlineLearner background thread started.")

    def stop(self) -> None:
        """
        Signal the background thread to stop and wait up to 5 seconds for it
        to finish draining the queue.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("OnlineLearner thread did not stop within 5 seconds.")
        logger.info("OnlineLearner stopped.")

    # ------------------------------------------------------------------
    # Private methods — stubs to be implemented in task 5.2
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Main loop: drain the queue, accumulate a batch, apply a gradient step.

        Continuously reads from the event queue until the stop event is set.
        Collects up to batch_size events per iteration, applies a gradient step
        when the batch is non-empty, and triggers a checkpoint every
        checkpoint_interval processed events.
        """
        while not self._stop_event.is_set():
            batch: list[dict] = []

            # Drain up to batch_size events from the queue
            while len(batch) < self.batch_size:
                try:
                    event = self._queue.get(timeout=0.1)
                    batch.append(event)
                except queue.Empty:
                    break

            if not batch:
                continue

            try:
                self._apply_gradient_step(batch)
                self._events_processed += len(batch)

                if self._events_processed % self.checkpoint_interval == 0:
                    self._checkpoint()
            except Exception:
                logger.error(
                    "OnlineLearner: exception during gradient step — batch of %d events discarded.",
                    len(batch),
                    exc_info=True,
                )
                batch.clear()
                continue

    def _apply_gradient_step(self, batch: list[dict]) -> None:
        """
        Apply a BPR-style gradient step using the interactions in *batch*.

        Builds (user, pos_item, neg_item, weight) triples from the batch:
          - Positive events (interaction_weight > 0): user liked pos_item;
            a random item is sampled as neg_item.
          - Negative events (interaction_weight < 0): user disliked safe_item;
            roles are swapped so safe_item becomes neg_item and a random item
            becomes pos_item.

        The BPR loss for each triple is scaled by abs(interaction_weight).
        Gradients are clipped to max L2 norm of 1.0 before the optimizer step.
        """
        users: list[int] = []
        pos_items: list[int] = []
        neg_items: list[int] = []
        weights: list[float] = []

        num_users = self.lightgcn.num_users
        num_items = self.lightgcn.num_items

        for event in batch:
            interaction_weight = event.get("interaction_weight", 0.0)
            if interaction_weight == 0.0:
                continue

            user_id = event.get("user_id")
            movie_id = event.get("movie_id")

            if user_id is None or movie_id is None:
                continue

            safe_user = int(user_id) % num_users
            safe_item = int(movie_id) % num_items

            if interaction_weight > 0:
                # Positive interaction: user liked safe_item.
                # Ensure neg_item != safe_item so BPR loss is non-zero.
                neg_item = random.randrange(num_items)
                while neg_item == safe_item:
                    neg_item = random.randrange(num_items)
                users.append(safe_user)
                pos_items.append(safe_item)
                neg_items.append(neg_item)
            else:
                # Negative interaction: user disliked safe_item — swap roles.
                # Ensure random_item != safe_item so BPR loss is non-zero.
                random_item = random.randrange(num_items)
                while random_item == safe_item:
                    random_item = random.randrange(num_items)
                users.append(safe_user)
                pos_items.append(random_item)
                neg_items.append(safe_item)

            weights.append(abs(interaction_weight))

        if not users:
            return

        user_tensor = torch.tensor(users, dtype=torch.long)
        pos_tensor = torch.tensor(pos_items, dtype=torch.long)
        neg_tensor = torch.tensor(neg_items, dtype=torch.long)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)

        # Retrieve embeddings directly from the embedding tables (no graph
        # propagation needed for the incremental update — keeps it lightweight)
        user_emb = self.lightgcn.user_embedding(user_tensor)
        pos_emb = self.lightgcn.item_embedding(pos_tensor)
        neg_emb = self.lightgcn.item_embedding(neg_tensor)

        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)

        # BPR loss scaled by interaction weight
        loss = (F.softplus(neg_scores - pos_scores) * weight_tensor).mean()

        emb_params = [
            self.lightgcn.user_embedding.weight,
            self.lightgcn.item_embedding.weight,
        ]

        # Lazily create the optimizer on first call and reuse it across steps
        # so Adam's momentum state (m, v, step count) accumulates correctly.
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(emb_params, lr=self.lr)

        self._optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent embedding collapse (Requirement 3.7)
        torch.nn.utils.clip_grad_norm_(emb_params, max_norm=1.0)

        self._optimizer.step()

    def _checkpoint(self) -> None:
        """
        Persist the current LightGCN embedding weights to *checkpoint_path*.

        Logs INFO on success and ERROR on failure without re-raising so the
        background loop can continue uninterrupted (Requirement 3.9, 3.10).
        """
        try:
            torch.save(self.lightgcn.state_dict(), self.checkpoint_path)
            logger.info(
                "OnlineLearner: checkpoint saved to %s after %d events.",
                self.checkpoint_path,
                self._events_processed,
            )
        except Exception as exc:
            logger.error(
                "OnlineLearner: failed to save checkpoint to %s — %s",
                self.checkpoint_path,
                exc,
            )
