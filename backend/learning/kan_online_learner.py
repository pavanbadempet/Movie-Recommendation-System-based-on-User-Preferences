"""
Online Gradient Updates for the KAN Ranker.

Incrementally fine-tunes the KAN ranker's edge-function coefficients
(Fourier sin/cos parameters) from live interaction events — closing the
feedback loop for the second-highest weighted model in the ensemble
(DR weight: 0.298).

Design:
- KAN operates on user + item embedding pairs, not raw IDs.
- User/item embeddings are sourced from LightGCN (shared representation).
- A BPR-style ranking loss drives the Fourier coefficient updates:
    KAN(u, pos_item) > KAN(u, neg_item)
- Only KAN's own parameters are updated; LightGCN embeddings remain
  under the control of OnlineLearner to prevent conflicting gradient signals.
- Thread-safe: daemon thread + bounded queue, identical lifecycle to OnlineLearner.
"""

from __future__ import annotations

import logging
from pathlib import Path
import queue
import random
import threading

import torch
import torch.nn.functional as F

from backend.models.kan_ranker import KANRanker
from backend.models.lightgcn import LightGCN

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


class KANOnlineLearner:
    """
    Background learner that incrementally fine-tunes the KAN ranker
    using live user interaction events.

    For each positive event, the KAN is trained to score the interacted item
    higher than a randomly sampled negative. This updates the learnable edge
    functions (Fourier coefficients) without touching LightGCN's embeddings,
    which are managed by OnlineLearner.
    """

    def __init__(
        self,
        kan: KANRanker,
        lightgcn: LightGCN,
        batch_size: int = 32,
        lr: float = 1e-4,
        checkpoint_interval: int = 750,
        checkpoint_path: Path = MODELS_DIR / "kan_online.pth",
    ) -> None:
        self.kan = kan
        self.lightgcn = lightgcn
        self.batch_size = batch_size
        self.lr = lr
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path

        self._queue: queue.Queue = queue.Queue(maxsize=5000)
        self._stop_event = threading.Event()
        self._events_processed = 0
        self._thread: threading.Thread | None = None
        # Only KAN's own Fourier coefficients and base weights are updated.
        # LightGCN embeddings are excluded — they're handled by OnlineLearner.
        self._optimizer: torch.optim.Adam | None = None

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def enqueue(self, event: dict) -> None:
        """Classify and enqueue a live event."""
        event_type = event.get("event_type")

        if event_type == "rating":
            rating = event.get("rating", 0.0)
            if rating >= 4.0:
                weight = 1.0
            elif rating < 2.5:
                weight = -0.5
            else:
                return
        elif event_type == "click":
            weight = 0.3
        else:
            return

        enriched = dict(event)
        enriched["interaction_weight"] = weight

        try:
            self._queue.put_nowait(enriched)
        except queue.Full:
            try:
                dropped = self._queue.get_nowait()
                logger.warning(
                    "KANOnlineLearner queue full — dropped event "
                    "(event_type=%s, user_id=%s, movie_id=%s).",
                    dropped.get("event_type"),
                    dropped.get("user_id"),
                    dropped.get("movie_id"),
                )
            except queue.Empty:
                pass
            self._queue.put_nowait(enriched)

    def start(self) -> None:
        """Start the background daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="KANOnlineLearner",
            daemon=True,
        )
        self._thread.start()
        logger.info("KANOnlineLearner background thread started.")

    def stop(self) -> None:
        """Signal the thread to stop and wait up to 5 seconds."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("KANOnlineLearner thread did not stop within 5 seconds.")
        logger.info("KANOnlineLearner stopped.")

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            batch: list[dict] = []
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
                    "KANOnlineLearner: exception during gradient step — batch of %d discarded.",
                    len(batch),
                    exc_info=True,
                )

    def _apply_gradient_step(self, batch: list[dict]) -> None:
        """
        Compute BPR loss on (user_emb, pos_item_emb, neg_item_emb) triples.

        Embeddings are sourced from LightGCN's current weights (no_grad) so
        KAN benefits from the freshest representations without coupling the
        two gradient flows.
        """
        users: list[int] = []
        pos_items: list[int] = []
        neg_items: list[int] = []
        weights: list[float] = []

        num_users = self.lightgcn.num_users
        num_items = self.lightgcn.num_items

        for event in batch:
            weight = event.get("interaction_weight", 0.0)
            if weight == 0.0:
                continue

            user_id = event.get("user_id")
            movie_id = event.get("movie_id")
            if user_id is None or movie_id is None:
                continue

            safe_user = int(user_id) % num_users
            safe_item = int(movie_id) % num_items
            neg_item = random.randrange(num_items)

            if weight > 0:
                users.append(safe_user)
                pos_items.append(safe_item)
                neg_items.append(neg_item)
            else:
                users.append(safe_user)
                pos_items.append(neg_item)
                neg_items.append(safe_item)

            weights.append(abs(weight))

        if not users:
            return

        u_tensor = torch.tensor(users, dtype=torch.long)
        p_tensor = torch.tensor(pos_items, dtype=torch.long)
        n_tensor = torch.tensor(neg_items, dtype=torch.long)
        w_tensor = torch.tensor(weights, dtype=torch.float32)

        # Embeddings from LightGCN — detached so no gradient flows back into LightGCN
        with torch.no_grad():
            u_emb = self.lightgcn.user_embedding(u_tensor).detach()   # [B, emb_dim]
            p_emb = self.lightgcn.item_embedding(p_tensor).detach()   # [B, emb_dim]
            n_emb = self.lightgcn.item_embedding(n_tensor).detach()   # [B, emb_dim]

        # KAN forward passes
        pos_scores = self.kan(u_emb, p_emb)   # [B]
        neg_scores = self.kan(u_emb, n_emb)   # [B]

        loss = (F.softplus(neg_scores - pos_scores) * w_tensor).mean()

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.kan.parameters(), lr=self.lr)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.kan.parameters(), max_norm=1.0)
        self._optimizer.step()

    def _checkpoint(self) -> None:
        try:
            torch.save(self.kan.state_dict(), self.checkpoint_path)
            logger.info(
                "KANOnlineLearner: checkpoint saved to %s after %d events.",
                self.checkpoint_path,
                self._events_processed,
            )
        except Exception as exc:
            logger.error("KANOnlineLearner: checkpoint failed — %s", exc)
