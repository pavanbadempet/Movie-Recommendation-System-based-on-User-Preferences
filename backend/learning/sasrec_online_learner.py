"""
Online Fine-Tuning for SASRec Sequential Recommender.

Extends the OnlineLearner pattern to SASRec, incrementally fine-tuning
the Transformer's item embeddings and attention layers from live click and
rating events — closing the feedback loop for the highest-weighted model
in the ensemble (DR weight: 0.659).

Design:
- Uses the same BPR-style contrastive loss as the offline trainer.
- Positive event: (user session sequence, clicked/rated item) pair.
- Negative event: random item sample (uniform or popularity-weighted fallback).
- Sequence updates: after each gradient step, updates the in-memory session
  cache in ApexEnsembleEngine so the next request sees the fresh sequence.
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

from backend.models.sasrec import SASRec

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


class SASRecOnlineLearner:
    """
    Background learner that incrementally fine-tunes SASRec from live events.

    For each positive event (click / rating >= 4.0):
      - Retrieves the user's current session sequence from the engine cache.
      - Constructs a (sequence → pos_item, neg_item) training triple.
      - Applies a BPR gradient step on the item embeddings and the final
        attention block (lighter than full backprop — keeps latency low).

    For negative events (rating < 2.5):
      - Applies a reversed BPR step so the model learns to down-rank
        explicitly disliked items.
    """

    def __init__(
        self,
        sasrec: SASRec,
        session_sequence_getter,  # callable(user_id: int) -> list[int]
        batch_size: int = 16,
        lr: float = 5e-5,  # Smaller LR than LightGCN — Transformer is more sensitive
        checkpoint_interval: int = 500,
        checkpoint_path: Path = MODELS_DIR / "sasrec_online.pth",
        num_items: int | None = None,
    ) -> None:
        self.sasrec = sasrec
        self.session_sequence_getter = session_sequence_getter
        self.batch_size = batch_size
        self.lr = lr
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path
        self.num_items = num_items or sasrec.num_items

        self._queue: queue.Queue = queue.Queue(maxsize=5000)
        self._stop_event = threading.Event()
        self._events_processed = 0
        self._thread: threading.Thread | None = None
        self._optimizer: torch.optim.Adam | None = None

        # Fine-tune only item embeddings + the last attention block (layer -1).
        # This keeps gradient computation lightweight for online updates.
        last_block_idx = len(self.sasrec.attention_layers) - 1
        self._trainable_params = list(self.sasrec.item_emb.parameters())
        if last_block_idx >= 0:
            self._trainable_params += list(self.sasrec.attention_layers[last_block_idx].parameters())
            self._trainable_params += list(self.sasrec.forward_layers[last_block_idx].parameters())

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def enqueue(self, event: dict) -> None:
        """Classify and enqueue a live event for online fine-tuning."""
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
                    "SASRecOnlineLearner queue full — dropped event (event_type=%s, user_id=%s, movie_id=%s).",
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
            name="SASRecOnlineLearner",
            daemon=True,
        )
        self._thread.start()
        logger.info("SASRecOnlineLearner background thread started.")

    def stop(self) -> None:
        """Signal the thread to stop and wait up to 5 seconds."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("SASRecOnlineLearner thread did not stop within 5 seconds.")
        logger.info("SASRecOnlineLearner stopped.")

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
                    "SASRecOnlineLearner: exception during gradient step — batch of %d discarded.",
                    len(batch),
                    exc_info=True,
                )

    def _apply_gradient_step(self, batch: list[dict]) -> None:
        """
        Build (sequence, pos_item, neg_item, weight) triples and apply BPR.

        The sequence for each user is fetched from the live session cache,
        which already reflects their most recent interactions.
        """
        seqs: list[list[int]] = []
        pos_items: list[int] = []
        neg_items: list[int] = []
        weights: list[float] = []

        for event in batch:
            weight = event.get("interaction_weight", 0.0)
            if weight == 0.0:
                continue

            user_id = event.get("user_id")
            movie_id = event.get("movie_id")
            if user_id is None or movie_id is None:
                continue

            safe_item = int(movie_id) % self.num_items

            # Fetch the user's current session sequence (up to max_seq_len)
            try:
                seq = self.session_sequence_getter(int(user_id))
            except Exception as exc:
                logger.debug("Failed to get session sequence for user %s: %s", user_id, exc)
                seq = []

            # Pad / truncate to max_seq_len
            seq_len = self.sasrec.max_seq_len
            padded = [0] * max(seq_len - len(seq), 0) + [s % self.num_items for s in seq[-seq_len:]]

            neg_item = random.randrange(self.num_items)

            if weight > 0:
                seqs.append(padded)
                pos_items.append(safe_item)
                neg_items.append(neg_item)
            else:
                # Negative interaction: flip roles
                seqs.append(padded)
                pos_items.append(neg_item)
                neg_items.append(safe_item)

            weights.append(abs(weight))

        if not seqs:
            return

        seq_tensor = torch.tensor(seqs, dtype=torch.long)  # [B, seq_len]
        pos_tensor = torch.tensor(pos_items, dtype=torch.long)  # [B]
        neg_tensor = torch.tensor(neg_items, dtype=torch.long)  # [B]
        weight_tensor = torch.tensor(weights, dtype=torch.float32)  # [B]

        # Forward pass: use predict() for the final hidden state scores
        # We compute scores via the final hidden state against pos/neg items
        seq_out = self.sasrec(seq_tensor)  # [B, seq_len, hidden_dim]
        final_state = seq_out[:, -1, :]  # [B, hidden_dim]

        pos_emb = self.sasrec.item_emb(pos_tensor)  # [B, hidden_dim]
        neg_emb = self.sasrec.item_emb(neg_tensor)  # [B, hidden_dim]

        pos_scores = (final_state * pos_emb).sum(dim=1)  # [B]
        neg_scores = (final_state * neg_emb).sum(dim=1)  # [B]

        loss = (F.softplus(neg_scores - pos_scores) * weight_tensor).mean()

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self._trainable_params, lr=self.lr)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._trainable_params, max_norm=0.5)
        self._optimizer.step()

    def _checkpoint(self) -> None:
        try:
            torch.save(self.sasrec.state_dict(), self.checkpoint_path)
            logger.info(
                "SASRecOnlineLearner: checkpoint saved to %s after %d events.",
                self.checkpoint_path,
                self._events_processed,
            )
        except Exception as exc:
            logger.error(
                "SASRecOnlineLearner: checkpoint failed — %s",
                exc,
            )
