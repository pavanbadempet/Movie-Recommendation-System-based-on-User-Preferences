"""
Adaptive Online Router Trainer — Self-Improving Mixture of Experts.

Continuously trains the ContextualRouter using real prediction feedback:
- Maintains a circular replay buffer of (user_state, per_model_scores) tuples
- After each ensemble prediction, records user state and model scores
- When buffer reaches min_train_size, performs mini-batch SGD on the router
- Uses Teacher-Student Loss Alignment: per-model quality → target routing probs
- Applies Inverse Propensity Scoring (IPS) to prevent feedback loops
- Periodically checkpoints router weights for crash recovery

This closes the loop between inference and training — the router gets smarter
with every prediction served, adapting to user distribution shifts in real time.
"""

from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
import threading
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from backend.models.contextual_router import ContextualRouter

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


class AdaptiveRouterTrainer:
    """
    Online trainer for the ContextualRouter using a circular replay buffer.

    Records prediction feedback and asynchronously trains the router to align
    routing probabilities with observed per-model prediction quality.

    Thread-safety: All buffer operations and training steps are guarded by
    a reentrant lock. Training is designed to run async on the module-level
    thread pool without blocking inference.
    """

    def __init__(
        self,
        router: ContextualRouter,
        buffer_capacity: int = 10_000,
        min_train_size: int = 256,
        batch_size: int = 64,
        lr: float = 1e-3,
        checkpoint_interval: int = 500,
        ips_clip: float = 5.0,
    ):
        """
        Args:
            router: The ContextualRouter instance to train.
            buffer_capacity: Maximum replay buffer size (circular eviction).
            min_train_size: Minimum samples before training begins.
            batch_size: Mini-batch size for SGD steps.
            lr: Learning rate for the router optimizer.
            checkpoint_interval: Save router weights every N training steps.
            ips_clip: Maximum IPS weight to prevent variance explosion.
        """
        self.router = router
        self.buffer_capacity = buffer_capacity
        self.min_train_size = min_train_size
        self.batch_size = batch_size
        self.ips_clip = ips_clip
        self.checkpoint_interval = checkpoint_interval

        # Circular replay buffer: deque with maxlen handles eviction automatically
        self._buffer: deque[tuple[torch.Tensor, torch.Tensor, list[str] | None]] = deque(maxlen=buffer_capacity)
        self._lock = threading.Lock()

        # Optimizer (lazy-initialized to avoid issues if router moves device)
        self._optimizer: torch.optim.Optimizer | None = None
        self._lr = lr

        # Statistics
        self._train_steps = 0
        self._total_samples_recorded = 0
        self._last_checkpoint_step = 0
        self._last_train_loss: float = 0.0
        self._cumulative_loss: float = 0.0

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        """Lazy-initialize optimizer on first training step."""
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.router.parameters(), lr=self._lr)
        return self._optimizer

    def record(
        self,
        user_state: torch.Tensor,
        model_scores: dict[str, float],
        selected_models: list[str] | None = None,
    ) -> None:
        """
        Record a prediction result into the replay buffer.

        Args:
            user_state: The user state vector [emb_dim + 4] used for routing.
            model_scores: Dict mapping model name → normalized score for this prediction.
            selected_models: Which models were actually selected by the router (for IPS).
        """
        # Convert model scores to ordered tensor matching router.model_names
        model_names = self.router.model_names
        score_tensor = torch.tensor(
            [model_scores.get(name, 0.5) for name in model_names],
            dtype=torch.float32,
        )

        with self._lock:
            self._buffer.append((user_state.detach().cpu(), score_tensor, selected_models))
            self._total_samples_recorded += 1

    @property
    def buffer_size(self) -> int:
        """Current number of samples in the replay buffer."""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """Whether enough samples have been collected to begin training."""
        return len(self._buffer) >= self.min_train_size

    def train_step(self) -> float | None:
        """
        Perform a single mini-batch training step on the router.

        Returns the training loss, or None if not enough samples.
        """
        if not self.is_ready:
            return None

        with self._lock:
            # Sample a random mini-batch from the buffer
            buffer_list = list(self._buffer)

        indices = np.random.choice(len(buffer_list), size=min(self.batch_size, len(buffer_list)), replace=False)
        batch = [buffer_list[i] for i in indices]

        user_states = torch.stack([b[0] for b in batch])
        model_scores = torch.stack([b[1] for b in batch])
        selected_models_list = [b[2] for b in batch]

        # Move to router's device
        device = next(self.router.parameters()).device
        user_states = user_states.to(device)
        model_scores = model_scores.to(device)

        # --- Teacher-Student Loss Alignment ---
        # Convert scores to target routing probabilities
        # Higher score → higher probability (the model that scored best should be routed to)
        # Using temperature scaling for sharper distributions
        temperature = 0.5
        targets = F.softmax(model_scores / temperature, dim=-1)

        # --- Inverse Propensity Scoring (IPS) Debiasing ---
        # Models that are selected more often get their training signal down-weighted
        # to prevent rich-get-richer feedback loops
        ips_weights = self._compute_ips_weights(selected_models_list, device)

        # --- Forward pass and loss ---
        optimizer = self._ensure_optimizer()
        self.router.train()
        optimizer.zero_grad()

        logits = self.router(user_states)  # [batch, num_models]
        router_log_probs = F.log_softmax(logits, dim=-1)

        # KL divergence with IPS weighting per sample
        # kl_div expects log(q) and p, computes sum(p * (log(p) - log(q)))
        per_sample_kl = F.kl_div(router_log_probs, targets, reduction="none").sum(dim=-1)
        weighted_loss = (per_sample_kl * ips_weights).mean()

        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.router.parameters(), max_norm=1.0)
        optimizer.step()

        self.router.eval()

        loss_val = weighted_loss.item()
        self._train_steps += 1
        self._last_train_loss = loss_val
        self._cumulative_loss += loss_val

        # Periodic checkpoint
        if (self._train_steps - self._last_checkpoint_step) >= self.checkpoint_interval:
            self._save_checkpoint()

        return loss_val

    def _compute_ips_weights(
        self,
        selected_models_list: list[list[str] | None],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute Inverse Propensity Score weights for each sample.

        Samples where models were selected by the router get down-weighted
        proportionally to how often that model is selected (prevents
        the router from becoming self-reinforcing).
        """
        model_names = self.router.model_names
        batch_size = len(selected_models_list)

        # Count global selection frequencies across the buffer
        with self._lock:
            selection_counts = dict.fromkeys(model_names, 0)
            total_selections = 0
            for _, _, selected in self._buffer:
                if selected is not None:
                    for m in selected:
                        selection_counts[m] = selection_counts.get(m, 0) + 1
                        total_selections += 1

        if total_selections == 0:
            return torch.ones(batch_size, device=device)

        # Per-sample IPS weight: 1 / P(selection)
        ips = torch.ones(batch_size, device=device)
        for i, selected in enumerate(selected_models_list):
            if selected is not None and len(selected) > 0:
                # Average propensity of selected models
                propensities = []
                for m in selected:
                    freq = selection_counts.get(m, 1) / max(total_selections, 1)
                    propensities.append(max(freq, 0.01))  # Floor to prevent div-by-zero
                avg_propensity = sum(propensities) / len(propensities)
                ips[i] = min(1.0 / avg_propensity, self.ips_clip)

        # Normalize IPS weights to have mean 1.0 (variance reduction)
        ips = ips / (ips.mean() + 1e-8)

        return ips

    def _save_checkpoint(self) -> None:
        """Save router weights to disk."""
        try:
            save_path = MODELS_DIR / "contextual_router.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.router.state_dict(), save_path)
            self._last_checkpoint_step = self._train_steps
            logger.info(
                "Router checkpoint saved at step %d (loss=%.4f)",
                self._train_steps,
                self._last_train_loss,
            )
        except Exception as exc:
            logger.warning("Failed to save router checkpoint: %s", exc)

    def force_checkpoint(self) -> None:
        """Force an immediate checkpoint save."""
        self._save_checkpoint()

    def get_stats(self) -> dict:
        """Return training statistics."""
        return {
            "buffer_size": len(self._buffer),
            "buffer_capacity": self.buffer_capacity,
            "total_samples_recorded": self._total_samples_recorded,
            "train_steps": self._train_steps,
            "last_train_loss": round(self._last_train_loss, 6),
            "avg_train_loss": round(self._cumulative_loss / max(self._train_steps, 1), 6),
            "is_ready": self.is_ready,
            "checkpoint_interval": self.checkpoint_interval,
            "last_checkpoint_step": self._last_checkpoint_step,
        }
