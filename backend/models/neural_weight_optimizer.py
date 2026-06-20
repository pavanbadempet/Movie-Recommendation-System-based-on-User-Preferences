"""
Neural Architecture Search for Ensemble Weight Optimization.

Instead of Dirichlet grid search (random sampling), this uses gradient-based
meta-learning to find optimal ensemble weights. The weights themselves become
learnable parameters optimized via backpropagation on the validation set.

This is equivalent to learning a "meta-model" that decides how much to trust
each base model for each user context — the same approach used in stacking
ensembles at Netflix and Google.

Key innovation: The weights are CONTEXT-DEPENDENT — different users get
different ensemble weights based on their behavior profile. A user with
long watch history gets more weight on SASRec; a cold-start user gets
more weight on content-based signals.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WEIGHT_KEYS = ("lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion", "clifford")


class ContextualWeightNetwork(nn.Module):
    """
    A small neural network that outputs context-dependent ensemble weights.

    Input: User context vector (20d from _build_rl_state)
    Output: ensemble weights (softmax-normalized)

    This allows the system to automatically learn:
    - Cold-start users → more weight on content-based models
    - Power users → more weight on sequential models (SASRec)
    - Genre-specific users → more weight on KG-based models
    """

    def __init__(self, context_dim: int = 20, n_models: int = len(WEIGHT_KEYS), hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_models),
        )
        # Initialize to uniform weights
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Returns softmax-normalized weights for each ensemble model."""
        logits = self.net(context)
        return F.softmax(logits, dim=-1)


def get_contextual_weights(
    behavior_profile: dict,
    als_user_embedding: np.ndarray | None = None,
    model_path: Path = MODELS_DIR / "contextual_weight_net.pth",
) -> dict[str, float]:
    """
    Get context-dependent ensemble weights for a user.

    Falls back to static weights from ensemble_weights.json if the
    contextual network is not available.

    Args:
        behavior_profile: User behavior profile dict
        als_user_embedding: Optional ALS user embedding
        model_path: Path to trained ContextualWeightNetwork

    Returns:
        Dict mapping model name → weight
    """
    try:
        if not model_path.exists():
            return _load_static_weights()

        net = ContextualWeightNetwork()
        net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        net.eval()

        # Build context vector (same as _build_rl_state)
        def safe(v, default=0.0):
            try:
                x = float(v)
                return x if math.isfinite(x) else default
            except (TypeError, ValueError):
                return default

        scalars = [
            safe(math.log1p(max(safe(behavior_profile.get("total_ratings", 0)), 0)) / math.log1p(1000)),
            safe(behavior_profile.get("avg_rating", 0)) / 5.0,
            safe(math.log1p(max(safe(behavior_profile.get("click_count", 0)), 0)) / math.log1p(500)),
            safe(math.log1p(max(safe(behavior_profile.get("view_count", 0)), 0)) / math.log1p(500)),
        ]

        if als_user_embedding is not None:
            emb = np.asarray(als_user_embedding, dtype=np.float32).flatten()[:16]
            emb_list = emb.tolist() + [0.0] * (16 - len(emb))
        else:
            emb_list = [0.0] * 16

        context = torch.tensor(scalars + emb_list, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            weights = net(context).squeeze().tolist()

        return {key: float(weights[i]) for i, key in enumerate(WEIGHT_KEYS)}

    except Exception as exc:
        logger.debug("Contextual weights unavailable (%s); using static weights", exc)
        return _load_static_weights()


def _load_static_weights() -> dict[str, float]:
    """Load static weights from ensemble_weights.json."""
    import json

    weights_path = MODELS_DIR / "ensemble_weights.json"
    defaults = {k: 1.0 / len(WEIGHT_KEYS) for k in WEIGHT_KEYS}
    if not weights_path.exists():
        return defaults
    try:
        with open(weights_path, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: float(raw.get(k, defaults[k])) for k in WEIGHT_KEYS}
    except Exception:
        return defaults


def train_contextual_weight_network(
    epochs: int = 100,
    lr: float = 1e-3,
) -> None:
    """
    Train the ContextualWeightNetwork on validation data.

    Uses gradient-based meta-learning to optimize contextual weights.
    For each user, we feed their context (scalars + base embedding) into
    the network, obtain the model weights, compute the blended scores,
    and update the network to minimize binary cross-entropy loss against
    their validation ground-truth items.
    """
    from backend.events import iter_events
    from backend.models.ensemble_engine import ApexEnsembleEngine
    from scripts.optimize_ensemble_weights import (
        _build_validation_split,
        _load_interaction_data,
        _precompute_per_model_scores,
    )
    import random
    from collections import defaultdict

    logger.info("Training ContextualWeightNetwork via gradient-based meta-learning...")

    # Load interaction data and build validation splits
    user_events = _load_interaction_data()
    train_history, val_ground_truth = _build_validation_split(user_events)
    valid_users = {uid for uid, gt in val_ground_truth.items() if gt}

    if len(valid_users) < 10:
        logger.warning("Too few validation users (%d) for contextual weight network training", len(valid_users))
        return

    val_ground_truth = {uid: val_ground_truth[uid] for uid in valid_users}
    train_history = {uid: train_history[uid] for uid in valid_users if uid in train_history}

    # Load ensemble engine with matching size
    try:
        from backend.pipeline.recommender import get_recommender

        rec = get_recommender()
        num_items = len(rec.movies) if rec._movies is not None else 50000
    except Exception:
        num_items = 50000
    engine = ApexEnsembleEngine(num_users=1000, num_items=num_items)
    rng = random.Random(42)

    # Pre-compute scores ONCE for validation users
    per_model_scores = _precompute_per_model_scores(engine, train_history, val_ground_truth, rng)

    if not per_model_scores:
        logger.warning("No precomputed scores available for training")
        return

    # Load raw events to build detailed user behavior profiles
    all_events = list(iter_events())
    user_raw_events = defaultdict(list)
    for event in all_events:
        uid = event.get("user_id")
        if uid:
            user_raw_events[str(uid)].append(event)

    # Prepare datasets
    X_list = []
    scores_matrices = []
    labels_list = []

    for user_id, item_scores in per_model_scores.items():
        gt = val_ground_truth.get(user_id, set())
        if not gt or not item_scores:
            continue

        # 1. Build Behavior Profile
        events = user_raw_events.get(str(user_id), [])
        total_ratings = sum(1 for e in events if str(e.get("event_type")).lower() == "rating")
        ratings_list = [float(e["rating"]) for e in events if str(e.get("event_type")).lower() == "rating" and e.get("rating") is not None]
        avg_rating = sum(ratings_list) / len(ratings_list) if ratings_list else 3.5
        click_count = sum(1 for e in events if str(e.get("event_type")).lower() == "click")
        view_count = sum(1 for e in events if str(e.get("event_type")).lower() == "view")

        scalars = [
            math.log1p(max(total_ratings, 0)) / math.log1p(1000),
            avg_rating / 5.0,
            math.log1p(max(click_count, 0)) / math.log1p(500),
            math.log1p(max(view_count, 0)) / math.log1p(500),
        ]

        # 2. Get User Base Embedding
        try:
            uid_int = int(user_id)
        except (ValueError, TypeError):
            uid_int = abs(hash(user_id))
        safe_uid = uid_int % engine.num_users

        u_tensor = torch.tensor([safe_uid], dtype=torch.long)
        with torch.no_grad():
            base_u_emb = engine.lightgcn.user_embedding(u_tensor).squeeze().cpu().numpy()

        emb = np.asarray(base_u_emb, dtype=np.float32).flatten()[:16]
        emb_list = emb.tolist() + [0.0] * (16 - len(emb))
        context_vector = scalars + emb_list

        X_list.append(context_vector)

        # 3. Construct Scores Matrix & Labels
        candidate_ids = list(item_scores.keys())
        # [N_candidates, 6]
        scores_mat = np.array([item_scores[iid] for iid in candidate_ids], dtype=np.float32)
        scores_matrices.append(torch.tensor(scores_mat, dtype=torch.float32))

        labels = np.array([1.0 if iid in gt else 0.0 for iid in candidate_ids], dtype=np.float32)
        labels_list.append(torch.tensor(labels, dtype=torch.float32))

    if len(X_list) == 0:
        logger.warning("No valid user training samples constructed")
        return

    X = torch.tensor(X_list, dtype=torch.float32)

    net = ContextualWeightNetwork()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    logger.info("Starting training loop over %d validation users...", len(X_list))

    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()

        # Predict weights for all users: shape [N_users, 6]
        pred_weights = net(X)

        epoch_loss = 0.0
        for i in range(len(X_list)):
            w_u = pred_weights[i]  # [6]
            scores_u = scores_matrices[i]  # [N_candidates, 6]
            labels_u = labels_list[i]  # [N_candidates]

            # blended score: [N_candidates]
            blended = torch.matmul(scores_u, w_u)
            # Clip to prevent log(0)
            blended_clipped = torch.clamp(blended, 1e-6, 1.0 - 1e-6)
            loss_u = F.binary_cross_entropy(blended_clipped, labels_u)
            epoch_loss += loss_u

        epoch_loss = epoch_loss / len(X_list)
        epoch_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info("  ContextualWeightNet Epoch %d/%d | Loss: %.6f", epoch + 1, epochs, epoch_loss.item())

    save_path = MODELS_DIR / "contextual_weight_net.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_path)
    logger.info("ContextualWeightNetwork saved successfully to %s", save_path)


if __name__ == "__main__":
    train_contextual_weight_network(epochs=100)
