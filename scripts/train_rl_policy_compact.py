"""
Compact RL Policy Training Script

Trains the Actor-Critic policy with state_dim=20 and action_dim=16 to match
the live serving path in backend/recommender.py (_build_rl_state).

State vector layout (20 floats):
  [0] log1p(total_ratings) / log1p(1000)
  [1] avg_rating / 5.0
  [2] log1p(click_count) / log1p(500)
  [3] log1p(view_count) / log1p(500)
  [4..19] ALS user embedding (16d), zeros if unavailable

Action vector (16 floats):
  A shift vector applied to LightGCN item embeddings (emb_dim=16).
  Higher dot-product with an item embedding → that item gets a score boost.

Training approach:
  - Offline behavioral cloning from real Event Store interactions
  - Conservative Q-Learning (CQL) penalty to prevent out-of-distribution actions
  - Reward shaping: +1.0 for rating>=4.0, -0.5 for rating<=2.0, +0.3 for click
  - Falls back to synthetic data if fewer than 50 real interactions exist

Usage:
    python scripts/train_rl_policy_compact.py [--epochs N] [--lr LR] [--batch-size B]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rl_policy import ActorCriticPolicy
from backend.events import iter_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Must match backend/recommender.py _build_rl_state
STATE_DIM = 20
ACTION_DIM = 16  # matches LightGCN emb_dim


# ---------------------------------------------------------------------------
# State builder — mirrors _build_rl_state in recommender.py exactly
# ---------------------------------------------------------------------------

def build_state(
    total_ratings: int,
    avg_rating: float,
    click_count: int,
    view_count: int,
    als_emb: np.ndarray | None = None,
) -> np.ndarray:
    """Build a 20-float state vector matching the serving path."""
    def safe(v: float) -> float:
        return v if math.isfinite(v) else 0.0

    scalars = [
        safe(math.log1p(max(total_ratings, 0)) / math.log1p(1000)),
        safe(avg_rating / 5.0),
        safe(math.log1p(max(click_count, 0)) / math.log1p(500)),
        safe(math.log1p(max(view_count, 0)) / math.log1p(500)),
    ]

    if als_emb is not None:
        emb = np.asarray(als_emb, dtype=np.float32).flatten()[:16]
        emb_list = emb.tolist() + [0.0] * (16 - len(emb))
    else:
        emb_list = [0.0] * 16

    state = np.array(scalars + emb_list, dtype=np.float32)
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    return state


# ---------------------------------------------------------------------------
# Data loading from Event Store
# ---------------------------------------------------------------------------

def load_training_data(min_interactions: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (states, actions, rewards) from the Event Store.

    State: 20-float vector per user (aggregated from all their events)
    Action: random unit vector in 16d (proxy — we don't have ground-truth actions)
    Reward: shaped from rating/click signals

    Returns arrays of shape [N, STATE_DIM], [N, ACTION_DIM], [N, 1].
    Falls back to synthetic data if fewer than min_interactions real events exist.
    """
    logger.info("Loading interaction data from Event Store...")

    # Aggregate per-user stats
    user_stats: dict[str, dict] = defaultdict(lambda: {
        "total_ratings": 0,
        "rating_sum": 0.0,
        "click_count": 0,
        "view_count": 0,
        "rewards": [],
    })

    total_events = 0
    for event in iter_events():
        et = str(event.get("event_type", "")).lower()
        uid = str(event.get("user_id") or "anonymous")
        stats = user_stats[uid]

        if et == "rating":
            rating = event.get("rating")
            if rating is not None:
                try:
                    r = float(rating)
                    stats["total_ratings"] += 1
                    stats["rating_sum"] += r
                    reward = 1.0 if r >= 4.0 else (-0.5 if r <= 2.0 else 0.0)
                    if reward != 0.0:
                        stats["rewards"].append(reward)
                    total_events += 1
                except (TypeError, ValueError):
                    pass
        elif et == "click":
            stats["click_count"] += 1
            stats["rewards"].append(0.3)
            total_events += 1
        elif et == "view":
            stats["view_count"] += 1
            total_events += 1

    logger.info("Found %d qualifying events across %d users.", total_events, len(user_stats))

    if total_events < min_interactions:
        logger.warning(
            "Only %d real interactions found (minimum %d). "
            "Falling back to synthetic training data.",
            total_events,
            min_interactions,
        )
        return _synthetic_data(batch_size=512)

    # Build training samples — one per reward signal
    states_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    rewards_list: list[float] = []

    rng = np.random.default_rng(seed=42)

    for uid, stats in user_stats.items():
        if not stats["rewards"]:
            continue

        avg_rating = (
            stats["rating_sum"] / stats["total_ratings"]
            if stats["total_ratings"] > 0
            else 3.0
        )
        state = build_state(
            total_ratings=stats["total_ratings"],
            avg_rating=avg_rating,
            click_count=stats["click_count"],
            view_count=stats["view_count"],
        )

        for reward in stats["rewards"]:
            states_list.append(state)
            # Action proxy: unit vector in direction of reward signal
            action = rng.standard_normal(ACTION_DIM).astype(np.float32)
            action /= np.linalg.norm(action) + 1e-8
            if reward < 0:
                action = -action  # negative reward → push away
            actions_list.append(action)
            rewards_list.append(reward)

    if len(states_list) < min_interactions:
        logger.warning(
            "Only %d reward-bearing samples built. Augmenting with synthetic data.",
            len(states_list),
        )
        s_syn, a_syn, r_syn = _synthetic_data(batch_size=max(256, min_interactions))
        states_list.extend(s_syn.tolist())
        actions_list.extend(a_syn.tolist())
        rewards_list.extend(r_syn.squeeze().tolist())

    states = np.array(states_list, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.float32)
    rewards = np.array(rewards_list, dtype=np.float32).reshape(-1, 1)

    logger.info(
        "Training dataset: %d samples, state_dim=%d, action_dim=%d",
        len(states),
        STATE_DIM,
        ACTION_DIM,
    )
    return states, actions, rewards


def _synthetic_data(batch_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic training data when real events are insufficient."""
    rng = np.random.default_rng(seed=0)
    states = rng.standard_normal((batch_size, STATE_DIM)).astype(np.float32)
    # Clip to plausible range [0, 1] for the scalar features
    states[:, :4] = np.clip(states[:, :4] * 0.3 + 0.5, 0.0, 1.0)
    actions = rng.standard_normal((batch_size, ACTION_DIM)).astype(np.float32)
    norms = np.linalg.norm(actions, axis=1, keepdims=True) + 1e-8
    actions /= norms
    rewards = (rng.standard_normal((batch_size, 1)) * 0.3 + 0.5).astype(np.float32)
    return states, actions, rewards


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int = 200,
    lr: float = 1e-4,
    batch_size: int = 256,
) -> None:
    """Train the compact Actor-Critic policy and save to models/rl_policy.pth."""

    logger.info("=" * 60)
    logger.info("Compact RL Policy Training (state_dim=%d, action_dim=%d)", STATE_DIM, ACTION_DIM)
    logger.info("=" * 60)

    # Load data
    states_np, actions_np, rewards_np = load_training_data()
    n_samples = len(states_np)

    states_t = torch.tensor(states_np, dtype=torch.float32)
    actions_t = torch.tensor(actions_np, dtype=torch.float32)
    rewards_t = torch.tensor(rewards_np, dtype=torch.float32)

    # Normalise rewards to [-1, 1] for stable training
    r_mean = rewards_t.mean()
    r_std = rewards_t.std() + 1e-8
    rewards_norm = (rewards_t - r_mean) / r_std

    # Initialise policy
    policy = ActorCriticPolicy(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    rng_idx = np.random.default_rng(seed=1)

    logger.info("Training for %d epochs, batch_size=%d, lr=%.2e", epochs, batch_size, lr)

    for epoch in range(epochs):
        # Sample a random mini-batch
        idx = rng_idx.integers(0, n_samples, size=min(batch_size, n_samples))
        s = states_t[idx]
        a_hist = actions_t[idx]
        r = rewards_norm[idx]

        # Forward pass
        action_mean, action_std, values = policy(s)

        # Critic loss: predict normalised reward
        critic_loss = F.mse_loss(values, r)

        # Advantage
        advantages = (r - values.detach())

        # Actor loss: behavioural cloning weighted by advantage
        dist = torch.distributions.Normal(action_mean, action_std.clamp(min=1e-4))
        log_probs = dist.log_prob(a_hist).sum(dim=-1, keepdim=True)
        actor_loss = -(log_probs * advantages).mean()

        # CQL conservative penalty: keep actions close to zero mean
        # (prevents out-of-distribution score explosions in serving)
        conservative_penalty = F.mse_loss(action_mean, torch.zeros_like(action_mean)) * 0.05

        # Entropy bonus: encourage exploration
        entropy_bonus = -dist.entropy().mean() * 0.01

        total_loss = critic_loss + actor_loss + conservative_penalty + entropy_bonus

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            logger.info(
                "Epoch %3d/%d | Loss: %.4f | Critic: %.4f | Actor: %.4f | CQL: %.4f",
                epoch + 1,
                epochs,
                total_loss.item(),
                critic_loss.item(),
                actor_loss.item(),
                conservative_penalty.item(),
            )

    # Save
    save_path = MODELS_DIR / "rl_policy.pth"
    torch.save(policy.state_dict(), save_path)
    logger.info("=" * 60)
    logger.info("Saved compact RL policy to %s", save_path)
    logger.info("state_dim=%d, action_dim=%d — compatible with serving path", STATE_DIM, ACTION_DIM)
    logger.info("=" * 60)

    # Quick sanity check: run one inference
    policy.eval()
    with torch.no_grad():
        test_state = torch.zeros(1, STATE_DIM)
        action, value = policy.get_action(test_state, deterministic=True)
        logger.info(
            "Sanity check — action shape: %s, value: %.4f",
            tuple(action.shape),
            value.item(),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train compact RL policy (state_dim=20, action_dim=16) for APEX serving path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
