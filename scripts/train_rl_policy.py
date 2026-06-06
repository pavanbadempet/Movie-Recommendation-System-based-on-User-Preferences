import logging
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.learning.rl_policy import ActorCriticPolicy
from backend.learning.rl_reward import RLRewardEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_offline_rl():
    """
    Trains the Actor-Critic policy using offline historical interaction logs.
    In a real environment, this would read from a massive Parquet dataset of user trajectories.
    We simulate Conservative Q-Learning (CQL) concepts by heavily penalizing actions that diverge
    from the historical behavioral policy to prevent distributional shift.
    """
    logger.info("Initializing Offline RL Training Pipeline...")

    # State = 768 (SBERT) + 3 (Dynamic)
    state_dim = 768 + 3
    # Action = 768 dimensional shift vector applied to query embedding
    action_dim = 768

    policy = ActorCriticPolicy(state_dim=state_dim, action_dim=action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    RLRewardEngine()

    # Simulate historical offline data (Batch Size: 256)
    # In production, this data comes from backend.events
    batch_size = 256
    epochs = 100

    logger.info(f"Training Actor-Critic for {epochs} epochs (Batch size: {batch_size})")

    for epoch in range(epochs):
        # 1. Sample historical transitions (s, a, r, s')
        states = torch.randn(batch_size, state_dim)
        # Historical actions (shift vectors)
        historical_actions = torch.randn(batch_size, action_dim)

        # Historical rewards (Assume some led to retention, some to churn)
        # We simulate a reward distribution with a mean of 0.5
        historical_rewards = torch.randn(batch_size, 1) + 0.5

        # 2. Forward pass
        action_mean, action_std, values = policy(states)

        # 3. Critic Loss (MSE against historical reward)
        # Value should predict the empirical reward
        critic_loss = F.mse_loss(values, historical_rewards)

        # 4. Actor Loss (Behavioral Cloning + Advantage)
        # Advantage tells us if the historical action was better than expected
        advantages = historical_rewards - values.detach()

        # Log probability of the historical action under our current policy
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(historical_actions).sum(dim=-1, keepdim=True)

        # Standard REINFORCE objective
        actor_loss = -(log_probs * advantages).mean()

        # Conservative Penalty: Penalize actions that deviate too far from the historical mean
        # to prevent out-of-distribution catastrophic failure (CQL concept)
        conservative_penalty = F.mse_loss(action_mean, torch.zeros_like(action_mean)) * 0.1

        total_loss = critic_loss + actor_loss + conservative_penalty

        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss.item():.4f} | Critic: {critic_loss.item():.4f} | Actor: {actor_loss.item():.4f}"
            )

    # Save Model
    save_path = MODELS_DIR / "rl_policy.pth"
    torch.save(policy.state_dict(), save_path)
    logger.info(f"Successfully trained and saved Actor-Critic policy to {save_path}")


if __name__ == "__main__":
    train_offline_rl()
