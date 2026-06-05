"""
Generative Diffusion Recommender - Training Experiment Harness.

This script trains the ConditionedDenoiser to predict added Gaussian noise,
using the user's historical embedding as the guiding condition.

If successful, the model learns to map a user's viewing history directly
to the mathematical embedding of their perfect next movie.
"""

import logging
from pathlib import Path
import sys

import numpy as np
import torch

# Add root directory to python path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader, Dataset

from backend.diffusion_recommender import LatentDiffusionRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
EVENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "events"
CHECKPOINT_PATH = MODELS_DIR / "diffusion_recommender.pth"


class DiffusionDataset(Dataset):
    """
    Constructs training pairs: (User History Embedding, Target Movie Embedding)
    """

    def __init__(self, item_embeddings, user_histories):
        self.item_embeddings = item_embeddings  # numpy array [num_items, emb_dim]
        self.user_histories = user_histories  # list of dicts: {"history": [idx1, idx2], "target": target_idx}

    def __len__(self):
        return len(self.user_histories)

    def __getitem__(self, idx):
        record = self.user_histories[idx]
        hist_indices = record["history"]
        target_idx = record["target"]

        # User embedding is the mean of their historical movie embeddings
        if len(hist_indices) > 0:
            user_emb = np.mean(self.item_embeddings[hist_indices], axis=0)
        else:
            user_emb = np.zeros_like(self.item_embeddings[0])

        target_emb = self.item_embeddings[target_idx]

        return torch.FloatTensor(user_emb), torch.FloatTensor(target_emb)


def build_synthetic_experiment_data(num_items=1000, emb_dim=384, num_users=5000):
    """
    Creates a synthetic dataset to prove the architecture works before scaling
    to the massive Kaggle datasets.
    """
    logger.info("Generating experimental semantic embeddings and user histories...")
    # Mocking Semantic Embeddings (e.g. SBERT 384-dimensional vectors)
    item_embeddings = np.random.randn(num_items, emb_dim).astype(np.float32)

    # Normalize to unit sphere (typical for semantic embeddings)
    norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    item_embeddings = item_embeddings / norms

    user_histories = []
    for _ in range(num_users):
        # Users watch between 3 and 15 movies
        hist_len = np.random.randint(3, 16)

        # Simulate a "genre cluster" by picking a random center and finding nearby items
        center = np.random.randn(emb_dim)
        distances = np.linalg.norm(item_embeddings - center, axis=1)
        cluster_indices = np.argsort(distances)[:50]

        # Pick history and target from this cluster
        watched = np.random.choice(cluster_indices, hist_len + 1, replace=False)
        history = watched[:-1].tolist()
        target = watched[-1]

        user_histories.append({"history": history, "target": target})

    return item_embeddings, user_histories


def train_diffusion_model():
    """Main training loop for the Denoising Diffusion Probabilistic Model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing Experiment on device: {device}")

    emb_dim = 384  # SBERT default dimension
    batch_size = 128
    epochs = 20

    # 1. Prepare Data
    item_embeddings, user_histories = build_synthetic_experiment_data(emb_dim=emb_dim)
    dataset = DiffusionDataset(item_embeddings, user_histories)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 2. Initialize Model
    model = LatentDiffusionRecommender(emb_dim=emb_dim, num_timesteps=100).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training Loop
    logger.info(f"Starting Training for {epochs} Epochs...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _batch_idx, (user_emb, target_emb) in enumerate(dataloader):
            user_emb = user_emb.to(device)
            target_emb = target_emb.to(device)

            optimizer.zero_grad()

            # The forward pass automatically corrupts target_emb with noise
            # and computes the MSE loss of the Denoiser predicting that noise.
            loss = model(true_item_emb=target_emb, user_emb=user_emb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1:02d}/{epochs} | Diffusion MSE Loss: {avg_loss:.6f}")

    # 4. Save Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    logger.info(f"Experiment successful. Model weights saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train_diffusion_model()
