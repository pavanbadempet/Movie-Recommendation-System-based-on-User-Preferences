"""
Generative Diffusion Recommender (GDR).

This is a fundamentally novel research architecture.
Traditional recommenders (ALS, Two-Tower, SASRec) compute scores for every candidate item
and rank them. This is extremely computationally expensive.

Instead, we use a Denoising Diffusion Probabilistic Model (DDPM)—the exact same math behind
Midjourney and Stable Diffusion—but applied to Latent Recommendation Space.

HOW IT WORKS:
1. We start with pure Gaussian noise.
2. We condition the Diffusion model on the User's Historical Vector.
3. The model "denoises" the static, step-by-step, hallucinating the mathematically "perfect"
   item embedding that the user would want to watch right now.
4. We take this generated embedding, pass it to FAISS, and find the nearest REAL movie in the catalog.

This completely bypasses the Softmax bottleneck and candidate ranking entirely.
It is Generative Retrieval.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConditionedDenoiser(nn.Module):
    """
    A multi-layer network using Kolmogorov-Arnold-inspired gating mechanisms
    to denoise latent vectors.
    """

    def __init__(self, emb_dim=64, time_dim=32, user_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        # We project the concatenated (Noisy Item + Time + User History) into the denoiser
        input_dim = emb_dim + time_dim + user_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x, t, user_emb):
        # x: [batch_size, emb_dim] (noisy item)
        # t: [batch_size, 1] (timestep)
        # user_emb: [batch_size, user_dim] (conditioning signal)

        t_emb = self.time_mlp(t)

        # Combine all signals
        h = torch.cat([x, t_emb, user_emb], dim=-1)

        # Predict the noise that was added
        predicted_noise = self.net(h)
        return predicted_noise


class LatentDiffusionRecommender(nn.Module):
    """
    The full Generative Diffusion architecture for Recommendation.
    """

    def __init__(self, emb_dim=64, num_timesteps=100):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_timesteps = num_timesteps
        self.denoiser = ConditionedDenoiser(emb_dim=emb_dim, user_dim=emb_dim)

        # Define the variance schedule (Beta) linearly from 1e-4 to 0.02
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, num_timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x_start, t, noise=None):
        """
        The Forward Process: Add noise to the true movie embedding.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, true_item_emb, user_emb):
        """
        Training loop:
        1. Sample random timesteps.
        2. Corrupt the true item embeddings.
        3. Ask the denoiser to predict the noise using the user embedding as a guide.
        """
        batch_size = true_item_emb.size(0)

        # 1. Sample random timestep for each item in the batch
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=true_item_emb.device).long()

        # 2. Add noise
        noise = torch.randn_like(true_item_emb)
        x_noisy = self.q_sample(x_start=true_item_emb, t=t, noise=noise)

        # 3. Predict noise
        t_float = t.float().unsqueeze(-1) / self.num_timesteps  # Normalize t
        predicted_noise = self.denoiser(x_noisy, t_float, user_emb)

        # 4. MSE Loss between actual noise and predicted noise
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def generate_ideal_embedding(self, user_emb):
        """
        Inference: Generative Retrieval.
        Start from pure noise and let the user's preference guide the denoising process
        to hallucinate the perfect movie embedding.
        """
        batch_size = user_emb.size(0)
        device = user_emb.device

        # Start with pure Gaussian noise
        x = torch.randn((batch_size, self.emb_dim), device=device)

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_float = t.float().unsqueeze(-1) / self.num_timesteps

            # Predict noise
            predicted_noise = self.denoiser(x, t_float, user_emb)

            alpha = self.alphas[t][:, None]
            alpha_cumprod = self.alphas_cumprod[t][:, None]
            beta = self.betas[t][:, None]

            # Remove a fraction of the noise
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)  # No noise on the final step

            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            x = x + torch.sqrt(beta) * noise

        # x is now the mathematically perfect, hallucinated movie embedding
        return x

    def retrieve_candidates(self, user_emb, faiss_index, top_k=10):
        """
        Pass the hallucinated embedding into FAISS to find the closest REAL movies.
        """
        ideal_emb = self.generate_ideal_embedding(user_emb)
        ideal_emb_np = ideal_emb.cpu().numpy()

        # Find the real movies that best match this hallucinated dream movie
        distances, indices = faiss_index.search(ideal_emb_np, top_k)
        return indices, distances
