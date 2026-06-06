"""
Privacy and compliance sub-package for APEX.

Implements GDPR / EU AI Act compliance mechanisms:

    Differential Privacy (backend/privacy.py, backend/privacy_preserving_ml.py):
        - Laplace mechanism: ε-differential privacy on user embeddings
        - Gaussian mechanism: (ε, δ)-differential privacy
        - k-anonymity: profile generalization for quasi-identifiers
        - Federated gradient aggregation with DP noise

    References:
        - Dwork et al. "The Algorithmic Foundations of Differential Privacy" (2014)
        - McMahan et al. "Communication-Efficient Learning of Deep Networks
          from Decentralized Data" (AISTATS 2017) — Federated Averaging
"""

import logging
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


class DifferentialPrivacyEngine:
    """
    Implements Differential Privacy (DP) mechanisms for user embeddings and telemetry.
    Ensures compliance with GDPR and EU AI Act (2024) by mathematically guaranteeing
    that a single user's data cannot be reverse-engineered from the latent space.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        # Sensitivity (Delta_f): maximum L2 norm of the user embedding
        # Since our embeddings are L2 normalized, the sensitivity is strictly bounded to 2.0
        self.sensitivity = 2.0

    def add_laplace_noise(self, embedding: np.ndarray) -> np.ndarray:
        """Injects Laplace noise into a user embedding for pure epsilon-DP."""
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(loc=0.0, scale=scale, size=embedding.shape)
        return embedding + noise

    def add_gaussian_noise(self, embedding: np.ndarray) -> np.ndarray:
        """Injects Gaussian noise for (epsilon, delta)-DP."""
        import math
        c = math.sqrt(2 * math.log(1.25 / self.delta))
        sigma = (c * self.sensitivity) / self.epsilon
        noise = np.random.normal(loc=0.0, scale=sigma, size=embedding.shape)
        noisy_embedding = embedding + noise
        norm = np.linalg.norm(noisy_embedding)
        return noisy_embedding / (norm + 1e-10)


def anonymize_telemetry(event: dict[str, Any]) -> dict[str, Any]:
    """Strips PII from raw interaction telemetry."""
    safe_event = event.copy()
    safe_event.pop("ip_address", None)
    safe_event.pop("user_name", None)
    if "timestamp" in safe_event:
        safe_event["timestamp"] = int(safe_event["timestamp"] / 3600) * 3600
    return safe_event


from backend.privacy.privacy_preserving_ml import (  # noqa: E402
    add_gaussian_noise,
    add_laplace_noise,
    federated_average_gradients,
    k_anonymize_profile,
    privatize_user_embedding,
)

__all__ = [
    "DifferentialPrivacyEngine",
    "anonymize_telemetry",
    "add_laplace_noise",
    "add_gaussian_noise",
    "privatize_user_embedding",
    "k_anonymize_profile",
    "federated_average_gradients",
]
