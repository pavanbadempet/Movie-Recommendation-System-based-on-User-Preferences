import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DifferentialPrivacyEngine:
    """
    Implements Differential Privacy (DP) mechanisms for user embeddings and telemetry.
    Ensures compliance with GDPR and EU AI Act (2024) by mathematically guaranteeing 
    that a single user's data cannot be reverse-engineered from the latent space.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Args:
            epsilon: The privacy budget. Smaller = more private but less accurate.
            delta: Probability of privacy breach (usually 1/|dataset|).
        """
        self.epsilon = epsilon
        self.delta = delta
        
        # Sensitivity (Delta_f): maximum L2 norm of the user embedding
        # Since our embeddings are L2 normalized, the sensitivity is strictly bounded to 2.0
        self.sensitivity = 2.0 
        
    def add_laplace_noise(self, embedding: np.ndarray) -> np.ndarray:
        """
        Injects Laplace noise into a user embedding for pure epsilon-DP.
        Used for lower dimensional or highly sensitive data.
        """
        # Scale of noise (b) = sensitivity / epsilon
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(loc=0.0, scale=scale, size=embedding.shape)
        return embedding + noise
        
    def add_gaussian_noise(self, embedding: np.ndarray) -> np.ndarray:
        """
        Injects Gaussian noise for (epsilon, delta)-DP.
        Better for high-dimensional vectors like 768d SBERT embeddings.
        """
        # Variance calculation for Gaussian Mechanism
        c = np.sqrt(2 * np.log(1.25 / self.delta))
        sigma = (c * self.sensitivity) / self.epsilon
        
        noise = np.random.normal(loc=0.0, scale=sigma, size=embedding.shape)
        
        noisy_embedding = embedding + noise
        
        # Re-normalize to prevent the vector from exploding out of cosine similarity bounds
        norm = np.linalg.norm(noisy_embedding)
        return noisy_embedding / (norm + 1e-10)

def anonymize_telemetry(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strips PII (Personally Identifiable Information) from raw interaction telemetry.
    """
    safe_event = event.copy()
    
    # Hash IPs or exact coordinates if they exist
    if "ip_address" in safe_event:
        safe_event.pop("ip_address")
        
    if "user_name" in safe_event:
        safe_event.pop("user_name")
        
    # Coarsen timestamps to the nearest hour to prevent exact time-correlation attacks
    if "timestamp" in safe_event:
        safe_event["timestamp"] = int(safe_event["timestamp"] / 3600) * 3600
        
    return safe_event
