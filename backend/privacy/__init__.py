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

from backend.privacy.privacy_preserving_ml import (
    add_gaussian_noise,
    add_laplace_noise,
    federated_average_gradients,
    k_anonymize_profile,
    privatize_user_embedding,
)
from backend.privacy.engine import DifferentialPrivacyEngine, anonymize_telemetry

__all__ = [
    "add_laplace_noise",
    "add_gaussian_noise",
    "privatize_user_embedding",
    "k_anonymize_profile",
    "federated_average_gradients",
    "DifferentialPrivacyEngine",
    "anonymize_telemetry",
]
