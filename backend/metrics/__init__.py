"""
Evaluation metrics sub-package for APEX.

Implements popularity-debiased evaluation metrics using Inverse Propensity
Scoring (IPS) following Schnabel et al. "Recommendations as Treatments"
(ICML 2016).

Note: Source modules remain in backend/ for backward compatibility.
This sub-package provides logical namespacing and a documentation anchor.
"""

from backend.debiased_metrics import (
    beyond_accuracy_metrics,
    calibration_score,
    compute_item_popularity,
    ips_ndcg_at_k,
)
from backend.evaluation import evaluate_recommendation_quality

__all__ = [
    "beyond_accuracy_metrics",
    "calibration_score",
    "compute_item_popularity",
    "evaluate_recommendation_quality",
    "ips_ndcg_at_k",
]
