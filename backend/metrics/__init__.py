"""
Evaluation metrics sub-package for APEX.

Implements popularity-debiased evaluation metrics using Inverse Propensity
Scoring (IPS) following Schnabel et al. "Recommendations as Treatments"
(ICML 2016).

Modules:
    debiased_metrics.py        — IPS-corrected NDCG, calibration, beyond-accuracy metrics
    evaluation.py              — Label-free artifact health evaluation (vector norms, coverage)
    recommendation_benchmark.py — Offline leave-one-out HR@k / NDCG@k benchmark
    search_benchmark.py        — Semantic search quality benchmark
    semantic_benchmark.py      — 17-case curated intent benchmark (HR@10 = 1.0)
    uncertainty_estimator.py   — Ensemble disagreement + cold-start confidence scoring

Note: Source modules remain in backend/ for backward compatibility.
This sub-package provides logical namespacing and a documentation anchor.
"""

from backend.metrics.benchmark_cache import (
    compute_recommendation_benchmark_cached,
    compute_semantic_benchmark_cached,
    get_cached_recommendation_benchmark,
    get_cached_semantic_benchmark,
)
from backend.metrics.debiased_metrics import (
    beyond_accuracy_metrics,
    calibration_score,
    compute_item_popularity,
    ips_ndcg_at_k,
)
from backend.metrics.evaluation import evaluate_recommendation_quality
from backend.metrics.recommendation_benchmark import evaluate_recommendation_benchmark
from backend.metrics.search_benchmark import evaluate_search_benchmark
from backend.metrics.semantic_benchmark import evaluate_semantic_benchmark
from backend.intelligence.uncertainty_estimator import (
    cold_start_boost,
    compute_confidence_score,
    coverage_uncertainty,
    ensemble_uncertainty,
)

__all__ = [
    # IPS-debiased metrics
    "compute_item_popularity",
    "ips_ndcg_at_k",
    "calibration_score",
    "beyond_accuracy_metrics",
    # Label-free evaluation
    "evaluate_recommendation_quality",
    # Offline benchmarks
    "evaluate_recommendation_benchmark",
    "evaluate_search_benchmark",
    "evaluate_semantic_benchmark",
    # Benchmark cache
    "get_cached_recommendation_benchmark",
    "compute_recommendation_benchmark_cached",
    "get_cached_semantic_benchmark",
    "compute_semantic_benchmark_cached",
    # Uncertainty quantification
    "ensemble_uncertainty",
    "coverage_uncertainty",
    "compute_confidence_score",
    "cold_start_boost",
]
