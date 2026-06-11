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

# ---------------------------------------------------------------------------
# Lazy Imports — Avoid importing heavy sub-modules at package initialization.
# ---------------------------------------------------------------------------
import importlib

_LAZY_MAPPING = {
    # IPS-debiased metrics
    "compute_item_popularity": "backend.metrics.debiased_metrics",
    "ips_ndcg_at_k": "backend.metrics.debiased_metrics",
    "calibration_score": "backend.metrics.debiased_metrics",
    "beyond_accuracy_metrics": "backend.metrics.debiased_metrics",
    # Label-free evaluation
    "evaluate_recommendation_quality": "backend.metrics.evaluation",
    # Offline benchmarks
    "evaluate_recommendation_benchmark": "backend.metrics.recommendation_benchmark",
    "evaluate_search_benchmark": "backend.metrics.search_benchmark",
    "evaluate_semantic_benchmark": "backend.metrics.semantic_benchmark",
    # Benchmark cache
    "get_cached_recommendation_benchmark": "backend.metrics.benchmark_cache",
    "compute_recommendation_benchmark_cached": "backend.metrics.benchmark_cache",
    "get_cached_semantic_benchmark": "backend.metrics.benchmark_cache",
    "compute_semantic_benchmark_cached": "backend.metrics.benchmark_cache",
    # Uncertainty quantification
    "ensemble_uncertainty": "backend.intelligence.uncertainty_estimator",
    "coverage_uncertainty": "backend.intelligence.uncertainty_estimator",
    "compute_confidence_score": "backend.intelligence.uncertainty_estimator",
    "cold_start_boost": "backend.intelligence.uncertainty_estimator",
}


def __getattr__(name: str):
    if name in _LAZY_MAPPING:
        module_path = _LAZY_MAPPING[name]
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(_LAZY_MAPPING.keys())


__all__ = list(_LAZY_MAPPING.keys())
