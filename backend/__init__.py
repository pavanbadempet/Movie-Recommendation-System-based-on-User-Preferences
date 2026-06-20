"""
APEX Backend Package.

Public API surface for the APEX recommendation engine.

Pipeline architecture (3-stage):
    RetrievalPipeline  → RankingPipeline  → RerankingPipeline
    (TurboVec+TF-IDF+KG)    (6-model ensemble)  (MMR+RL+LLM)

Shared data types flow between stages via pipeline_types:
    CandidateItem → RankedItem → FinalItem

Serving tier auto-detection:
    TierDetector resolves Tier1 (GPU) / Tier2 (ONNX CPU) / Tier3 (TurboVec lite)
    at startup based on available hardware.

Ensemble models (6):
    LightGCN, SASRec, KAN, QuantumFluidRecommender,
    HyperbolicRecommender, LatentDiffusionRecommender

Weights are DR-optimized (Doubly Robust IPS) and stored in
models/ensemble_weights.json. Hot-reloadable without restart via
ApexEnsembleEngine.reload_weights().
"""

# ---------------------------------------------------------------------------
# Pipeline types — stable public API, safe to import anywhere
# ---------------------------------------------------------------------------
from backend.pipeline.pipeline_types import CandidateItem, FinalItem, RankedItem

# ---------------------------------------------------------------------------
# Pipeline stages — import on demand to avoid heavy ML deps at package import
# ---------------------------------------------------------------------------
# from backend.pipeline.retrieval_pipeline import RetrievalPipeline, RetrievalConfig
# from backend.pipeline.ranking_pipeline import RankingPipeline, RankingConfig
# from backend.pipeline.reranking_pipeline import RerankingPipeline, RerankingConfig
# ---------------------------------------------------------------------------
# Serving tier detection — lightweight, no ML deps
# ---------------------------------------------------------------------------
from backend.serving.serving_tier import TierDetector, resolve_serving_tier

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------
__version__ = "2.0.0"

__all__ = [
    # Pipeline types
    "CandidateItem",
    "RankedItem",
    "FinalItem",
    # Serving tier
    "TierDetector",
    "resolve_serving_tier",
    # Version
    "__version__",
]
