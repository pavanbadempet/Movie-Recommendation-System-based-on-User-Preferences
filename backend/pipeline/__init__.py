"""
3-Stage Recommendation Pipeline sub-package for APEX.

The pipeline decomposes the recommendation process into three focused stages
with well-defined interfaces (CandidateItem → RankedItem → FinalItem):

Stage 1 — Retrieval (RetrievalPipeline):
    Sources: TurboVec ANN + TF-IDF sparse + Knowledge Graph
    Output:  ~500 CandidateItem objects, deduplicated via max-pool
    Config:  RetrievalConfig(turbovec_k, tfidf_k, kg_k, low_memory, enable_kg)

Stage 2 — Ranking (RankingPipeline):
    Sources: 6-model neural ensemble + optional learned ranker (LightGBM/MMoE)
    Extras:  IPS popularity-bias correction (apply_ips_reranking=True by default)
    Output:  RankedItem list, sorted by blended score
    Config:  RankingConfig(ensemble_weight, ranker_weight, use_neural_ensemble,
             use_learned_ranker, apply_ips_reranking, item_popularity, ips_clip_val)

Stage 3 — Reranking (RerankingPipeline):
    Steps:   RL safety filter → quality gate → MMR diversity → LLM explanation
    Output:  FinalItem list, subset of ranked items
    Config:  RerankingConfig(mmr_lambda, enable_llm_reranking, enable_rl_safety,
             quality_threshold)

Support modules:
    diversity_reranker.py   — standalone MMR diversity reranking utilities
    ranker.py               — LightGBM/sklearn learned ranker loader
    multi_objective_ranker.py — Pareto-optimal multi-objective ranking

Shared types (pipeline_types.py):
    CandidateItem — retrieval stage output
    RankedItem    — ranking stage output
    FinalItem     — reranking stage output

Import graph (strictly acyclic):
    pipeline_types ← retrieval_pipeline, ranking_pipeline, reranking_pipeline ← recommender ← main
"""

# ---------------------------------------------------------------------------
# Lazy Imports — Avoid importing heavy ML dependencies (e.g. torch, turbovec,
# numpy, sklearn) at package initialization time.
# ---------------------------------------------------------------------------
import importlib

_LAZY_MAPPING = {
    # Shared types
    "CandidateItem": "backend.pipeline.pipeline_types",
    "RankedItem": "backend.pipeline.pipeline_types",
    "FinalItem": "backend.pipeline.pipeline_types",
    # Stage 1: Retrieval
    "RetrievalPipeline": "backend.pipeline.retrieval_pipeline",
    "RetrievalConfig": "backend.pipeline.retrieval_pipeline",
    # Stage 2: Ranking
    "RankingPipeline": "backend.pipeline.ranking_pipeline",
    "RankingConfig": "backend.pipeline.ranking_pipeline",
    # Stage 3: Reranking
    "RerankingPipeline": "backend.pipeline.reranking_pipeline",
    "RerankingConfig": "backend.pipeline.reranking_pipeline",
    # Support
    "submodular_rerank": "backend.pipeline.diversity_reranker",
    "pareto_rank": "backend.pipeline.multi_objective_ranker",
    "load_ranker": "backend.pipeline.ranker",
    "build_training_frame": "backend.pipeline.ranker_training",
    "train_nova_ranker": "backend.pipeline.ranker_training",
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
