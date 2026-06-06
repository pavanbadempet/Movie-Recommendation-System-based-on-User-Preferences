"""
3-Stage Recommendation Pipeline sub-package for APEX.

The pipeline decomposes the recommendation process into three focused stages
with well-defined interfaces (CandidateItem → RankedItem → FinalItem):

Stage 1 — Retrieval (RetrievalPipeline):
    Sources: FAISS ANN + TF-IDF sparse + Knowledge Graph
    Output:  ~500 CandidateItem objects, deduplicated via max-pool
    Config:  RetrievalConfig(faiss_k, tfidf_k, kg_k, low_memory, enable_kg)

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

from backend.pipeline.diversity_reranker import submodular_rerank
from backend.pipeline.multi_objective_ranker import pareto_rank
from backend.pipeline.pipeline_types import CandidateItem, FinalItem, RankedItem
from backend.pipeline.ranker import load_ranker
from backend.pipeline.ranker_training import build_training_frame, train_nova_ranker
from backend.pipeline.ranking_pipeline import RankingConfig, RankingPipeline
from backend.pipeline.reranking_pipeline import RerankingConfig, RerankingPipeline
from backend.pipeline.retrieval_pipeline import RetrievalConfig, RetrievalPipeline

__all__ = [
    # Shared types
    "CandidateItem",
    "RankedItem",
    "FinalItem",
    # Stage 1: Retrieval
    "RetrievalPipeline",
    "RetrievalConfig",
    # Stage 2: Ranking
    "RankingPipeline",
    "RankingConfig",
    # Stage 3: Reranking
    "RerankingPipeline",
    "RerankingConfig",
    # Support
    "submodular_rerank",
    "pareto_rank",
    "load_ranker",
    "build_training_frame",
    "train_nova_ranker",
]
