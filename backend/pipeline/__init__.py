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
    Output:  RankedItem list, sorted by blended score
    Config:  RankingConfig(ensemble_weight, ranker_weight, use_neural_ensemble, use_learned_ranker)

Stage 3 — Reranking (RerankingPipeline):
    Steps:   RL safety filter → quality gate → MMR diversity → LLM explanation
    Output:  FinalItem list, subset of ranked items
    Config:  RerankingConfig(mmr_lambda, enable_llm_reranking, enable_rl_safety, quality_threshold)

Shared types (pipeline_types.py):
    CandidateItem — retrieval stage output
    RankedItem    — ranking stage output
    FinalItem     — reranking stage output

Import graph (strictly acyclic):
    pipeline_types ← retrieval_pipeline, ranking_pipeline, reranking_pipeline ← recommender ← main
"""

from backend.pipeline_types import CandidateItem, FinalItem, RankedItem
from backend.ranking_pipeline import RankingConfig, RankingPipeline
from backend.reranking_pipeline import RerankingConfig, RerankingPipeline
from backend.retrieval_pipeline import RetrievalConfig, RetrievalPipeline

__all__ = [
    # Types
    "CandidateItem",
    "RankedItem",
    "FinalItem",
    # Pipelines
    "RetrievalPipeline",
    "RetrievalConfig",
    "RankingPipeline",
    "RankingConfig",
    "RerankingPipeline",
    "RerankingConfig",
]
