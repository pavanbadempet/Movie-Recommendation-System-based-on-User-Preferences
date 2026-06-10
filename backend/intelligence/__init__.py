"""
Cognitive intelligence sub-package for APEX.

Implements the Layer 4 intelligence stack — the features that elevate
recommendations from "statistically likely" to "genuinely understood":

Knowledge & Reasoning:
    knowledge_graph.py       — NetworkX multi-hop semantic reasoning
                               (User → Liked Theme → New Movie)
    cross_domain_kg.py       — Cross-domain KG enrichment (books, music,
                               games cross-pollination)
    semantic_twin.py         — Deterministic semantic item twin construction
                               (abstract mood/theme fingerprinting)
    content_understanding.py — HuggingFace Zero-Shot NLP classification +
                               NER entity extraction

Personalization:
    query_understanding.py    — Intent parsing (mood, genre, era, abstract concept)
    llm_explanations.py       — LLM-generated 1-sentence personalized explanations
    openrouter_client.py      — OpenRouter API client (GPT-4o / Llama 3)
    multimodal_fusion.py      — CLIP visual + SBERT text embedding fusion
    vision_encoder.py         — Poster image CLIP encoder

Long-horizon intelligence:
    long_horizon_rl.py        — 30/90-day churn risk + preference stability modeling
    temporal_preference.py    — Time-decay weighted preference modeling
    contextual_bandit.py      — Thompson Sampling / UCB exploration for discovery
    exploration_engine.py     — Controlled serendipity and novelty injection
    attention_user_model.py   — Attention-weighted session sequence user model

Compliance:
    uncertainty_estimator.py  — Ensemble disagreement + cold-start confidence scoring
                                (lives in backend.metrics but logically part of Layer 4)
"""

from backend.intelligence.content_understanding import ContentUnderstandingEngine
from backend.intelligence.contextual_bandit import get_bandit_engine
from backend.intelligence.cross_domain_kg import enrich_knowledge_graph_with_cross_domain
from backend.intelligence.exploration_engine import ThompsonSamplingBandit, get_thompson_bandit
from backend.intelligence.knowledge_graph import KnowledgeGraphEngine
from backend.intelligence.llm_explanations import generate_explanation
from backend.intelligence.long_horizon_rl import (
    compute_preference_stability,
    estimate_churn_risk,
    long_horizon_score_adjustment,
)
from backend.intelligence.multimodal_fusion import MultiModalFusionIndex
from backend.intelligence.openrouter_client import chat_completion
from backend.intelligence.query_understanding import intent_score, parse_query_intent
from backend.intelligence.semantic_twin import build_semantic_twin, compare_semantic_twins
from backend.intelligence.temporal_preference import build_temporal_user_profile, temporal_score_boost
from backend.intelligence.vision_encoder import VisionEncoder
from backend.learning.rl_reward import RLRewardEngine
from backend.models.attention_user_model import build_attended_user_embedding, get_user_attention_encoder

__all__ = [
    # Knowledge & Reasoning
    "KnowledgeGraphEngine",
    "enrich_knowledge_graph_with_cross_domain",
    "build_semantic_twin",
    "compare_semantic_twins",
    "ContentUnderstandingEngine",
    # Personalization
    "parse_query_intent",
    "intent_score",
    "generate_explanation",
    "chat_completion",
    "MultiModalFusionIndex",
    "VisionEncoder",
    # Session modeling
    "get_user_attention_encoder",
    "build_attended_user_embedding",
    # Long-horizon RL
    "estimate_churn_risk",
    "compute_preference_stability",
    "long_horizon_score_adjustment",
    "build_temporal_user_profile",
    "temporal_score_boost",
    # RL reward
    "RLRewardEngine",
    # Exploration
    "get_bandit_engine",
    "ThompsonSamplingBandit",
    "get_thompson_bandit",
]
