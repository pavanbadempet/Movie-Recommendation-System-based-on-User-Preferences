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

# ---------------------------------------------------------------------------
# Lazy Imports — Avoid importing heavy sub-modules at package initialization.
# ---------------------------------------------------------------------------
import importlib

_LAZY_MAPPING = {
    # Knowledge & Reasoning
    "KnowledgeGraphEngine": "backend.intelligence.knowledge_graph",
    "enrich_knowledge_graph_with_cross_domain": "backend.intelligence.cross_domain_kg",
    "build_semantic_twin": "backend.intelligence.semantic_twin",
    "compare_semantic_twins": "backend.intelligence.semantic_twin",
    "ContentUnderstandingEngine": "backend.intelligence.content_understanding",
    # Personalization
    "parse_query_intent": "backend.intelligence.query_understanding",
    "intent_score": "backend.intelligence.query_understanding",
    "generate_explanation": "backend.intelligence.llm_explanations",
    "chat_completion": "backend.intelligence.openrouter_client",
    "MultiModalFusionIndex": "backend.intelligence.multimodal_fusion",
    "VisionEncoder": "backend.intelligence.vision_encoder",
    # Session modeling
    "get_user_attention_encoder": "backend.models.attention_user_model",
    "build_attended_user_embedding": "backend.models.attention_user_model",
    # Long-horizon RL
    "estimate_churn_risk": "backend.intelligence.long_horizon_rl",
    "compute_preference_stability": "backend.intelligence.long_horizon_rl",
    "long_horizon_score_adjustment": "backend.intelligence.long_horizon_rl",
    "build_temporal_user_profile": "backend.intelligence.temporal_preference",
    "temporal_score_boost": "backend.intelligence.temporal_preference",
    # RL reward
    "RLRewardEngine": "backend.learning.rl_reward",
    # Exploration
    "get_bandit_engine": "backend.intelligence.contextual_bandit",
    "ThompsonSamplingBandit": "backend.intelligence.exploration_engine",
    "get_thompson_bandit": "backend.intelligence.exploration_engine",
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
