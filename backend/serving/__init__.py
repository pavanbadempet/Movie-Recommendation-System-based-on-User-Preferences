"""
Serving infrastructure sub-package for APEX.

Handles hardware-adaptive tier selection, ONNX runtime inference,
online learning, and active inference self-healing.

Modules:
    serving_tier.py          — TierDetector: auto-selects Tier1/2/3 at startup
    onnx_engine.py           — ONNX Runtime quantized inference (Tier2)
    online_learner.py        — Real-time LightGCN mini-batch updates
    active_inference_engine.py — Free-energy minimization self-healing
    realtime_feature_updater.py — In-memory session sequence index

Tier selection logic:
    Tier1: GPU present AND RAM >= 16 GB  → full 6-model PyTorch ensemble
    Tier2: No GPU AND RAM >= 8 GB        → ONNX Runtime quantized models
    Tier3: RAM < 8 GB                    → FAISS + TF-IDF only

Override via environment:
    NOVA_SERVING_TIER=tier1|tier2|tier3  (explicit override)
    NOVA_SERVING_PROFILE=full|lite       (legacy mapping)
"""

from backend.serving_tier import HardwareProfile, TierDetector, resolve_serving_tier

__all__ = [
    "TierDetector",
    "HardwareProfile",
    "resolve_serving_tier",
]
