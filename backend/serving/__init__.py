"""
Serving infrastructure sub-package for APEX.

Handles hardware-adaptive tier selection, ONNX runtime inference,
online learning, and active inference self-healing.

Modules:
    serving_tier.py                 — TierDetector: auto-selects Tier1/2/3 at startup
    onnx_engine.py                  — ONNX Runtime quantized inference (Tier2)
    online_learner.py               — Real-time LightGCN mini-batch updates
    sasrec_online_learner.py        — Real-time SASRec fine-tuning from live events
    kan_online_learner.py           — Real-time KAN Fourier coefficient updates
    online_learning_coordinator.py  — Unified fan-out coordinator (LightGCN + SASRec + KAN)
    active_inference_engine.py      — Free-energy minimization self-healing
    realtime_feature_updater.py     — In-memory session sequence index
    slo.py                          — Request SLO telemetry (p50/p95/p99 latency tracking)
    cache.py                        — Redis-backed in-process cache
    feature_store.py                — Real-time user feature store (Redis-backed)

Tier selection logic:
    Tier1: GPU present AND RAM >= 16 GB  → full 6-model PyTorch ensemble + all online learners
    Tier2: No GPU AND RAM >= 8 GB        → ONNX Runtime quantized models
    Tier3: RAM < 8 GB                    → FAISS + TF-IDF only

Online learning (Tier1 only):
    All three highest-weighted ensemble models receive incremental gradient
    updates from live click and rating events via OnlineLearningCoordinator:
      - LightGCN  (DR weight 0.005) — BPR embedding updates
      - SASRec    (DR weight 0.659) — attention + item embedding fine-tuning
      - KAN       (DR weight 0.298) — Fourier coefficient updates

Override via environment:
    NOVA_SERVING_TIER=tier1|tier2|tier3  (explicit override)
    NOVA_SERVING_PROFILE=full|lite       (legacy mapping)
"""

from backend.intelligence.active_inference_engine import get_active_inference_engine
from backend.serving.artifact_health import evaluate_artifact_health
from backend.serving.artifact_validator import ArtifactValidator
from backend.learning.kan_online_learner import KANOnlineLearner
from backend.models.model_loader import ensure_model_files
from backend.learning.online_learner import OnlineLearner
from backend.learning.online_learning_coordinator import OnlineLearningCoordinator
from backend.serving.onnx_engine import get_onnx_engine
from backend.serving.realtime_feature_updater import get_user_session_sequence, update_user_index
from backend.learning.sasrec_online_learner import SASRecOnlineLearner
from backend.serving.serving_tier import HardwareProfile, TierDetector, resolve_serving_tier
from backend.serving.slo import RequestSloTracker, build_slo_report

__all__ = [
    # Tier detection
    "TierDetector",
    "HardwareProfile",
    "resolve_serving_tier",
    # ONNX inference
    "get_onnx_engine",
    # Online learning
    "OnlineLearner",
    "SASRecOnlineLearner",
    "KANOnlineLearner",
    "OnlineLearningCoordinator",
    # Active inference
    "get_active_inference_engine",
    # Real-time features
    "get_user_session_sequence",
    "update_user_index",
    # Artifact management
    "evaluate_artifact_health",
    "ArtifactValidator",
    "ensure_model_files",
    # SLO
    "RequestSloTracker",
    "build_slo_report",
]
