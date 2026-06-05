"""
Adaptive Serving Tier Detection.

Auto-detects hardware at startup and resolves the appropriate serving tier:
  tier1 — Enterprise: GPU + full 6-model ensemble + Redis
  tier2 — Professional: ONNX CPU inference + PostgreSQL
  tier3 — Starter: FAISS + TF-IDF only + SQLite

Usage:
    from backend.serving_tier import resolve_serving_tier
    tier, reason = resolve_serving_tier()
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    gpu_available: bool
    ram_gb: float
    cpu_cores: int


class TierDetector:
    VALID_TIERS = frozenset({"tier1", "tier2", "tier3"})

    def __init__(self):
        self._profile: HardwareProfile | None = None
        self._tier: str | None = None
        self._reason: str | None = None
        self._detected: bool = False

    def detect(self) -> HardwareProfile:
        """Detect hardware metrics with safe defaults on any exception."""
        # GPU
        try:
            import torch

            gpu_available = torch.cuda.is_available()
        except Exception as exc:
            logger.warning("GPU detection failed (%s); defaulting to False", exc)
            gpu_available = False

        # RAM
        try:
            import psutil

            ram_gb = psutil.virtual_memory().total / (1024**3)
        except Exception as exc:
            logger.warning("RAM detection failed (%s); defaulting to 4.0 GB", exc)
            ram_gb = 4.0

        # CPU cores
        try:
            cpu_cores = os.cpu_count() or 2
        except Exception as exc:
            logger.warning("CPU core detection failed (%s); defaulting to 2", exc)
            cpu_cores = 2

        self._profile = HardwareProfile(
            gpu_available=bool(gpu_available),
            ram_gb=float(max(ram_gb, 0.1)),
            cpu_cores=int(max(cpu_cores, 1)),
        )
        self._detected = True
        logger.info(
            "Hardware detected: gpu=%s, ram_gb=%.1f, cpu_cores=%d",
            self._profile.gpu_available,
            self._profile.ram_gb,
            self._profile.cpu_cores,
        )
        return self._profile

    def resolve(self) -> tuple[str, str]:
        """Return (tier, reason). Calls detect() if not yet done."""
        if not self._detected:
            self.detect()

        profile = self._profile

        # 1. Explicit override
        explicit = os.getenv("NOVA_SERVING_TIER", "").strip().lower()
        if explicit:
            if explicit in self.VALID_TIERS:
                tier, reason = explicit, "explicit_override"
            else:
                logger.error(
                    "NOVA_SERVING_TIER=%r is not valid; falling back to auto-detection",
                    explicit,
                )
                tier, reason = self._auto_select(profile)
        else:
            # 2. Legacy profile mapping
            legacy = os.getenv("NOVA_SERVING_PROFILE", "").strip().lower()
            if legacy == "full":
                tier, reason = "tier1", "legacy_profile_mapping"
            elif legacy in {"lite", "light", "low-memory", "metadata"}:
                tier, reason = "tier3", "legacy_profile_mapping"
            else:
                # 3. Hardware auto-detection
                tier, reason = self._auto_select(profile)

        self._tier = tier
        self._reason = reason
        logger.info(
            "Serving tier resolved: %s (reason=%s, gpu=%s, ram_gb=%.1f, cpu_cores=%d)",
            tier,
            reason,
            profile.gpu_available,
            profile.ram_gb,
            profile.cpu_cores,
        )
        return tier, reason

    def _auto_select(self, profile: HardwareProfile) -> tuple[str, str]:
        if profile.ram_gb < 8.0:
            return "tier3", "hardware_auto_detection"
        if profile.gpu_available and profile.ram_gb >= 16.0:
            return "tier1", "hardware_auto_detection"
        return "tier2", "hardware_auto_detection"


_detector: TierDetector | None = None
_detector_lock: threading.Lock | None = None


def get_tier_detector() -> TierDetector:
    """Return the module-level singleton TierDetector (thread-safe)."""
    global _detector, _detector_lock
    if _detector_lock is None:
        import threading

        _detector_lock = threading.Lock()
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = TierDetector()
    return _detector


def resolve_serving_tier() -> tuple[str, str]:
    """Return (tier, reason). Safe to call before lifespan completes."""
    return get_tier_detector().resolve()
