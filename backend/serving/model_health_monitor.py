"""
Model Health Monitor — Production-Grade Ensemble Reliability.

Real-time quality tracking for each model in the 6-model ensemble:
- EWMA (Exponentially Weighted Moving Average) prediction error per model
- EWMA latency per model
- Failure rate tracking (exceptions during scoring)
- Auto-disable: if a model's error exceeds 2× ensemble mean for 100+ predictions
- Auto-recovery: disabled models are shadow-probed every 50 predictions;
  5 consecutive successful probes re-enable the model

This is the same pattern used in Netflix's Zuul and Google's Envoy
for circuit-breaking unreliable microservices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAMES = ("lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion")


@dataclass
class ModelState:
    """Mutable state for a single ensemble model."""

    name: str

    # EWMA tracking (alpha = smoothing factor, higher = more weight on recent)
    ewma_error: float = 0.0
    ewma_latency_ms: float = 0.0
    ewma_alpha: float = 0.1

    # Failure tracking
    total_predictions: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0

    # Auto-disable
    is_disabled: bool = False
    disabled_at: float = 0.0
    degradation_counter: int = 0  # How many consecutive preds exceeded threshold
    degradation_window: int = 100  # Must exceed for this many preds to trigger disable

    # Auto-recovery (shadow probing)
    probe_interval: int = 50  # Probe every N predictions after disable
    probe_success_count: int = 0
    probe_success_threshold: int = 5  # Re-enable after this many consecutive good probes
    predictions_since_disable: int = 0

    # History for reporting
    last_error: float = 0.0
    last_latency_ms: float = 0.0
    last_updated: float = field(default_factory=time.time)


class ModelHealthMonitor:
    """
    Real-time health monitoring and circuit-breaking for ensemble models.

    Thread-safe: all state mutations are guarded by a lock.
    """

    def __init__(
        self,
        model_names: tuple[str, ...] = MODEL_NAMES,
        error_threshold_multiplier: float = 2.0,
        ewma_alpha: float = 0.1,
    ):
        """
        Args:
            model_names: Names of the ensemble models to monitor.
            error_threshold_multiplier: A model is flagged if its EWMA error
                exceeds this multiplier × ensemble mean error.
            ewma_alpha: Smoothing factor for EWMA (higher = more reactive).
        """
        self._lock = threading.Lock()
        self.error_threshold_multiplier = error_threshold_multiplier

        self._states: dict[str, ModelState] = {
            name: ModelState(name=name, ewma_alpha=ewma_alpha) for name in model_names
        }

        self._global_prediction_count = 0

    def record_prediction(
        self,
        model_name: str,
        error: float,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """
        Record the result of a single model prediction.

        Args:
            model_name: Name of the model.
            error: Absolute prediction error (e.g. deviation from ensemble mean).
            latency_ms: Prediction latency in milliseconds.
            success: Whether the prediction completed without exception.
        """
        with self._lock:
            state = self._states.get(model_name)
            if state is None:
                return

            state.total_predictions += 1
            state.last_error = error
            state.last_latency_ms = latency_ms
            state.last_updated = time.time()

            if not success:
                state.total_failures += 1
                state.consecutive_failures += 1
                # Rapid failures also trigger degradation
                state.degradation_counter += 1
            else:
                state.consecutive_failures = 0
                # Update EWMA
                state.ewma_error = state.ewma_alpha * error + (1 - state.ewma_alpha) * state.ewma_error
                state.ewma_latency_ms = state.ewma_alpha * latency_ms + (1 - state.ewma_alpha) * state.ewma_latency_ms

            self._global_prediction_count += 1

            # Check degradation for active models
            if not state.is_disabled:
                self._check_degradation(state)
            else:
                # Auto-recovery: shadow probing
                state.predictions_since_disable += 1
                if success and state.predictions_since_disable % state.probe_interval == 0:
                    self._check_recovery(state, error)

    def _check_degradation(self, state: ModelState) -> None:
        """Check if a model should be auto-disabled (called under lock)."""
        # Compute ensemble mean error (excluding disabled models)
        active_errors = [s.ewma_error for s in self._states.values() if not s.is_disabled and s.total_predictions > 10]

        if not active_errors:
            return

        mean_error = sum(active_errors) / len(active_errors)

        # Check if this model's error exceeds threshold
        threshold = mean_error * self.error_threshold_multiplier
        if state.ewma_error > threshold and mean_error > 0.01:
            state.degradation_counter += 1
        else:
            state.degradation_counter = max(0, state.degradation_counter - 1)

        # Also check for rapid failures
        if state.consecutive_failures >= 10:
            state.degradation_counter = state.degradation_window  # Force disable

        # Trigger disable
        if state.degradation_counter >= state.degradation_window:
            state.is_disabled = True
            state.disabled_at = time.time()
            state.predictions_since_disable = 0
            state.probe_success_count = 0
            logger.warning(
                "Model '%s' AUTO-DISABLED: EWMA error=%.4f exceeds %.1f× ensemble mean=%.4f "
                "for %d consecutive predictions",
                state.name,
                state.ewma_error,
                self.error_threshold_multiplier,
                mean_error,
                state.degradation_window,
            )

    def _check_recovery(self, state: ModelState, probe_error: float) -> None:
        """Check if a disabled model should be re-enabled (called under lock)."""
        # Compute current ensemble mean error
        active_errors = [s.ewma_error for s in self._states.values() if not s.is_disabled and s.total_predictions > 10]

        if not active_errors:
            # All models disabled — force re-enable
            self._reenable_model(state, reason="all models disabled")
            return

        mean_error = sum(active_errors) / len(active_errors)
        threshold = mean_error * self.error_threshold_multiplier

        if probe_error <= threshold:
            state.probe_success_count += 1
            if state.probe_success_count >= state.probe_success_threshold:
                self._reenable_model(state, reason="shadow probe recovery")
        else:
            state.probe_success_count = 0

    def _reenable_model(self, state: ModelState, reason: str) -> None:
        """Re-enable a disabled model (called under lock)."""
        state.is_disabled = False
        state.degradation_counter = 0
        state.probe_success_count = 0
        state.predictions_since_disable = 0
        logger.info(
            "Model '%s' AUTO-RECOVERED (%s) after %.1fs disabled",
            state.name,
            reason,
            time.time() - state.disabled_at,
        )

    def get_active_models(self) -> list[str]:
        """Return list of model names that are currently healthy and active."""
        with self._lock:
            active = [name for name, state in self._states.items() if not state.is_disabled]
            # Safety: never return empty — re-enable all if all disabled
            if not active:
                logger.warning("All models disabled! Re-enabling all models.")
                for state in self._states.values():
                    state.is_disabled = False
                    state.degradation_counter = 0
                active = list(self._states.keys())
            return active

    def is_model_active(self, model_name: str) -> bool:
        """Check if a specific model is currently active."""
        with self._lock:
            state = self._states.get(model_name)
            return state is not None and not state.is_disabled

    def force_disable(self, model_name: str) -> bool:
        """Manually disable a model (e.g. from admin API)."""
        with self._lock:
            state = self._states.get(model_name)
            if state is None:
                return False
            state.is_disabled = True
            state.disabled_at = time.time()
            state.predictions_since_disable = 0
            state.probe_success_count = 0
            logger.info("Model '%s' manually disabled", model_name)
            return True

    def force_enable(self, model_name: str) -> bool:
        """Manually re-enable a model (e.g. from admin API)."""
        with self._lock:
            state = self._states.get(model_name)
            if state is None:
                return False
            self._reenable_model(state, reason="manual re-enable")
            return True

    def get_health_report(self) -> dict[str, Any]:
        """
        Generate a comprehensive health report for all models.

        Returns a dict with per-model stats and ensemble-level summary.
        """
        with self._lock:
            models = {}
            total_active = 0
            total_disabled = 0

            for name, state in self._states.items():
                failure_rate = state.total_failures / max(state.total_predictions, 1)
                models[name] = {
                    "status": "disabled" if state.is_disabled else "active",
                    "ewma_error": round(state.ewma_error, 6),
                    "ewma_latency_ms": round(state.ewma_latency_ms, 2),
                    "total_predictions": state.total_predictions,
                    "total_failures": state.total_failures,
                    "failure_rate": round(failure_rate, 4),
                    "consecutive_failures": state.consecutive_failures,
                    "degradation_counter": state.degradation_counter,
                    "degradation_window": state.degradation_window,
                }
                if state.is_disabled:
                    models[name]["disabled_at"] = state.disabled_at
                    models[name]["disabled_duration_s"] = round(time.time() - state.disabled_at, 1)
                    models[name]["probe_success_count"] = state.probe_success_count
                    total_disabled += 1
                else:
                    total_active += 1

            return {
                "models": models,
                "summary": {
                    "total_models": len(self._states),
                    "active_models": total_active,
                    "disabled_models": total_disabled,
                    "global_prediction_count": self._global_prediction_count,
                },
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_monitor: ModelHealthMonitor | None = None
_monitor_lock = threading.Lock()


def get_model_health_monitor() -> ModelHealthMonitor:
    """Get or create the module-level ModelHealthMonitor singleton."""
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = ModelHealthMonitor()
    return _monitor
