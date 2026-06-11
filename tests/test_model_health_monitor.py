"""
Tests for ModelHealthMonitor — per-model quality tracking with auto-disable/recovery.

Covers:
- EWMA calculation correctness
- Auto-disable triggers after sustained degradation
- Auto-recovery after shadow probing
- Health report structure
- Manual enable/disable
- Safety: all-models-disabled fallback
"""

import pytest

from backend.serving.model_health_monitor import ModelHealthMonitor


@pytest.fixture
def monitor():
    return ModelHealthMonitor(
        model_names=("lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion"),
        error_threshold_multiplier=2.0,
        ewma_alpha=0.3,  # Higher alpha for faster convergence in tests
    )


class TestEWMATracking:
    def test_ewma_error_updates(self, monitor):
        """EWMA error should track recorded values."""
        # Record several predictions with known error
        for _ in range(20):
            monitor.record_prediction("lightgcn", error=0.5, latency_ms=10.0)

        state = monitor._states["lightgcn"]
        # After many observations, EWMA should converge toward 0.5
        assert 0.3 < state.ewma_error < 0.7

    def test_ewma_latency_updates(self, monitor):
        """EWMA latency should track recorded values."""
        for _ in range(20):
            monitor.record_prediction("quantum", error=0.1, latency_ms=50.0)

        state = monitor._states["quantum"]
        assert 30.0 < state.ewma_latency_ms < 70.0

    def test_failure_tracking(self, monitor):
        """Failures should increment counters."""
        monitor.record_prediction("kan", error=1.0, latency_ms=100.0, success=False)
        monitor.record_prediction("kan", error=1.0, latency_ms=100.0, success=False)

        state = monitor._states["kan"]
        assert state.total_failures == 2
        assert state.consecutive_failures == 2

    def test_consecutive_failures_reset_on_success(self, monitor):
        """Consecutive failure counter should reset on success."""
        monitor.record_prediction("sasrec", error=1.0, latency_ms=50.0, success=False)
        monitor.record_prediction("sasrec", error=1.0, latency_ms=50.0, success=False)
        monitor.record_prediction("sasrec", error=0.1, latency_ms=10.0, success=True)

        state = monitor._states["sasrec"]
        assert state.consecutive_failures == 0
        assert state.total_failures == 2


class TestAutoDisable:
    def test_model_disabled_after_sustained_high_error(self, monitor):
        """A model with persistently high error should be auto-disabled."""
        # First, establish a baseline for other models
        for _ in range(20):
            for m in ("lightgcn", "quantum", "sasrec", "kan", "hyperbolic"):
                monitor.record_prediction(m, error=0.1, latency_ms=10.0)

        # Now record sustained high error for diffusion
        for _ in range(150):
            monitor.record_prediction("diffusion", error=1.0, latency_ms=100.0)

        state = monitor._states["diffusion"]
        assert state.is_disabled, "Diffusion model should be auto-disabled after sustained high error"

    def test_model_disabled_after_rapid_failures(self, monitor):
        """10 consecutive failures should force-disable a model."""
        # Establish baselines
        for _ in range(20):
            for m in ("lightgcn", "quantum", "sasrec", "kan", "hyperbolic"):
                monitor.record_prediction(m, error=0.1, latency_ms=10.0)

        # 10 consecutive failures for diffusion
        for _ in range(10):
            monitor.record_prediction("diffusion", error=0.0, latency_ms=0.0, success=False)

        state = monitor._states["diffusion"]
        assert state.is_disabled

    def test_disabled_model_excluded_from_active(self, monitor):
        """Disabled models should not appear in get_active_models()."""
        monitor.force_disable("diffusion")
        active = monitor.get_active_models()
        assert "diffusion" not in active
        assert len(active) == 5


class TestAutoRecovery:
    def test_model_recovers_after_shadow_probing(self, monitor):
        """A disabled model should recover after successful shadow probes."""
        # Disable the model
        monitor.force_disable("kan")
        state = monitor._states["kan"]

        # Establish baselines for active models
        for _ in range(20):
            for m in ("lightgcn", "quantum", "sasrec", "hyperbolic", "diffusion"):
                monitor.record_prediction(m, error=0.1, latency_ms=10.0)

        # Shadow probe at intervals (probe_interval=50)
        # Simulate enough predictions to trigger probes
        for i in range(1, 300):
            monitor.record_prediction("kan", error=0.05, latency_ms=5.0, success=True)

        # After enough successful probes, model should be re-enabled
        assert not state.is_disabled, "Model should auto-recover after successful shadow probes"

    def test_all_disabled_safety_fallback(self, monitor):
        """If all models are disabled, get_active_models should re-enable all."""
        for m in ("lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion"):
            monitor.force_disable(m)

        active = monitor.get_active_models()
        assert len(active) == 6  # All re-enabled as safety fallback


class TestManualControl:
    def test_force_disable(self, monitor):
        result = monitor.force_disable("lightgcn")
        assert result is True
        assert monitor._states["lightgcn"].is_disabled

    def test_force_enable(self, monitor):
        monitor.force_disable("lightgcn")
        result = monitor.force_enable("lightgcn")
        assert result is True
        assert not monitor._states["lightgcn"].is_disabled

    def test_force_disable_nonexistent(self, monitor):
        result = monitor.force_disable("nonexistent_model")
        assert result is False

    def test_is_model_active(self, monitor):
        assert monitor.is_model_active("lightgcn") is True
        monitor.force_disable("lightgcn")
        assert monitor.is_model_active("lightgcn") is False


class TestHealthReport:
    def test_report_structure(self, monitor):
        report = monitor.get_health_report()
        assert "models" in report
        assert "summary" in report
        assert report["summary"]["total_models"] == 6
        assert report["summary"]["active_models"] == 6
        assert report["summary"]["disabled_models"] == 0

    def test_report_per_model_fields(self, monitor):
        monitor.record_prediction("lightgcn", error=0.1, latency_ms=10.0)
        report = monitor.get_health_report()

        lgcn = report["models"]["lightgcn"]
        assert lgcn["status"] == "active"
        assert "ewma_error" in lgcn
        assert "ewma_latency_ms" in lgcn
        assert "total_predictions" in lgcn
        assert "failure_rate" in lgcn
        assert lgcn["total_predictions"] == 1

    def test_report_disabled_model_details(self, monitor):
        monitor.force_disable("diffusion")
        report = monitor.get_health_report()

        diff = report["models"]["diffusion"]
        assert diff["status"] == "disabled"
        assert "disabled_at" in diff
        assert "disabled_duration_s" in diff
        assert report["summary"]["disabled_models"] == 1
        assert report["summary"]["active_models"] == 5

    def test_global_prediction_count(self, monitor):
        for _ in range(5):
            monitor.record_prediction("lightgcn", error=0.1, latency_ms=10.0)
            monitor.record_prediction("quantum", error=0.2, latency_ms=20.0)

        report = monitor.get_health_report()
        assert report["summary"]["global_prediction_count"] == 10


class TestEngineIntegration:
    def test_engine_has_health_monitor(self):
        """Verify that the engine singleton initializes the health monitor."""
        import os

        os.environ["NOVA_DISABLE_MODEL_DOWNLOADS"] = "1"
        os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key-for-ci-only"

        from backend.models.ensemble_engine import get_apex_engine

        engine = get_apex_engine()

        assert engine.health_monitor is not None
        assert engine.router_trainer is not None

    def test_engine_get_system_health(self):
        """Verify the get_system_health method returns structured data."""
        import os

        os.environ["NOVA_DISABLE_MODEL_DOWNLOADS"] = "1"
        os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key-for-ci-only"

        from backend.models.ensemble_engine import get_apex_engine

        engine = get_apex_engine()

        health = engine.get_system_health()
        assert health["engine"] == "ApexEnsembleEngine"
        assert "model_health" in health
        assert "router_trainer" in health
        assert "privacy_budget" in health
