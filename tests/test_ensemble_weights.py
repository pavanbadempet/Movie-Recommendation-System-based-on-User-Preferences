"""
Property-based and unit tests for ApexEnsembleEngine._load_weights and reload_weights.

Feature: apex-peak-capability, Property 3: Ensemble Weights Sum to One
Validates: Requirements 2.5
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from hypothesis import assume, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WEIGHT_KEYS = ("lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion", "clifford")
DEFAULT_WEIGHTS = {
    "lightgcn": 0.60,
    "quantum": 0.20,
    "sasrec": 0.10,
    "clifford": 0.05,
    "kan": 0.00,
    "hyperbolic": 0.05,
    "diffusion": 0.00,
}


def _make_engine(tmp_path: Path):
    """Return a small ApexEnsembleEngine with MODELS_DIR patched to tmp_path."""
    with (
        patch("backend.models.ensemble_engine.QuantumFluidRecommender"),
        patch("backend.models.ensemble_engine.HyperbolicRecommender"),
        patch("backend.models.ensemble_engine.KANRanker"),
        patch("backend.models.ensemble_engine.LatentDiffusionRecommender"),
        patch("backend.models.ensemble_engine.SASRec"),
        patch("backend.models.ensemble_engine.LightGCN"),
        patch("backend.models.ensemble_engine.CliffordRecommender"),
        patch("backend.models.ensemble_engine.ApexEnsembleEngine._inject_pyspark_priors"),
        patch("backend.models.ensemble_engine.ApexEnsembleEngine._load_trained_weights"),
        patch("backend.models.ensemble_engine.MODELS_DIR", tmp_path),
    ):
        from backend.models.ensemble_engine import ApexEnsembleEngine

        engine = ApexEnsembleEngine(num_users=10, num_items=100, emb_dim=4)
    return engine


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@given(
    vals=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=7,
        max_size=7,
    )
)
@settings(max_examples=100, deadline=None, suppress_health_check=["function_scoped_fixture"])
def test_weights_sum_to_one(tmp_path, vals):
    """
    Feature: apex-peak-capability, Property 3
    For any 7 non-negative floats written to ensemble_weights.json,
    _load_weights must return values that sum to 1.0 ± 1e-6 with all >= 0.
    """
    assume(sum(vals) > 0)

    weights_file = tmp_path / "ensemble_weights.json"
    weights_file.write_text(json.dumps(dict(zip(WEIGHT_KEYS, vals, strict=False))), encoding="utf-8")

    with patch("backend.models.ensemble_engine.MODELS_DIR", tmp_path):
        from backend.models.ensemble_engine import ApexEnsembleEngine

        with (
            patch("backend.models.ensemble_engine.QuantumFluidRecommender"),
            patch("backend.models.ensemble_engine.HyperbolicRecommender"),
            patch("backend.models.ensemble_engine.KANRanker"),
            patch("backend.models.ensemble_engine.LatentDiffusionRecommender"),
            patch("backend.models.ensemble_engine.SASRec"),
            patch("backend.models.ensemble_engine.LightGCN"),
            patch("backend.models.ensemble_engine.CliffordRecommender"),
            patch("backend.models.ensemble_engine.ApexEnsembleEngine._inject_pyspark_priors"),
            patch("backend.models.ensemble_engine.ApexEnsembleEngine._load_trained_weights"),
        ):
            engine = ApexEnsembleEngine(num_users=10, num_items=100, emb_dim=4)
            loaded = engine._load_weights()

    total = sum(loaded.values())
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"
    assert all(v >= 0 for v in loaded.values()), "Negative weight found"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestEnsembleWeightsUnit:
    def test_load_valid_file(self, tmp_path, monkeypatch):
        """Valid file with weights summing to 1.0 → exact values returned."""
        import backend.models.ensemble_engine as ee_mod

        monkeypatch.setattr(ee_mod, "MODELS_DIR", tmp_path)
        weights = dict(zip(WEIGHT_KEYS, [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05], strict=False))
        (tmp_path / "ensemble_weights.json").write_text(json.dumps(weights))
        engine = _make_engine(tmp_path)
        loaded = engine._load_weights()
        assert abs(sum(loaded.values()) - 1.0) < 1e-6
        assert abs(loaded["lightgcn"] - 0.3) < 1e-6

    def test_missing_file_returns_defaults(self, tmp_path, monkeypatch):
        """No file → hard-coded defaults returned."""
        import backend.models.ensemble_engine as ee_mod

        monkeypatch.setattr(ee_mod, "MODELS_DIR", tmp_path)
        engine = _make_engine(tmp_path)
        loaded = engine._load_weights()
        assert abs(loaded["lightgcn"] - 0.60) < 1e-6
        assert abs(loaded["quantum"] - 0.20) < 1e-6
        assert abs(loaded["sasrec"] - 0.10) < 1e-6

    def test_malformed_json_returns_defaults(self, tmp_path, monkeypatch):
        """Invalid JSON → defaults returned."""
        import backend.models.ensemble_engine as ee_mod

        monkeypatch.setattr(ee_mod, "MODELS_DIR", tmp_path)
        (tmp_path / "ensemble_weights.json").write_text("not valid json {{{")
        engine = _make_engine(tmp_path)
        loaded = engine._load_weights()
        assert abs(loaded["lightgcn"] - 0.60) < 1e-6

    def test_missing_key_returns_defaults(self, tmp_path, monkeypatch):
        """JSON missing 'kan' key → defaults returned."""
        import backend.models.ensemble_engine as ee_mod

        monkeypatch.setattr(ee_mod, "MODELS_DIR", tmp_path)
        weights = {k: 1 / 6 for k in WEIGHT_KEYS if k != "kan"}
        (tmp_path / "ensemble_weights.json").write_text(json.dumps(weights))
        engine = _make_engine(tmp_path)
        loaded = engine._load_weights()
        assert abs(loaded["lightgcn"] - 0.60) < 1e-6

    def test_unnormalised_weights_are_renormalised(self, tmp_path, monkeypatch):
        """Weights summing to 2.0 → returned weights sum to 1.0."""
        import backend.models.ensemble_engine as ee_mod

        monkeypatch.setattr(ee_mod, "MODELS_DIR", tmp_path)
        weights = {k: v * 2 for k, v in DEFAULT_WEIGHTS.items()}
        (tmp_path / "ensemble_weights.json").write_text(json.dumps(weights))
        engine = _make_engine(tmp_path)
        loaded = engine._load_weights()
        assert abs(sum(loaded.values()) - 1.0) < 1e-6

    def test_reload_weights_updates_engine_weights(self, tmp_path, monkeypatch):
        """reload_weights() updates engine._weights in place."""
        import backend.models.ensemble_engine as ee_mod

        monkeypatch.setattr(ee_mod, "MODELS_DIR", tmp_path)
        engine = _make_engine(tmp_path)
        new_weights = dict.fromkeys(WEIGHT_KEYS, 1 / 7)
        (tmp_path / "ensemble_weights.json").write_text(json.dumps(new_weights))
        result = engine.reload_weights()
        assert abs(result["lightgcn"] - 1 / 7) < 1e-5
        assert abs(engine._weights["lightgcn"] - 1 / 7) < 1e-5
