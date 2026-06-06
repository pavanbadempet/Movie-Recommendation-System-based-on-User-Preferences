"""Unit tests for the MLOps statistical drift and lineage registration engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.serving.mlops_engine import MLOpsEngine


def test_ks_drift_detection():
    """MLOpsEngine should compute KS statistics and detect features that have drifted."""
    engine = MLOpsEngine(Path("dummy/path"))

    # Create baseline from a normal distribution
    np.random.seed(42)
    base_popularity = np.random.normal(loc=10.0, scale=2.0, size=100)
    baseline_df = pd.DataFrame({"popularity": base_popularity})

    # Case 1: Same distribution (no drift)
    same_popularity = np.random.normal(loc=10.0, scale=2.0, size=100)
    same_df = pd.DataFrame({"popularity": same_popularity})

    # Case 2: Shifted distribution (drift)
    shifted_popularity = np.random.normal(loc=15.0, scale=2.0, size=100)
    shifted_df = pd.DataFrame({"popularity": shifted_popularity})

    # Run KS checks
    same_results = engine.compute_ks_drift(baseline_df, same_df, ["popularity"])
    shifted_results = engine.compute_ks_drift(baseline_df, shifted_df, ["popularity"])

    # Assert p-value for same distribution is high (fail to reject null hypothesis)
    assert same_results["popularity"]["p_value"] > 0.05
    assert same_results["popularity"]["statistic"] < 0.25

    # Assert p-value for shifted distribution is extremely small (reject null hypothesis)
    assert shifted_results["popularity"]["p_value"] < 0.01
    assert shifted_results["popularity"]["statistic"] > 0.5


def test_embedding_drift_detection():
    """MLOpsEngine should calculate embedding centroid shifts correctly."""
    engine = MLOpsEngine(Path("dummy/path"))

    # Base embeddings
    np.random.seed(42)
    base_embeds = np.random.rand(10, 8).astype(np.float32)

    # Identical embeddings (zero shift)
    zero_results = engine.compute_embedding_drift(base_embeds, base_embeds)
    assert abs(zero_results["mean_alignment_shift"]) < 1e-5
    assert abs(zero_results["centroid_cosine_similarity"] - 1.0) < 1e-5

    # Shifted embeddings (centroid is moved)
    shifted_embeds = base_embeds + np.ones((10, 8), dtype=np.float32) * 5.0
    shifted_results = engine.compute_embedding_drift(base_embeds, shifted_embeds)
    assert shifted_results["mean_alignment_shift"] > 0.0
    assert shifted_results["centroid_cosine_similarity"] < 1.0


def test_validate_and_register_run(tmp_path):
    """Lineage log is correctly recorded, hashes computed, and promotions/drift flagged."""
    registry_file = tmp_path / "lineage_registry.json"
    engine = MLOpsEngine(registry_file)

    # Base dataframes
    np.random.seed(42)
    baseline_df = pd.DataFrame({
        "popularity": np.random.normal(10.0, 1.0, 50),
        "vote_count": np.random.normal(100.0, 10.0, 50),
        "content_quality_score": np.random.normal(0.8, 0.05, 50)
    })
    baseline_embeds = np.random.normal(0.0, 1.0, (50, 8)).astype(np.float32)

    # New matching data (promoted run) - close deterministic offset to prevent false alarms
    new_df = baseline_df.copy()
    new_df["popularity"] += 0.01
    new_df["vote_count"] += 1
    new_df["content_quality_score"] += 0.001
    new_embeds = baseline_embeds + np.random.normal(0.0, 0.01, (50, 8)).astype(np.float32)

    dummy_tq = tmp_path / "dummy_turbovec.tq"
    dummy_tq.write_bytes(b"dummy_index_bytes")

    # Case 1: Healthy Run
    report_healthy = engine.validate_and_register_run(
        run_id="run-001",
        new_df=new_df,
        new_embeds=new_embeds,
        turbovec_path=dummy_tq,
        baseline_df=baseline_df,
        baseline_embeds=baseline_embeds,
    )

    assert report_healthy["promotion_status"] == "promoted"
    assert report_healthy["drift_analysis"]["drift_detected"] is False
    assert report_healthy["hashes"]["turbovec_index"] != "none"

    # Case 2: Drifted Run
    drift_df = pd.DataFrame({
        "popularity": np.random.normal(25.0, 1.0, 50),
        "vote_count": np.random.normal(1000.0, 10.0, 50),
        "content_quality_score": np.random.normal(0.1, 0.05, 50)
    })
    report_drifted = engine.validate_and_register_run(
        run_id="run-002",
        new_df=drift_df,
        new_embeds=new_embeds,
        turbovec_path=dummy_tq,
        baseline_df=baseline_df,
        baseline_embeds=baseline_embeds,
    )

    assert report_drifted["promotion_status"] == "needs_review"
    assert report_drifted["drift_analysis"]["drift_detected"] is True
    assert len(report_drifted["drift_analysis"]["drift_reasons"]) > 0

    # Load registry history file and assert records exist
    assert registry_file.exists()
    history = json.loads(registry_file.read_text(encoding="utf-8"))
    assert len(history) == 2
    assert history[0]["run_id"] == "run-001"
    assert history[1]["run_id"] == "run-002"
