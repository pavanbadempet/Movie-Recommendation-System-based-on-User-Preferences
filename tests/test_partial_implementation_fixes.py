import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch


def test_load_offline_metrics_requires_a_real_report(tmp_path):
    from backend.api.evaluation_routes import load_offline_metrics

    with pytest.raises(FileNotFoundError, match="offline evaluation report"):
        load_offline_metrics([tmp_path / "missing.json"])


def test_load_offline_metrics_returns_report_with_provenance(tmp_path):
    from backend.api.evaluation_routes import load_offline_metrics

    report_path = tmp_path / "offline_eval_report.json"
    report_path.write_text(json.dumps({"ndcg_at_10": 0.25}), encoding="utf-8")

    report = load_offline_metrics([report_path])

    assert report["ndcg_at_10"] == 0.25
    assert report["provenance"]["source"] == "offline_evaluation_report"
    assert report["provenance"]["report_path"] == str(report_path.resolve())


def test_fairness_auditor_requires_real_catalog_data(tmp_path):
    from scripts.fairness_audit import FairnessAuditor

    with pytest.raises(FileNotFoundError, match=r"movies_transformed\.parquet"):
        FairnessAuditor(data_dir=tmp_path)


def test_fairness_report_is_diagnostic_and_does_not_invent_privacy_evidence():
    from scripts.fairness_audit import FairnessAuditor

    auditor = FairnessAuditor.__new__(FairnessAuditor)
    report = auditor.generate_report(
        recommendation_slates=[[1, 2], [2, 3]],
        user_history_genres={"Action": 2, "Comedy": 1},
        recommended_genres={"Action": 1, "Comedy": 2},
    )

    assert "not a compliance certification" in report
    assert "EU AI Act" not in report
    assert "Differential Privacy Evidence" in report
    assert "NOT EVALUATED" in report
    assert "✅ ACTIVE" not in report


def test_hybrid_training_requires_gold_embeddings(tmp_path):
    from scripts.train_hybrid_architecture import load_pyspark_embeddings

    with pytest.raises(FileNotFoundError, match="Gold embeddings"):
        load_pyspark_embeddings(gold_dir=tmp_path)


def test_hybrid_training_maps_real_positive_interactions(tmp_path):
    from scripts.train_hybrid_architecture import load_positive_interactions, load_pyspark_embeddings

    gold_dir = tmp_path / "gold"
    user_dir = gold_dir / "model_user_embeddings"
    item_dir = gold_dir / "model_item_embeddings"
    user_dir.mkdir(parents=True)
    item_dir.mkdir(parents=True)
    pd.DataFrame(
        {"id": [10, 20], "features": [np.ones(4, dtype=np.float32), np.full(4, 2.0, dtype=np.float32)]}
    ).to_parquet(user_dir / "part.parquet")
    pd.DataFrame(
        {"id": [100, 200], "features": [np.ones(4, dtype=np.float32), np.full(4, 3.0, dtype=np.float32)]}
    ).to_parquet(item_dir / "part.parquet")
    ratings_path = tmp_path / "ratings.parquet"
    pd.DataFrame(
        {
            "userId": [10, 20, 999],
            "movieId": [200, 100, 100],
            "rating": [4.5, 2.0, 5.0],
        }
    ).to_parquet(ratings_path)

    bundle = load_pyspark_embeddings(gold_dir=gold_dir)
    users, items = load_positive_interactions(ratings_path, bundle, minimum_rating=3.5)

    assert torch.equal(users, torch.tensor([0]))
    assert torch.equal(items, torch.tensor([1]))
    assert bundle.user_tensor.shape == (2, 4)
    assert bundle.item_tensor.shape == (2, 4)


def test_compact_rl_uses_real_movie_embeddings_for_actions(monkeypatch):
    from scripts import train_rl_policy_compact as compact

    events = [
        {"event_type": "rating", "user_id": "u1", "movie_id": 10, "rating": 5.0},
        {"event_type": "rating", "user_id": "u1", "movie_id": 20, "rating": 1.0},
    ]
    monkeypatch.setattr(compact, "iter_events", lambda: iter(events))
    item_embeddings = {
        10: np.array([3.0] + [0.0] * 15, dtype=np.float32),
        20: np.array([0.0, 4.0] + [0.0] * 14, dtype=np.float32),
    }

    states, actions, rewards = compact.load_training_data(
        min_interactions=2,
        item_embeddings=item_embeddings,
    )

    assert states.shape == (2, 20)
    assert np.allclose(actions[0], np.array([1.0] + [0.0] * 15))
    assert np.allclose(actions[1], np.array([0.0, -1.0] + [0.0] * 14))
    assert rewards[:, 0].tolist() == [1.0, -0.5]


def test_compact_rl_refuses_insufficient_real_training_data(monkeypatch):
    from scripts import train_rl_policy_compact as compact

    monkeypatch.setattr(compact, "iter_events", lambda: iter([]))
    with pytest.raises(RuntimeError, match="real reward-bearing interactions"):
        compact.load_training_data(min_interactions=1, item_embeddings={})


def test_legacy_rl_entrypoint_uses_serving_dimensions():
    from scripts import train_rl_policy

    assert train_rl_policy.STATE_DIM == 20
    assert train_rl_policy.ACTION_DIM == 16


def test_legacy_rl_entrypoint_runs_as_a_script():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "train_rl_policy.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Train compact RL policy" in result.stdout


def test_monitoring_snapshot_uses_backend_telemetry_without_synthetic_values():
    from frontend.monitoring import build_monitoring_snapshot

    payloads = {
        "/health": {"status": "healthy", "movie_count": 100, "serving_tier": "tier2"},
        "/v1/platform/status": {
            "status": "ready",
            "event_store": {"mode": "postgres", "durable": True, "total_events": 50},
        },
        "/v1/artifacts/health": {"status": "ready", "row_counts": {"movies": 100}},
        "/v1/events/features": {"event_type_counts": {"view": 20, "rating": 5}, "top_searches": []},
        "/v1/events/recommendation-analytics": {
            "impression_count": 10,
            "click_count": 2,
            "click_through_rate": 0.2,
        },
    }

    def api_get(path, params=None, timeout=15):
        return payloads[path]

    snapshot = build_monitoring_snapshot(api_get)

    assert snapshot["total_events"] == 50
    assert snapshot["event_type_counts"] == {"view": 20, "rating": 5}
    assert snapshot["click_through_rate"] == 0.2
    assert snapshot["durable"] is True
    assert snapshot["telemetry_source"] == "backend_api"


def test_repository_documentation_links_for_contributors_and_frontend_deploy_exist():
    root = Path(__file__).resolve().parents[1]

    for relative_path in (
        "AGENTS.md",
        "CODE_OF_CONDUCT.md",
        ".github/workflows/frontend-pages.yml",
    ):
        assert (root / relative_path).is_file(), relative_path


def test_project_rating_document_does_not_present_planned_features_as_complete():
    root = Path(__file__).resolve().parents[1]
    rating = (root / "docs" / "PROJECT_RATING_10_10.md").read_text(encoding="utf-8")

    assert "Perfect Score (10/10)" not in rating
    assert "READY FOR PRODUCTION DEPLOYMENT" not in rating
    assert "shadcn/ui" not in rating
    assert "Real-time updates with WebSocket" not in rating
    assert "Advanced state management with React Query" not in rating
    assert "verified status" in rating.lower()
