import pandas as pd

from etl.lakehouse import write_movie_scd_snapshot, write_versioned_snapshot
from scripts.inspect_lakehouse import inspect_lakehouse, summarize_scd_table, summarize_versioned_table


def test_summarize_versioned_table_reports_latest_manifest(tmp_path):
    gold_base = tmp_path / "gold"
    first = pd.DataFrame(
        {
            "id": [1],
            "title": ["Avatar"],
            "overview": ["Pandora"],
            "tags": ["avatar pandora"],
        }
    )
    second = pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Avatar", "Inception"],
            "overview": ["Pandora", "Dreams"],
            "tags": ["avatar pandora", "inception dreams"],
        }
    )

    write_versioned_snapshot(first, gold_base, "movies_features", "run-001", "2026-05-01")
    write_versioned_snapshot(second, gold_base, "movies_features", "run-002", "2026-05-02")

    summary = summarize_versioned_table(gold_base, "movies_features")

    assert summary["status"] == "ready"
    assert summary["version_count"] == 2
    assert summary["latest"]["run_id"] == "run-002"
    assert summary["latest"]["row_count"] == 2
    assert len(summary["versions"]) == 2


def test_summarize_scd_table_reports_current_and_changed_rows(tmp_path):
    gold_base = tmp_path / "gold"
    day_1 = pd.DataFrame(
        [
            {"id": 1, "title": "Avatar", "overview": "Original", "genres": "Action"},
            {"id": 2, "title": "Inception", "overview": "Dreams", "genres": "Sci-Fi"},
        ]
    )
    day_2 = pd.DataFrame(
        [
            {"id": 1, "title": "Avatar", "overview": "Updated", "genres": "Action"},
            {"id": 2, "title": "Inception", "overview": "Dreams", "genres": "Sci-Fi"},
            {"id": 3, "title": "Interstellar", "overview": "Space", "genres": "Sci-Fi"},
        ]
    )

    write_movie_scd_snapshot(day_1, gold_base, "20260501T010000Z", "2026-05-01")
    write_movie_scd_snapshot(day_2, gold_base, "20260502T010000Z", "2026-05-02")

    summary = summarize_scd_table(
        gold_base,
        as_of_ts="2026-05-02T12:00:00Z",
        compare_from="2026-05-01T12:00:00Z",
        compare_to="2026-05-02T12:00:00Z",
    )

    assert summary["status"] == "ready"
    assert summary["scd"]["current_rows"] == 3
    assert summary["scd"]["historical_versions"] == 1
    assert summary["scd"]["business_keys"] == 3
    assert summary["scd"]["as_of"]["active_rows"] == 3
    assert summary["scd"]["comparison"]["changed_count"] == 1
    assert summary["scd"]["comparison"]["new_count"] == 1


def test_inspect_lakehouse_marks_partial_when_only_gold_exists(tmp_path):
    gold_base = tmp_path / "gold"
    features = pd.DataFrame(
        {
            "id": [1],
            "title": ["Avatar"],
            "overview": ["Pandora"],
            "tags": ["avatar pandora"],
        }
    )

    write_versioned_snapshot(features, gold_base, "movies_features", "run-001", "2026-05-01")
    write_movie_scd_snapshot(features, gold_base, "20260501T010000Z", "2026-05-01")

    report = inspect_lakehouse(
        base_paths={
            "bronze": tmp_path / "bronze",
            "silver": tmp_path / "silver",
            "gold": gold_base,
        }
    )

    assert report["status"] == "partial"
    assert report["ready_table_count"] == 2
    assert report["tables"]["gold.movies_features"]["status"] == "ready"
    assert report["tables"]["gold.dim_movie_scd"]["scd"]["current_rows"] == 1
