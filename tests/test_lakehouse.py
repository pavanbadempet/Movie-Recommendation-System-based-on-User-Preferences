"""
Tests for lakehouse data models and local time travel helpers.
"""

import pandas as pd
import pytest

from etl.lakehouse import (
    as_of_scd,
    build_movie_scd_snapshot,
    compare_scd_as_of,
    list_table_versions,
    load_table_version,
    validate_table_contract,
    write_movie_scd_snapshot,
    write_versioned_snapshot,
)


def test_write_versioned_snapshot_and_load_as_of_date(tmp_path):
    first = pd.DataFrame({
        "id": [1, 2],
        "title": ["Matrix", "Inception"],
        "overview": ["Red pill", "Dreams"],
        "tags": ["matrix sci-fi", "inception dreams"],
    })
    second = pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["Matrix", "Inception", "Interstellar"],
        "overview": ["Red pill", "Dreams", "Space"],
        "tags": ["matrix sci-fi", "inception dreams", "interstellar space"],
    })

    first_manifest = write_versioned_snapshot(
        first,
        tmp_path,
        table_name="movies_features",
        run_id="run-001",
        run_date="2026-05-01",
    )
    second_manifest = write_versioned_snapshot(
        second,
        tmp_path,
        table_name="movies_features",
        run_id="run-002",
        run_date="2026-05-02",
    )

    versions = list_table_versions(tmp_path, "movies_features")
    historical = load_table_version(tmp_path, "movies_features", as_of_date="2026-05-01")
    latest = load_table_version(tmp_path, "movies_features")

    assert len(versions) == 2
    assert first_manifest["row_count"] == 2
    assert second_manifest["row_count"] == 3
    assert len(historical) == 2
    assert len(latest) == 3
    assert (tmp_path / "movies_features" / "_latest.json").exists()


def test_validate_table_contract_rejects_missing_required_columns():
    df = pd.DataFrame({"id": [1], "title": ["Matrix"]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_table_contract(df, "movies_curated")


def test_bronze_raw_contract_allows_untrusted_source_rows():
    df = pd.DataFrame({"id": [None], "title": [None], "overview": [None]})

    contract = validate_table_contract(df, "movies_raw")

    assert contract["rows"] == 1
    assert contract["required_columns"] == []


def test_scd_as_of_time_travel_and_comparison():
    history = pd.DataFrame({
        "id": [1, 1, 2, 3],
        "title": ["Matrix", "Matrix", "Inception", "Interstellar"],
        "overview": ["Old overview", "New overview", "Dreams", "Space"],
        "record_hash": ["old", "new", "same", "added"],
        "effective_start_at": [
            "2026-05-01T00:00:00Z",
            "2026-05-02T00:00:00Z",
            "2026-05-01T00:00:00Z",
            "2026-05-02T00:00:00Z",
        ],
        "effective_end_at": [
            "2026-05-02T00:00:00Z",
            "9999-12-31T00:00:00",
            "9999-12-31T00:00:00",
            "9999-12-31T00:00:00",
        ],
        "is_current": [False, True, True, True],
    })

    before = as_of_scd(history, "2026-05-01T12:00:00Z")
    after = as_of_scd(history, "2026-05-02T12:00:00Z")
    diff = compare_scd_as_of(history, "2026-05-01T12:00:00Z", "2026-05-02T12:00:00Z")

    assert set(before["id"]) == {1, 2}
    assert before[before["id"] == 1].iloc[0]["overview"] == "Old overview"
    assert set(after["id"]) == {1, 2, 3}
    assert after[after["id"] == 1].iloc[0]["overview"] == "New overview"
    assert diff["changed_count"] == 1
    assert diff["new_count"] == 1
    assert diff["removed_count"] == 0


def test_build_movie_scd_snapshot_dedupes_and_tracks_changes():
    incoming = pd.DataFrame(
        [
            {"id": 1, "title": "Avatar", "overview": "Pandora", "genres": "Action", "vote_count": 10},
            {"id": 1, "title": "Avatar", "overview": "Pandora", "genres": "Action", "vote_count": 99},
            {"id": 2, "title": "Inception", "overview": "Dreams", "genres": "Sci-Fi", "vote_count": 50},
        ]
    )

    history = build_movie_scd_snapshot(incoming, effective_ts="2026-05-01T01:00:00Z")

    assert len(history) == 2
    assert set(history["id"]) == {1, 2}
    assert history[history["id"] == 1].iloc[0]["vote_count"] == 99
    assert history["is_current"].tolist() == [True, True]


def test_write_movie_scd_snapshot_and_time_travel(tmp_path):
    day_1 = pd.DataFrame(
        [
            {"id": 1, "title": "Avatar", "overview": "Original overview", "genres": "Action, Sci-Fi"},
            {"id": 2, "title": "Inception", "overview": "Dream heist", "genres": "Sci-Fi"},
        ]
    )
    day_2 = pd.DataFrame(
        [
            {"id": 1, "title": "Avatar", "overview": "Updated overview", "genres": "Action, Sci-Fi"},
            {"id": 2, "title": "Inception", "overview": "Dream heist", "genres": "Sci-Fi"},
            {"id": 3, "title": "Interstellar", "overview": "Space and time", "genres": "Sci-Fi"},
        ]
    )

    first = write_movie_scd_snapshot(
        incoming_df=day_1,
        base_path=tmp_path,
        run_id="20260501T010000Z",
        run_date="2026-05-01",
    )
    second = write_movie_scd_snapshot(
        incoming_df=day_2,
        base_path=tmp_path,
        run_id="20260502T010000Z",
        run_date="2026-05-02",
    )

    history = load_table_version(tmp_path, "dim_movie_scd")
    before = as_of_scd(history, "2026-05-01T12:00:00Z")
    after = as_of_scd(history, "2026-05-02T12:00:00Z")
    diff = compare_scd_as_of(history, "2026-05-01T12:00:00Z", "2026-05-02T12:00:00Z")

    assert first["current_rows"] == 2
    assert second["current_rows"] == 3
    assert second["total_versions"] == 4
    assert second["manifest"]["row_count"] == 4
    assert set(before["id"]) == {1, 2}
    assert before[before["id"] == 1].iloc[0]["overview"] == "Original overview"
    assert set(after["id"]) == {1, 2, 3}
    assert after[after["id"] == 1].iloc[0]["overview"] == "Updated overview"
    assert diff["changed_count"] == 1
    assert diff["new_count"] == 1
