import pandas as pd

from etl.scd import (
    DEFAULT_HIGH_DATE,
    SCD_CURRENT_COL,
    SCD_END_COL,
    SCD_START_COL,
    apply_scd_type2,
)


def test_apply_scd_type2_initial_load():
    incoming = pd.DataFrame(
        [
            {"id": 1, "title": "Inception", "genres": "Sci-Fi", "vote_average": 8.8},
            {"id": 2, "title": "Memento", "genres": "Thriller", "vote_average": 8.4},
        ]
    )

    result = apply_scd_type2(
        existing=None,
        incoming=incoming,
        key_columns=["id"],
        tracked_columns=["title", "genres", "vote_average"],
        effective_ts="2026-05-02T00:00:00",
    )

    assert len(result) == 2
    assert result[SCD_CURRENT_COL].tolist() == [True, True]
    assert set(result[SCD_END_COL]) == {DEFAULT_HIGH_DATE}


def test_apply_scd_type2_no_change_keeps_single_current_version():
    incoming = pd.DataFrame([{"id": 1, "title": "Inception", "genres": "Sci-Fi"}])
    existing = apply_scd_type2(
        existing=None,
        incoming=incoming,
        key_columns=["id"],
        tracked_columns=["title", "genres"],
        effective_ts="2026-05-01T00:00:00",
    )

    result = apply_scd_type2(
        existing=existing,
        incoming=incoming,
        key_columns=["id"],
        tracked_columns=["title", "genres"],
        effective_ts="2026-05-02T00:00:00",
    )

    assert len(result) == 1
    assert result.iloc[0][SCD_START_COL] == "2026-05-01T00:00:00"
    assert bool(result.iloc[0][SCD_CURRENT_COL]) is True


def test_apply_scd_type2_changed_record_expires_old_version():
    day_1 = pd.DataFrame([{"id": 1, "title": "Inception", "genres": "Sci-Fi"}])
    day_2 = pd.DataFrame([{"id": 1, "title": "Inception", "genres": "Sci-Fi, Thriller"}])

    existing = apply_scd_type2(
        existing=None,
        incoming=day_1,
        key_columns=["id"],
        tracked_columns=["title", "genres"],
        effective_ts="2026-05-01T00:00:00",
    )
    result = apply_scd_type2(
        existing=existing,
        incoming=day_2,
        key_columns=["id"],
        tracked_columns=["title", "genres"],
        effective_ts="2026-05-02T00:00:00",
    )

    assert len(result) == 2
    old_version = result[~result[SCD_CURRENT_COL]].iloc[0]
    new_version = result[result[SCD_CURRENT_COL]].iloc[0]

    assert old_version[SCD_END_COL] == "2026-05-02T00:00:00"
    assert new_version["genres"] == "Sci-Fi, Thriller"
    assert new_version[SCD_END_COL] == DEFAULT_HIGH_DATE


def test_apply_scd_type2_new_key_inserts_current_record():
    existing = apply_scd_type2(
        existing=None,
        incoming=pd.DataFrame([{"id": 1, "title": "Inception", "genres": "Sci-Fi"}]),
        key_columns=["id"],
        tracked_columns=["title", "genres"],
        effective_ts="2026-05-01T00:00:00",
    )

    result = apply_scd_type2(
        existing=existing,
        incoming=pd.DataFrame(
            [
                {"id": 1, "title": "Inception", "genres": "Sci-Fi"},
                {"id": 2, "title": "Memento", "genres": "Thriller"},
            ]
        ),
        key_columns=["id"],
        tracked_columns=["title", "genres"],
        effective_ts="2026-05-02T00:00:00",
    )

    assert len(result) == 2
    assert set(result[result[SCD_CURRENT_COL]]["id"]) == {1, 2}
