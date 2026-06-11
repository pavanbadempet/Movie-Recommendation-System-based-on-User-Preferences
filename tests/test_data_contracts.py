"""Tests for machine-readable dataset contracts."""

import pandas as pd
import pytest

from etl.data_contracts import load_contract, validate_dataframe_contract


@pytest.fixture
def silver_movies_df():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Movie A", "Movie B"],
            "overview": ["Story A", "Story B"],
            "metadata_completeness": [0.9, 0.6],
            "content_quality_score": [0.8, 0.5],
            "quality_bucket": ["premium", "standard"],
            "searchable": [True, True],
            "recommendable": [True, True],
            "is_adult_content": [False, False],
            "public_demo_eligible": [True, True],
        }
    )


def test_load_contract_reads_machine_readable_schema():
    contract = load_contract("silver_movies")

    assert contract["name"] == "silver_movies"
    assert contract["primary_key"] == ["id"]
    assert "quality_bucket" in contract["columns"]


def test_validate_dataframe_contract_accepts_valid_silver_df(silver_movies_df):
    result = validate_dataframe_contract(silver_movies_df, "silver_movies", stage="silver")

    assert result["contract_name"] == "silver_movies"
    assert result["rows"] == 2


def test_validate_dataframe_contract_rejects_missing_required_columns(silver_movies_df):
    invalid_df = silver_movies_df.drop(columns=["title"])

    with pytest.raises(ValueError, match="missing contract columns"):
        validate_dataframe_contract(invalid_df, "silver_movies", stage="silver")


def test_validate_dataframe_contract_rejects_duplicate_primary_keys(silver_movies_df):
    invalid_df = silver_movies_df.copy()
    invalid_df.loc[1, "id"] = 1

    with pytest.raises(ValueError, match="duplicate rows found"):
        validate_dataframe_contract(invalid_df, "silver_movies", stage="silver")


def test_validate_dataframe_contract_rejects_invalid_enum_values(silver_movies_df):
    invalid_df = silver_movies_df.copy()
    invalid_df.loc[1, "quality_bucket"] = "elite"

    with pytest.raises(ValueError, match="invalid values"):
        validate_dataframe_contract(invalid_df, "silver_movies", stage="silver")


def test_assert_batch_invariants_uses_contract_validation():
    from etl.pandas_etl import assert_batch_invariants

    invalid_df = pd.DataFrame(
        {
            "id": [1],
            "title": ["Movie A"],
            "overview": ["Story A"],
            "metadata_completeness": [0.9],
            "content_quality_score": [0.8],
            "quality_bucket": ["elite"],
            "searchable": [True],
            "recommendable": [True],
            "is_adult_content": [False],
            "public_demo_eligible": [True],
        }
    )

    with pytest.raises(ValueError, match="invalid values"):
        assert_batch_invariants(invalid_df, stage="silver")
