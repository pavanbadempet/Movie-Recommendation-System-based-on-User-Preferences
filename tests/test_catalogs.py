"""
Tests for customer catalog onboarding.
"""

from backend.data.catalogs import persist_catalog_upload, profile_catalog_csv

SAMPLE_CSV = """id,title,overview,genres,cast,original_language,release_date,vote_average,popularity
1,Arrival,A linguist works with the military to communicate with alien visitors.,Sci-Fi,"Amy Adams, Jeremy Renner",en,2016-11-11,7.6,88.1
2,,Short text,Drama,,en,2020-01-01,6.0,10.0
1,Duplicate,A duplicate catalog row with enough description to profile as valid.,Drama,,en,2021-01-01,6.2,9.0
"""


def test_profile_catalog_csv_flags_quality_issues():
    profile = profile_catalog_csv(
        SAMPLE_CSV,
        tenant_id="acme",
        catalog_id="movies",
        sample_size=2,
    )

    assert profile["tenant_id"] == "acme"
    assert profile["catalog_id"] == "movies"
    assert profile["total_rows_profiled"] == 3
    assert profile["valid_rows"] == 2
    assert profile["invalid_rows"] == 1
    assert profile["missing_title_rows"] == 1
    assert profile["weak_description_rows"] == 1
    assert profile["duplicate_source_content_ids"] == 1
    assert profile["ready_for_ingestion"] is False
    assert "duplicate_source_content_ids" in profile["warnings"]
    assert len(profile["samples"]) == 2


def test_persist_catalog_upload_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_CATALOG_UPLOAD_PATH", str(tmp_path))

    manifest = persist_catalog_upload(
        SAMPLE_CSV,
        tenant_id="acme",
        catalog_id="movies",
        filename="catalog.csv",
    )

    assert manifest["upload_id"]
    assert manifest["profile"]["total_rows_profiled"] == 3
    assert manifest["raw_path"].endswith("raw.csv")
    assert manifest["manifest_path"].endswith("manifest.json")
