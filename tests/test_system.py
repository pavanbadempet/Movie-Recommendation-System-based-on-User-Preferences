"""
System Integration Test.
Simulates a complete data pipeline run and verifies API response.
Run this ensures the components work together: 
Ingest -> Transform -> Index -> Recommender API
"""
import pytest
import json
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import shutil
import os

from etl import pandas_etl
from backend.recommender import Recommender

def test_full_system_flow():
    """
    End-to-End System Test:
    1. Create dummy CSV data
    2. Run Pandas ETL pipeline
    3. Verify artifacts (parquet, embeddings, index)
    4. Initialize Recommender with these artifacts
    5. Verify recommendations are returned
    """
    # Manually create so we can control cleanup errors
    temp_dir = tempfile.mkdtemp()
    try:
        temp_path = Path(temp_dir)
        
        # Setup mock paths structure
        raw_dir = temp_path / "data" / "raw"
        processed_dir = temp_path / "data" / "processed"
        bronze_dir = temp_path / "data" / "bronze"
        silver_dir = temp_path / "data" / "silver"
        gold_dir = temp_path / "data" / "gold"
        quality_dir = temp_path / "data" / "quality"
        manifest_dir = temp_path / "data" / "manifests"
        models_dir = temp_path / "models"
        
        for p in [raw_dir, processed_dir, bronze_dir, silver_dir, gold_dir, quality_dir, manifest_dir, models_dir]:
            p.mkdir(parents=True)
            
        # 1. Create Dummy Data
        csv_path = raw_dir / "TMDB_all_movies.csv"
        df_raw = pd.DataFrame({
            "id": [1, 2, 3],
            "title": ["Matrix", "Inception", "Interstellar"],
            "overview": ["Red pill blue pill", "Dreams within dreams", "Space travel data"],
            "vote_count": [1000, 2000, 1500],  # Above 50 threshold
            "genres": ["[{'name': 'Sci-Fi'}]", "[{'name': 'Sci-Fi'}]", "[{'name': 'Sci-Fi'}]"],
            "vote_average": [8.7, 8.8, 8.6],
            "release_date": ["1999-03-31", "2010-07-16", "2014-11-07"],
            "poster_path": ["/path.jpg", "/path.jpg", "/path.jpg"],
        })
        df_raw.to_csv(csv_path, index=False)
        
        # Mock paths in modules
        import etl.config
        
        # Monkeypatch Config Paths
        class MockPaths:
            raw_data = raw_dir
            processed_data = processed_dir
            bronze_data = bronze_dir
            silver_data = silver_dir
            gold_data = gold_dir
            quality_reports = quality_dir
            manifests = manifest_dir
            models = models_dir
            logs = temp_path / "logs"
            
        # Apply mock to modules
        pandas_etl.paths = MockPaths()
        
        # 2. Run ETL Pipeline (Ingest, Transform, Index)
        metrics = pandas_etl.run_pipeline(raw_data_path=csv_path, run_id="test-run", run_date="2026-05-02")
        
        assert metrics["success"] is True
        assert metrics["final_rows"] == 3
        assert metrics["quality"]["total_rows"] == 3
        assert metrics["quality"]["duplicate_ids"] == 0
        assert metrics["quality_gates"]["silver"]["rows"] == 3
        assert metrics["quality_gates"]["gold"]["vector_rows"] == 3
        assert metrics["quality_gates"]["serving"]["index_size"] == 3
        assert metrics["artifacts"]["movies"]["exists"] is True
        assert metrics["time_travel_artifacts"]["movies_raw"]["row_count"] == 3
        assert metrics["time_travel_artifacts"]["movies_curated"]["row_count"] == 3
        assert metrics["time_travel_artifacts"]["movies_features"]["row_count"] == 3
        
        # 3. Verify Artifacts
        assert (processed_dir / "movies_transformed.parquet").exists()
        assert (models_dir / "sbert_embeddings.npy").exists()
        assert (models_dir / "faiss.index").exists()
        assert (quality_dir / "test-run.json").exists()
        assert (manifest_dir / "test-run.json").exists()
        assert (bronze_dir / "run_id=test-run" / "movies_raw.parquet").exists()
        assert (silver_dir / "run_id=test-run" / "movies_curated.parquet").exists()
        assert (gold_dir / "run_id=test-run" / "movies_features.parquet").exists()
        assert (bronze_dir / "movies_raw" / "run_date=2026-05-02" / "run_id=test-run" / "data.parquet").exists()
        assert (silver_dir / "movies_curated" / "run_date=2026-05-02" / "run_id=test-run" / "_manifest.json").exists()
        assert (gold_dir / "movies_features" / "_latest.json").exists()

        manifest = json.loads((manifest_dir / "test-run.json").read_text(encoding="utf-8"))
        assert manifest["run_id"] == "test-run"
        assert manifest["row_counts"]["raw_rows"] == 3
        assert manifest["row_counts"]["serving_rows"] == 3
        assert manifest["artifacts"]["faiss_index"]["exists"] is True
        assert manifest["stage_artifacts"]["bronze"]["exists"] is True
        assert manifest["stage_artifacts"]["silver"]["exists"] is True
        assert manifest["stage_artifacts"]["gold"]["exists"] is True
        assert manifest["quality_gates"]["serving"]["index_size"] == 3
        assert manifest["time_travel_artifacts"]["movies_features"]["row_count"] == 3

        from etl.lakehouse import load_table_version

        gold_snapshot = load_table_version(gold_dir, "movies_features", as_of_date="2026-05-02")
        assert len(gold_snapshot) == 3
        
        # 4. Update Recommender to use these paths and Test
        import backend.recommender
        backend.recommender.MODELS_DIR = models_dir
        backend.recommender.DATA_DIR = processed_dir
        
        rec = Recommender().load()
        assert len(rec.movies) == 3
        
        # 5. Get Recommendations
        results = rec.recommend_by_title("Matrix", n=2)
        assert len(results) == 2
        # Should recommend Inception or Interstellar (same genre)
        assert results[0]["title"] in ["Inception", "Interstellar"]
        
        print("\n✅ System Flow Verified: ETL -> Artifacts -> Recommender -> Output")
        
    finally:
        # Cleanup with error ignore for Windows file locks
        shutil.rmtree(temp_dir, ignore_errors=True)
