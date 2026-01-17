"""
System Integration Test.
Simulates a complete data pipeline run and verifies API response.
Run this ensures the components work together: 
Ingest -> Transform -> Index -> Recommender API
"""
import pytest
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
        models_dir = temp_path / "models"
        
        for p in [raw_dir, processed_dir, models_dir]:
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
            models = models_dir
            logs = temp_path / "logs"
            
        # Apply mock to modules
        pandas_etl.paths = MockPaths()
        
        # 2. Run ETL Pipeline (Ingest, Transform, Index)
        metrics = pandas_etl.run_pipeline(raw_data_path=csv_path)
        
        assert metrics["success"] is True
        assert metrics["final_rows"] == 3
        
        # 3. Verify Artifacts
        assert (processed_dir / "movies_transformed.parquet").exists()
        assert (models_dir / "sbert_embeddings.npy").exists()
        assert (models_dir / "faiss.index").exists()
        
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
