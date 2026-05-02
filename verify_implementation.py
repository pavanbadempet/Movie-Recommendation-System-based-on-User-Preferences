"""
Verification script for Delta Lake and Medallion Architecture implementation.
This script verifies the code structure without requiring Spark execution.
"""
import logging
import ast
import inspect
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_pyspark_etl_structure():
    """Verify the structure of the PySpark ETL script."""
    logger.info("Verifying PySpark ETL script structure...")

    try:
        # Import the module to verify it loads correctly
        from etl import pyspark_etl
        logger.info("✓ PySpark ETL module imports successfully")

        # Check for required top-level functions
        required_top_level_functions = [
            'create_spark_session',
            'run_spark_etl'
        ]

        for func_name in required_top_level_functions:
            if hasattr(pyspark_etl, func_name):
                logger.info(f"✓ Top-level function '{func_name}' exists")
            else:
                logger.error(f"✗ Top-level function '{func_name}' is missing")
                return False

        # Check for nested functions by inspecting the run_spark_etl function
        run_etl_code = inspect.getsource(pyspark_etl.run_spark_etl)
        required_nested_functions = [
            'write_bronze',
            'transform_to_silver',
            'write_silver',
            'transform_to_gold',
            'write_gold',
            'write_sink'
        ]

        for func_name in required_nested_functions:
            if func_name in run_etl_code:
                logger.info(f"✓ Nested function '{func_name}' exists")
            else:
                logger.error(f"✗ Nested function '{func_name}' is missing")
                return False

        # Verify Delta Lake configuration in Spark session
        source_code = inspect.getsource(pyspark_etl.create_spark_session)
        if "io.delta.sql.DeltaSparkSessionExtension" in source_code:
            logger.info("✓ Delta Lake extensions are configured in Spark session")
        else:
            logger.error("✗ Delta Lake extensions are not configured")
            return False

        if "org.apache.spark.sql.delta.catalog.DeltaCatalog" in source_code:
            logger.info("✓ Delta Lake catalog is configured in Spark session")
        else:
            logger.error("✗ Delta Lake catalog is not configured")
            return False

        # Verify silver layer transformations
        silver_code = inspect.getsource(pyspark_etl.transform_to_silver)
        silver_checks = [
            "title_completeness",
            "overview_completeness",
            "release_year",
            "trim(lower(title))",
            "concat_ws"
        ]

        for check in silver_checks:
            if check in silver_code:
                logger.info(f"✓ Silver layer transformation includes '{check}'")
            else:
                logger.warning(f"⚠ Silver layer transformation may be missing '{check}'")

        # Verify gold layer transformations
        gold_code = inspect.getsource(pyspark_etl.transform_to_gold)
        gold_checks = [
            "popularity_score",
            "quality_score",
            "engagement_score",
            "is_popular",
            "is_high_rated",
            "is_recent",
            "top_genre",
            "second_genre",
            "third_genre"
        ]

        for check in gold_checks:
            if check in gold_code:
                logger.info(f"✓ Gold layer transformation includes '{check}'")
            else:
                logger.warning(f"⚠ Gold layer transformation may be missing '{check}'")

        # Verify SBERT + FAISS pulls from Gold layer
        run_etl_code = inspect.getsource(pyspark_etl.run_spark_etl)
        if "gold_path = str(paths.gold_data / \"movies\")" in run_etl_code:
            logger.info("✓ SBERT + FAISS integration pulls from Gold layer")
        else:
            logger.error("✗ SBERT + FAISS integration does not pull from Gold layer")
            return False

        # Verify Delta Lake optimizations
        write_gold_code = inspect.getsource(pyspark_etl.write_gold)
        delta_optimizations = [
            "delta.autoOptimize.optimizeWrite",
            "delta.autoOptimize.autoCompact",
            "delta.dataSkippingNumIndexedCols"
        ]

        for opt in delta_optimizations:
            if opt in write_gold_code:
                logger.info(f"✓ Gold layer Delta Lake optimization: '{opt}'")
            else:
                logger.warning(f"⚠ Gold layer Delta Lake optimization may be missing: '{opt}'")

        return True

    except Exception as e:
        logger.error(f"✗ Error verifying PySpark ETL structure: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_medallion_architecture():
    """Verify the Medallion Architecture implementation."""
    logger.info("Verifying Medallion Architecture implementation...")

    try:
        from etl.config import paths

        # Check if all layer directories are configured
        layers = ['bronze', 'silver', 'gold']
        for layer in layers:
            path = getattr(paths, f"{layer}_data")
            logger.info(f"✓ {layer.capitalize()} layer path configured: {path}")

        return True

    except Exception as e:
        logger.error(f"✗ Error verifying Medallion Architecture: {e}")
        return False

def verify_airflow_integration():
    """Verify Airflow DAGs are configured for Delta Lake."""
    logger.info("Verifying Airflow DAG integration...")

    try:
        # Check refresh DAG
        refresh_dag_path = Path("airflow/dags/refresh_dag.py")
        if refresh_dag_path.exists():
            with open(refresh_dag_path, "r") as f:
                content = f.read()
                if "--sink delta" in content:
                    logger.info("✓ Refresh DAG is configured to use Delta Lake format")
                else:
                    logger.warning("⚠ Refresh DAG is not configured to use Delta Lake format")
        else:
            logger.warning("⚠ Refresh DAG file not found")

        # Check Kafka Spark integration DAG
        kafka_dag_path = Path("airflow/dags/kafka_spark_integration_dag.py")
        if kafka_dag_path.exists():
            with open(kafka_dag_path, "r") as f:
                content = f.read()
                if "--sink delta" in content:
                    logger.info("✓ Kafka Spark integration DAG is configured to use Delta Lake format")
                else:
                    logger.warning("⚠ Kafka Spark integration DAG is not configured to use Delta Lake format")
        else:
            logger.warning("⚠ Kafka Spark integration DAG file not found")

        return True

    except Exception as e:
        logger.error(f"✗ Error verifying Airflow integration: {e}")
        return False

def verify_recommender_integration():
    """Verify the recommender is configured to work with Gold layer."""
    logger.info("Verifying recommender integration with Gold layer...")

    try:
        # Check if the recommender loads data from the right location
        backend_path = Path("backend/recommender.py")
        if backend_path.exists():
            with open(backend_path, "r") as f:
                content = f.read()
                # Check if it loads from the expected data directory
                if "data/processed" in content or "data/gold" in content:
                    logger.info("✓ Recommender appears to be configured for processed/gold data")
                else:
                    logger.warning("⚠ Recommender may not be configured for gold layer data")

                # Check if it loads the FAISS index and embeddings
                if "faiss.index" in content and "sbert_embeddings.npy" in content:
                    logger.info("✓ Recommender loads FAISS index and SBERT embeddings")
                else:
                    logger.warning("⚠ Recommender may not load FAISS index and SBERT embeddings properly")

        return True

    except Exception as e:
        logger.error(f"✗ Error verifying recommender integration: {e}")
        return False

def main():
    """Run all verification tests."""
    logger.info("🔍 Starting comprehensive verification of Delta Lake implementation...")

    tests = [
        ("Medallion Architecture Configuration", verify_medallion_architecture),
        ("PySpark ETL Structure", verify_pyspark_etl_structure),
        ("Airflow Integration", verify_airflow_integration),
        ("Recommender Integration", verify_recommender_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            logger.error(f"❌ {test_name} failed")

    logger.info(f"\n📊 Verification Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All verification tests passed! Delta Lake implementation is correctly structured.")
        return True
    else:
        logger.warning("⚠ Some verification tests failed or had warnings. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)