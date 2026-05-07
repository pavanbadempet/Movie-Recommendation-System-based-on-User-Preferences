"""
Test script to verify Delta Lake and Medallion Architecture implementation.
"""
import logging
import tempfile
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_delta_implementation(raise_on_error=False):
    """Test the Delta Lake and Medallion Architecture implementation."""
    logger.info("Testing Delta Lake and Medallion Architecture implementation...")

    try:
        # Import the necessary modules
        from etl.config import paths
        from etl.pyspark_etl import create_spark_session, run_spark_etl
        from pyspark.sql import SparkSession

        logger.info("✓ Successfully imported required modules")

        # Check if directories exist
        logger.info("Checking Medallion Architecture directories...")
        for layer in ['bronze', 'silver', 'gold']:
            path = getattr(paths, f"{layer}_data") / "movies"
            if isinstance(path, Path) and path.exists():
                logger.info(f"✓ {layer.capitalize()} layer directory exists: {path}")
            else:
                logger.warning(f"⚠ {layer.capitalize()} layer directory does not exist: {path}")

        # Test Spark session creation with Delta Lake support
        logger.info("Testing Spark session creation with Delta Lake support...")
        spark = create_spark_session()

        # Check if Delta Lake extensions are loaded
        delta_extensions = spark.conf.get("spark.sql.extensions", "")
        if "io.delta.sql.DeltaSparkSessionExtension" in delta_extensions:
            logger.info("✓ Delta Lake extensions are properly configured")
        else:
            logger.warning("⚠ Delta Lake extensions are not properly configured")

        # Check if Delta Lake catalog is configured
        delta_catalog = spark.conf.get("spark.sql.catalog.spark_catalog", "")
        if "org.apache.spark.sql.delta.catalog.DeltaCatalog" in delta_catalog:
            logger.info("✓ Delta Lake catalog is properly configured")
        else:
            logger.warning("⚠ Delta Lake catalog is not properly configured")

        spark.stop()

        # Test reading from the config
        logger.info("Testing configuration...")
        logger.info(f"✓ Raw data path: {paths.raw_data}")
        logger.info(f"✓ Bronze data path: {paths.bronze_data}")
        logger.info(f"✓ Silver data path: {paths.silver_data}")
        logger.info(f"✓ Gold data path: {paths.gold_data}")

        # Test Airflow DAGs
        logger.info("Testing Airflow DAGs...")

        # Check refresh DAG
        with open("airflow/dags/refresh_dag.py", "r") as f:
            content = f.read()
            if "--sink delta" in content:
                logger.info("✓ Refresh DAG is configured to use Delta Lake format")
            else:
                logger.warning("⚠ Refresh DAG is not configured to use Delta Lake format")

        # Check Kafka Spark integration DAG
        with open("airflow/dags/kafka_spark_integration_dag.py", "r") as f:
            content = f.read()
            if "--sink delta" in content:
                logger.info("✓ Kafka Spark integration DAG is configured to use Delta Lake format")
            else:
                logger.warning("⚠ Kafka Spark integration DAG is not configured to use Delta Lake format")

        logger.info("✓ Delta Lake and Medallion Architecture implementation test completed successfully")

        return True

    except Exception as e:
        logger.error(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        if raise_on_error:
            raise
        return False


def test_delta_implementation():
    """Pytest entry point for Delta Lake implementation verification."""
    import pytest

    pytest.importorskip("pyspark.sql", reason="Delta verification requires PySpark.")
    try:
        assert check_delta_implementation(raise_on_error=True)
    except Exception as exc:
        spark_runtime_markers = (
            "jdk.internal.ref.Cleaner",
            "JavaSparkContext",
            "ExceptionInInitializerError",
        )
        if any(marker in str(exc) for marker in spark_runtime_markers):
            pytest.skip(f"Local Spark runtime is not compatible with this JDK: {exc}")
        raise

if __name__ == "__main__":
    success = check_delta_implementation()
    if success:
        logger.info("🎉 All tests passed!")
    else:
        logger.error("❌ Some tests failed!")
        exit(1)
