import logging
import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ensemble_engine import get_apex_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def test_apex_engine():
    logger.info("=== INITIALIZING APEX ENSEMBLE ENGINE TEST ===")

    # 1. Initialize Engine
    logger.info("Loading Models...")
    try:
        apex = get_apex_engine(num_items=10000)
        logger.info("✅ Apex Engine Initialized Successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        return

    # 2. Test Forward Pass
    logger.info("Testing Forward Pass through all 4 paradigms...")
    test_user = 42
    test_candidates = [100, 200, 300, 400, 500]

    try:
        scores = apex.predict_ensemble(test_user, test_candidates)
        logger.info("✅ Forward Pass Successful.")

        logger.info("--- Output Scores ---")
        for item_id, score in scores.items():
            logger.info(f"Item {item_id}: Score = {score:.4f}")

    except Exception as e:
        logger.error(f"❌ Forward Pass Failed: {e}")
        return

    logger.info("=== TEST PASSED ===")


if __name__ == "__main__":
    test_apex_engine()
