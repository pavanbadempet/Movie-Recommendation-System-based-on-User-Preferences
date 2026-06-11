"""
Smoke tests: static correctness checks for the TurboVec migration.

These tests assert, without executing any heavy model code, that:
1. No production module contains a bare `import faiss` statement.
2. MODEL_FILES in model_loader.py does not contain "faiss.index".
3. REQUIRED_FILES in validate_serving_artifacts.py contains "turbovec.tq".
4. The migration utility and recall evaluation scripts exist on disk.

Requirements: 1.5, 2.6, 4.6, 5.7, 6.6, 7.5
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Production modules that must NOT contain `import faiss`
# ---------------------------------------------------------------------------
PRODUCTION_MODULES = [
    "etl/pandas_etl.py",
    "etl/pyspark_etl.py",
    "backend/pipeline/recommender_core.py",
    "backend/intelligence/multimodal_fusion.py",
    "scripts/train_two_tower.py",
]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. No bare `import faiss` in production modules
# ---------------------------------------------------------------------------


def test_pandas_etl_has_no_faiss_import():
    """etl/pandas_etl.py must not import faiss. (Requirement 1.5)"""
    source = _read("etl/pandas_etl.py")
    assert "import faiss" not in source, "etl/pandas_etl.py still contains 'import faiss'"


def test_pyspark_etl_has_no_faiss_import():
    """etl/pyspark_etl.py must not import faiss. (Requirement 2.6)"""
    source = _read("etl/pyspark_etl.py")
    assert "import faiss" not in source, "etl/pyspark_etl.py still contains 'import faiss'"


def test_recommender_core_has_no_faiss_import():
    """backend/pipeline/recommender_core.py must not import faiss. (Requirement 5.7)"""
    source = _read("backend/pipeline/recommender_core.py")
    assert "import faiss" not in source, "backend/pipeline/recommender_core.py still contains 'import faiss'"


def test_multimodal_fusion_has_no_faiss_import():
    """backend/intelligence/multimodal_fusion.py must not import faiss. (Requirement 6.6)"""
    source = _read("backend/intelligence/multimodal_fusion.py")
    assert "import faiss" not in source, "backend/intelligence/multimodal_fusion.py still contains 'import faiss'"


def test_train_two_tower_has_no_faiss_import():
    """scripts/train_two_tower.py must not import faiss. (Requirement 7.5)"""
    source = _read("scripts/train_two_tower.py")
    assert "import faiss" not in source, "scripts/train_two_tower.py still contains 'import faiss'"


# ---------------------------------------------------------------------------
# 2. MODEL_FILES in model_loader.py must not contain "faiss.index"
# ---------------------------------------------------------------------------


def test_model_loader_has_no_faiss_index_key():
    """MODEL_FILES in model_loader.py must not register faiss.index. (Requirement 4.6)"""
    import backend.models.model_loader as loader

    assert "faiss.index" not in loader.MODEL_FILES, "model_loader.MODEL_FILES still contains 'faiss.index' entry"
    assert "turbovec.tq" in loader.MODEL_FILES, "model_loader.MODEL_FILES does not contain 'turbovec.tq' entry"


# ---------------------------------------------------------------------------
# 3. REQUIRED_FILES in validate_serving_artifacts.py must contain "turbovec.tq"
# ---------------------------------------------------------------------------


def test_validate_serving_artifacts_requires_turbovec():
    """REQUIRED_FILES must contain turbovec.tq and not faiss.index. (Requirements 11.1, 11.2)"""
    from scripts.validate_serving_artifacts import REQUIRED_FILES

    assert "turbovec.tq" in REQUIRED_FILES, "validate_serving_artifacts.REQUIRED_FILES does not contain 'turbovec.tq'"
    assert "faiss.index" not in REQUIRED_FILES, "validate_serving_artifacts.REQUIRED_FILES still contains 'faiss.index'"


# ---------------------------------------------------------------------------
# 4. Migration utility and recall evaluation scripts exist on disk
# ---------------------------------------------------------------------------


def test_migrate_faiss_to_turbovec_script_exists():
    """scripts/migrate_faiss_to_turbovec.py must exist on disk. (Requirement 8.4)"""
    script = REPO_ROOT / "scripts" / "migrate_faiss_to_turbovec.py"
    assert script.exists(), f"Migration script not found: {script}"
    assert script.is_file(), f"Expected a file, not a directory: {script}"


def test_evaluate_turbovec_recall_script_exists():
    """scripts/evaluate_turbovec_recall.py must exist on disk. (Requirement 9.3)"""
    script = REPO_ROOT / "scripts" / "evaluate_turbovec_recall.py"
    assert script.exists(), f"Recall evaluation script not found: {script}"
    assert script.is_file(), f"Expected a file, not a directory: {script}"
