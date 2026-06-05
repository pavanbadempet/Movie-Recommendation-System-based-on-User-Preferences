"""
Property-based test for ablation report serialization round-trip.
# Feature: architecture-design-perfection, Property 11: Ablation Report Serialization Round-Trip
"""
import json
import sys
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# Add scripts/ to path so we can import ablation_study
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from ablation_study import AblationReport, ModelAblationResult


def _model_result_strategy():
    return st.builds(
        ModelAblationResult,
        model=st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        ),
        ndcg_without=st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        delta=st.one_of(
            st.none(),
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        marginal_contribution_pct=st.one_of(
            st.none(),
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        ),
    )


def _ablation_report_strategy():
    return st.builds(
        AblationReport,
        run_timestamp=st.just("2024-01-01T00:00:00Z"),
        full_ensemble_ndcg=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        models=st.lists(_model_result_strategy(), min_size=0, max_size=6),
    )


# Property 11: Ablation Report Serialization Round-Trip
# Validates: Requirements ablation report correctness
@given(_ablation_report_strategy())
@settings(max_examples=100)
def test_ablation_report_serialization_roundtrip(report):
    """Serializing and deserializing an AblationReport produces identical data.

    # Feature: architecture-design-perfection, Property 11: Ablation Report Serialization Round-Trip
    """
    import dataclasses

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "ablation_report.json"

        # Serialize via dataclasses.asdict + json.dumps (same as save_report internals)
        report_dict = dataclasses.asdict(report)
        output_path.write_text(
            json.dumps(report_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Deserialize
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

    # Verify top-level numeric field round-trips within tolerance
    assert abs(loaded["full_ensemble_ndcg"] - report.full_ensemble_ndcg) < 1e-9, (
        f"full_ensemble_ndcg mismatch: {loaded['full_ensemble_ndcg']} != {report.full_ensemble_ndcg}"
    )

    # Verify timestamp is preserved exactly
    assert loaded["run_timestamp"] == report.run_timestamp

    # Verify per-model results
    assert len(loaded["models"]) == len(report.models), (
        f"models length mismatch: {len(loaded['models'])} != {len(report.models)}"
    )

    for orig, loaded_m in zip(report.models, loaded["models"]):
        assert orig.model == loaded_m["model"], (
            f"model name mismatch: {orig.model!r} != {loaded_m['model']!r}"
        )

        if orig.ndcg_without is not None:
            assert loaded_m["ndcg_without"] is not None
            assert abs(loaded_m["ndcg_without"] - orig.ndcg_without) < 1e-9, (
                f"ndcg_without mismatch for {orig.model}: "
                f"{loaded_m['ndcg_without']} != {orig.ndcg_without}"
            )
        else:
            assert loaded_m["ndcg_without"] is None, (
                f"Expected ndcg_without=None for {orig.model}, got {loaded_m['ndcg_without']}"
            )

        if orig.delta is not None:
            assert loaded_m["delta"] is not None
            assert abs(loaded_m["delta"] - orig.delta) < 1e-9, (
                f"delta mismatch for {orig.model}: {loaded_m['delta']} != {orig.delta}"
            )
        else:
            assert loaded_m["delta"] is None, (
                f"Expected delta=None for {orig.model}, got {loaded_m['delta']}"
            )

        if orig.marginal_contribution_pct is not None:
            assert loaded_m["marginal_contribution_pct"] is not None
            assert abs(loaded_m["marginal_contribution_pct"] - orig.marginal_contribution_pct) < 1e-9, (
                f"marginal_contribution_pct mismatch for {orig.model}: "
                f"{loaded_m['marginal_contribution_pct']} != {orig.marginal_contribution_pct}"
            )
        else:
            assert loaded_m["marginal_contribution_pct"] is None, (
                f"Expected marginal_contribution_pct=None for {orig.model}, "
                f"got {loaded_m['marginal_contribution_pct']}"
            )
