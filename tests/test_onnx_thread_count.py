# Feature: perfect-10-final, Property 4: ONNX thread count binding
"""
Property 4 — ONNX thread count binding.

For any cpu_cores value in the range 1–256, when ONNXEngine is instantiated,
the ONNX Runtime SessionOptions.intra_op_num_threads SHALL be set to that
cpu_cores value.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.serving.onnx_engine import ONNXEngine


@given(st.integers(min_value=1, max_value=256))
@settings(max_examples=50, deadline=None)
def test_onnx_thread_count_binding(n: int) -> None:
    """
    **Validates: Requirements 2.2**

    Property 4: For any cpu_cores value in [1, 256], ONNXEngine.__init__
    must configure onnxruntime.SessionOptions with intra_op_num_threads == n
    for every model it loads.
    """
    mock_options = MagicMock()

    # Patch ort.SessionOptions to return our controlled mock so we can
    # inspect what intra_op_num_threads was assigned.
    # Patch ort.InferenceSession so no real ONNX file is needed.
    # Patch Path.exists to return True so the if path.exists() branch runs.
    with (
        patch("backend.serving.onnx_engine.ort.SessionOptions", return_value=mock_options),
        patch("backend.serving.onnx_engine.ort.InferenceSession", return_value=MagicMock()),
        patch.object(Path, "exists", return_value=True),
    ):
        ONNXEngine(cpu_cores=n)

    # Every load_model call in __init__ must have set intra_op_num_threads to n.
    # Because all calls share the same mock_options instance, the last write
    # (and all writes) must equal n.
    assert mock_options.intra_op_num_threads == n, (
        f"Expected intra_op_num_threads={n}, got {mock_options.intra_op_num_threads}"
    )
