from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest


def _write_identity_model(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["features"], ["scores"])],
        "identity_model",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 4])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, 4])],
    )
    model = helper.make_model(graph, producer_name="test")
    onnx.save(model, path)


def _write_external_data_model(path: Path) -> None:
    bias = numpy_helper.from_array(np.ones(4, dtype=np.float32), name="bias")
    graph = helper.make_graph(
        [helper.make_node("Add", ["features", "bias"], ["scores"])],
        "external_data_model",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 4])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, 4])],
        [bias],
    )
    model = helper.make_model(graph, producer_name="test")
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="ranker.onnx.data",
        size_threshold=0,
    )


def test_build_triton_repository_writes_real_model_and_config(tmp_path):
    from scripts.export_to_triton import build_triton_repository

    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    source_model = onnx_dir / "ranker.onnx"
    _write_identity_model(source_model)

    output_dir = tmp_path / "triton"
    exported = build_triton_repository(onnx_dir=onnx_dir, output_dir=output_dir)

    assert exported == ["ranker"]
    assert (output_dir / "ranker" / "1" / "model.onnx").read_bytes() == source_model.read_bytes()

    config = (output_dir / "ranker" / "config.pbtxt").read_text(encoding="utf-8")
    assert 'platform: "onnxruntime_onnx"' in config
    assert 'name: "features"' in config
    assert "data_type: TYPE_FP32" in config
    assert "dims: [ -1, 4 ]" in config
    assert 'name: "scores"' in config


def test_build_triton_repository_validates_and_copies_external_tensor_data(tmp_path):
    from scripts.export_to_triton import build_triton_repository

    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    _write_external_data_model(onnx_dir / "ranker.onnx")

    output_dir = tmp_path / "triton"
    build_triton_repository(onnx_dir=onnx_dir, output_dir=output_dir)

    assert (output_dir / "ranker" / "1" / "ranker.onnx.data").is_file()


def test_build_triton_repository_fails_when_no_onnx_models_exist(tmp_path):
    from scripts.export_to_triton import build_triton_repository

    with pytest.raises(FileNotFoundError, match="No ONNX models found"):
        build_triton_repository(onnx_dir=tmp_path / "missing", output_dir=tmp_path / "triton")
