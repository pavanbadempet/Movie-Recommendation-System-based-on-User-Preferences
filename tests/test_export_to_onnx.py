from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scripts import export_to_onnx


def test_load_checkpoint_strict_rejects_incompatible_weights(tmp_path):
    checkpoint = tmp_path / "model.pth"
    torch.save(nn.Linear(3, 2).state_dict(), checkpoint)

    with pytest.raises(RuntimeError, match="incompatible"):
        export_to_onnx.load_checkpoint_strict(nn.Linear(4, 2), checkpoint, "test model")


def test_export_model_to_onnx_propagates_export_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(export_to_onnx, "ONNX_DIR", Path(tmp_path))

    def fail_export(*args, **kwargs):
        raise RuntimeError("onnx export failed")

    monkeypatch.setattr(torch.onnx, "export", fail_export)

    with pytest.raises(RuntimeError, match="onnx export failed"):
        export_to_onnx.export_model_to_onnx(
            nn.Linear(4, 1),
            torch.zeros(1, 4),
            "linear.onnx",
            input_names=["features"],
            output_names=["score"],
            dynamic_axes={"features": {0: "batch_size"}, "score": {0: "batch_size"}},
        )
