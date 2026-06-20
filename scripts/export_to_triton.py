"""Package validated ONNX models as an NVIDIA Triton model repository."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import shutil

import onnx
from onnx import TensorProto

logger = logging.getLogger(__name__)

DEFAULT_ONNX_DIR = Path("models") / "onnx"
DEFAULT_TRITON_MODEL_REPO = Path("triton_model_repository")
_VALID_MODEL_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_TRITON_DATA_TYPES = {
    TensorProto.FLOAT: "TYPE_FP32",
    TensorProto.UINT8: "TYPE_UINT8",
    TensorProto.INT8: "TYPE_INT8",
    TensorProto.UINT16: "TYPE_UINT16",
    TensorProto.INT16: "TYPE_INT16",
    TensorProto.INT32: "TYPE_INT32",
    TensorProto.INT64: "TYPE_INT64",
    TensorProto.STRING: "TYPE_STRING",
    TensorProto.BOOL: "TYPE_BOOL",
    TensorProto.FLOAT16: "TYPE_FP16",
    TensorProto.DOUBLE: "TYPE_FP64",
    TensorProto.UINT32: "TYPE_UINT32",
    TensorProto.UINT64: "TYPE_UINT64",
    TensorProto.BFLOAT16: "TYPE_BF16",
}


def _tensor_dims(value_info) -> list[int]:
    dims: list[int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(int(dim.dim_value) if dim.HasField("dim_value") else -1)
    return dims


def _triton_tensor_block(kind: str, value_info) -> str:
    elem_type = value_info.type.tensor_type.elem_type
    try:
        data_type = _TRITON_DATA_TYPES[elem_type]
    except KeyError as exc:
        type_name = TensorProto.DataType.Name(elem_type)
        raise ValueError(f"Unsupported ONNX tensor type for {value_info.name}: {type_name}") from exc
    dims = ", ".join(str(value) for value in _tensor_dims(value_info))
    return (
        f"{kind} [\n"
        "  {\n"
        f'    name: "{value_info.name}"\n'
        f"    data_type: {data_type}\n"
        f"    dims: [ {dims} ]\n"
        "  }\n"
        "]"
    )


def _config_for_model(model_name: str, model: onnx.ModelProto) -> str:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializer_names]
    outputs = list(model.graph.output)
    if not inputs or not outputs:
        raise ValueError(f"ONNX model {model_name} must expose at least one input and one output")

    sections = [
        f'name: "{model_name}"',
        'platform: "onnxruntime_onnx"',
        "max_batch_size: 0",
    ]
    sections.extend(_triton_tensor_block("input", value) for value in inputs)
    sections.extend(_triton_tensor_block("output", value) for value in outputs)
    return "\n\n".join(sections) + "\n"


def _copy_external_data(model: onnx.ModelProto, source_dir: Path, version_dir: Path) -> None:
    copied: set[Path] = set()
    for initializer in model.graph.initializer:
        if initializer.data_location != TensorProto.EXTERNAL:
            continue
        metadata = {entry.key: entry.value for entry in initializer.external_data}
        location = metadata.get("location")
        if not location:
            raise ValueError(f"External initializer {initializer.name} has no data location")
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe external ONNX data path: {location}")

        source = source_dir / relative
        if not source.is_file():
            raise FileNotFoundError(f"External ONNX data file not found: {source}")
        destination = version_dir / relative
        if destination in copied:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.add(destination)


def build_triton_repository(
    onnx_dir: Path = DEFAULT_ONNX_DIR,
    output_dir: Path = DEFAULT_TRITON_MODEL_REPO,
) -> list[str]:
    """Copy valid ONNX graphs into Triton's model-repository layout.

    Returns the exported Triton model names. The function fails if no ONNX
    models exist or if any graph/config cannot be validated.
    """
    onnx_dir = Path(onnx_dir)
    output_dir = Path(output_dir)
    model_paths = sorted(onnx_dir.glob("*.onnx")) if onnx_dir.is_dir() else []
    if not model_paths:
        raise FileNotFoundError(f"No ONNX models found in {onnx_dir}")

    exported: list[str] = []
    for source_path in model_paths:
        model_name = source_path.stem
        if not _VALID_MODEL_NAME.fullmatch(model_name):
            raise ValueError(f"Invalid Triton model name derived from {source_path.name}: {model_name}")

        onnx.checker.check_model(source_path)
        model = onnx.load(source_path, load_external_data=False)
        config = _config_for_model(model_name, model)

        model_dir = output_dir / model_name
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, version_dir / "model.onnx")
        _copy_external_data(model, source_path.parent, version_dir)
        (model_dir / "config.pbtxt").write_text(config, encoding="utf-8")
        exported.append(model_name)
        logger.info("Packaged ONNX model for Triton: %s", model_name)

    return exported


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRITON_MODEL_REPO)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    exported = build_triton_repository(args.onnx_dir, args.output_dir)
    logger.info("Triton repository ready at %s with %d model(s): %s", args.output_dir, len(exported), ", ".join(exported))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Triton repository export failed")
        raise SystemExit(1)
