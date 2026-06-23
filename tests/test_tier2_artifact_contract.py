from pathlib import Path

import yaml


def test_clean_deployment_configs_default_to_tier3():
    render = yaml.safe_load(Path("render.yaml").read_text(encoding="utf-8"))
    env = {item["key"]: item.get("value") for item in render["services"][0]["envVars"] if "value" in item}
    values = yaml.safe_load(Path("k8s/helm/apex/values.yaml").read_text(encoding="utf-8"))

    assert env["NOVA_SERVING_TIER"] == "tier3"
    assert env["NOVA_SERVING_PROFILE"] == "lite"
    assert values["servingTier"] == "tier3"
    assert values["servingProfile"] == "lite"


def test_explicit_tier2_fails_when_required_onnx_models_are_missing(monkeypatch):
    import backend.serving.app_startup as startup
    import backend.serving.onnx_engine as onnx_module

    class MissingEngine:
        def missing_required_models(self):
            return ["lightgcn", "mmoe_ranker"]

    detector = type("Detector", (), {"_profile": type("Profile", (), {"cpu_cores": 2})()})()
    monkeypatch.setenv("NOVA_SERVING_TIER", "tier2")
    monkeypatch.setattr(onnx_module, "get_onnx_engine", lambda cpu_cores=0: MissingEngine())

    try:
        startup._start_tier2_engine(detector)
    except RuntimeError as exc:
        assert "lightgcn" in str(exc)
    else:
        raise AssertionError("Explicit Tier 2 must fail without its required ONNX artifacts")


def test_onnx_engine_reports_required_model_gaps_without_loading_runtime():
    from backend.serving.onnx_engine import ONNXEngine

    engine = ONNXEngine.__new__(ONNXEngine)
    engine.sessions = {"lightgcn": object()}

    missing = engine.missing_required_models()

    assert "lightgcn" not in missing
    assert "mmoe_ranker" in missing
