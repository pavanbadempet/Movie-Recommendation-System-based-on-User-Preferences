import json


def test_deployment_plan_uses_root_dockerfile_and_helm_chart(tmp_path):
    from scripts.deploy_production import DeploymentManager

    manager = DeploymentManager(
        "staging",
        dry_run=True,
        skip_checks=True,
        image_repository="registry.example/apex",
        image_tag="abc123",
        backup_dir=tmp_path,
    )

    commands = manager.deployment_commands()

    assert commands[0] == [
        "docker",
        "build",
        "-f",
        "Dockerfile",
        "-t",
        "registry.example/apex:abc123",
        ".",
    ]
    helm_command = commands[-1]
    assert helm_command[:5] == ["helm", "upgrade", "--install", "apex-staging", "k8s/helm/apex"]
    assert "k8s/staging/deployment.yaml" not in " ".join(helm_command)
    assert "--atomic" in helm_command


def test_skip_checks_is_honored_by_deploy(tmp_path, monkeypatch):
    from scripts.deploy_production import DeploymentManager

    manager = DeploymentManager(
        "staging",
        dry_run=True,
        skip_checks=True,
        image_repository="registry.example/apex",
        image_tag="abc123",
        backup_dir=tmp_path,
    )
    monkeypatch.setattr(manager, "pre_deployment_checks", lambda: (_ for _ in ()).throw(AssertionError("called")))

    assert manager.deploy() is True


def test_backup_receipt_persists_helm_revision_and_values(tmp_path, monkeypatch):
    from scripts.deploy_production import DeploymentManager

    manager = DeploymentManager(
        "staging",
        image_repository="registry.example/apex",
        image_tag="abc123",
        backup_dir=tmp_path,
    )

    def fake_capture(command):
        if "history" in command:
            return json.dumps([{"revision": 7, "status": "deployed"}])
        if "get" in command:
            return "replicaCount: 2\n"
        raise AssertionError(command)

    monkeypatch.setattr(manager, "_capture", fake_capture)
    receipt = manager.create_backup_receipt()

    assert receipt is not None
    assert receipt["revision"] == 7
    persisted = json.loads((tmp_path / receipt["filename"]).read_text(encoding="utf-8"))
    assert persisted["revision"] == 7
    assert persisted["values_yaml"] == "replicaCount: 2\n"
