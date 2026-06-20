from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_kafka_compose_binds_sensitive_ports_to_localhost_and_requires_airflow_admin_secret():
    compose_path = ROOT / "docker-compose.kafka-cluster.yml"
    text = compose_path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)

    assert "--password admin" not in text
    assert "AIRFLOW_ADMIN_PASSWORD=${AIRFLOW_ADMIN_PASSWORD:?" in text

    for service_name in (
        "airflow-webserver",
        "kafka-1",
        "kafka-2",
        "kafka-3",
        "kafka-ui",
        "spark",
        "backend",
        "frontend",
    ):
        for port in config["services"][service_name].get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{service_name} publishes {port} beyond localhost"


def test_kafka_compose_preserves_full_airflow_bootstrap_script():
    compose_path = ROOT / "docker-compose.kafka-cluster.yml"
    text = compose_path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)

    command = config["services"]["airflow-init"]["command"]
    assert command[0] == "-c"
    assert len(command) == 2

    bootstrap_script = command[1]
    for required_command in (
        "airflow db migrate",
        "airflow users create",
        "mkdir -p /root/.kaggle",
        "chmod 600 /root/.kaggle/kaggle.json",
    ):
        assert required_command in bootstrap_script

    assert "spark-sql-kafka-0-10_2.12:3.5.0" not in text
    assert "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0" in (
        ROOT / "airflow/dags/kafka_spark_integration_dag.py"
    ).read_text(encoding="utf-8")


def test_spark_kafka_dag_bounds_untrusted_event_stream():
    text = (ROOT / "airflow/dags/kafka_spark_integration_dag.py").read_text(encoding="utf-8")

    assert ".option('maxOffsetsPerTrigger'," in text
    assert "allowed_event_types" in text
    assert ".isin(allowed_event_types)" in text


def test_hugging_face_upload_ignores_secret_files():
    from scripts.hf_upload import build_ignore_patterns

    patterns = set(build_ignore_patterns())

    for required in {
        ".env",
        ".env.*",
        "frontend/.env",
        "frontend/.env.*",
        "**/.env",
        "**/.env.*",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
        "*secret*",
        "*credentials*",
    }:
        assert required in patterns


def test_data_refresh_workflow_does_not_write_hf_token_into_notebook_source():
    text = (ROOT / ".github/workflows/data-refresh.yml").read_text(encoding="utf-8")

    assert "HF_TOKEN_PLACEHOLDER" not in text
    assert "os.environ.get('HF_TOKEN'" not in text


def test_secrets_scan_uses_least_privilege_and_pins_trufflehog_to_sha():
    text = (ROOT / ".github/workflows/secrets-scan.yml").read_text(encoding="utf-8")

    assert re.search(r"(?m)^permissions:\n\s+contents:\s+read\s*$", text)
    assert "trufflesecurity/trufflehog@main" not in text
    assert re.search(r"trufflesecurity/trufflehog@[0-9a-f]{40}", text)
