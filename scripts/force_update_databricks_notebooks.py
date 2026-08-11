import os
import base64
import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}

NOTEBOOKS = [
    "00_kaggle_download",
    "01_pyspark_etl",
    "01b_streaming_events",
    "01c_gpu_embeddings",
    "02_export_to_neon",
    "doppler_config"
]

def force_update_notebooks():
    print("Force updating Databricks Workspace Notebooks via REST API...")

    for nb in NOTEBOOKS:
        local_path = f"databricks_notebooks/{nb}.py"
        workspace_path = f"/Users/pavan9b@gmail.com/Movie-Recommendation-System/databricks_notebooks/{nb}"

        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()

            b64_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")

            import_url = f"{DATABRICKS_HOST}/api/2.0/workspace/import"
            payload = {
                "path": workspace_path,
                "format": "SOURCE",
                "language": "PYTHON",
                "content": b64_content,
                "overwrite": True
            }

            res = requests.post(import_url, headers=HEADERS, json=payload)
            print(f"Import {nb}: Status {res.status_code}")

    print("\nTriggering fresh run of Streaming Job (Job ID: 772367112113846)...")
    requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/runs/cancel", headers=HEADERS, json={"run_id": 136400498247464})
    run_res = requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/run-now", headers=HEADERS, json={"job_id": 772367112113846})
    print(f"Fresh Run Status: {run_res.status_code} - {run_res.text}")

if __name__ == "__main__":
    force_update_notebooks()
