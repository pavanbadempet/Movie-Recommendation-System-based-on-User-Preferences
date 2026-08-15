import os

import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}


def preserve_gpu_and_trigger():
    print("Keeping Serverless GPU (1xA10) compute settings intact...")
    run_res = requests.post(
        f"{DATABRICKS_HOST}/api/2.1/jobs/run-now", headers=HEADERS, json={"job_id": 303494851952917}
    )
    print(f"Trigger Run Response: {run_res.status_code} - {run_res.text}")


if __name__ == "__main__":
    preserve_gpu_and_trigger()
