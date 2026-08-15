import os

import requests

DATABRICKS_HOST = "https://dbc-0d2f31ec-d157.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}

r = requests.post(f"{DATABRICKS_HOST}/api/2.1/jobs/run-now", headers=HEADERS, json={"job_id": 303494851952917})
print(f"Trigger New Job Run Status: {r.status_code} - {r.text}")
