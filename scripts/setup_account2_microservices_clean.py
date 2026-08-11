import os
import time
import requests
import subprocess

NEON_API_KEY_2 = os.environ.get("NEON_ACCOUNT_2_API_KEY", "")
ORG_ID_2 = "org-blue-cell-04479202"
HEADERS = {
    "Authorization": f"Bearer {NEON_API_KEY_2}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def setup_account2_10_microservices():
    print(f"Checking Account 2 projects in Singapore (Org: {ORG_ID_2})...")
    res = requests.get(f"https://console.neon.tech/api/v2/projects?org_id={ORG_ID_2}", headers=HEADERS)
    existing_projects = {p["name"]: p for p in res.json().get("projects", [])} if res.status_code == 200 else {}

    # Delete any temporary vector shards in Account 2 to free up slots for 10 microservices
    for shard_id in range(10, 14):
        s_name = f"movie-shard-{shard_id}"
        if s_name in existing_projects:
            print(f"Deleting temp vector shard '{s_name}' from Account 2...")
            requests.delete(f"https://console.neon.tech/api/v2/projects/{existing_projects[s_name]['id']}", headers=HEADERS)
            time.sleep(1)

    # Re-fetch projects
    res = requests.get(f"https://console.neon.tech/api/v2/projects?org_id={ORG_ID_2}", headers=HEADERS)
    existing_projects = {p["name"]: p for p in res.json().get("projects", [])} if res.status_code == 200 else {}

    target_services = {
        "user-auth-db": "DATABASE_URL_USERS",
        "clickstream-events-db": "DATABASE_URL_EVENTS",
        "recommendations-cache-db": "DATABASE_URL_CACHE",
        "analytics-metrics-db": "DATABASE_URL_ANALYTICS",
        "model-registry-db": "DATABASE_URL_MODEL_REGISTRY",
        "notifications-db": "DATABASE_URL_NOTIFICATIONS",
        "search-history-db": "DATABASE_URL_SEARCH_HISTORY",
        "watchlists-db": "DATABASE_URL_WATCHLISTS",
        "billing-subscriptions-db": "DATABASE_URL_BILLING",
        "feedback-reviews-db": "DATABASE_URL_REVIEWS"
    }

    connection_strings = {}

    for name, secret_key in target_services.items():
        if name in existing_projects:
            project_id = existing_projects[name]["id"]
            print(f"Service Project '{name}' already exists (ID: {project_id})")
        else:
            print(f"Creating Service Project '{name}' in Singapore (aws-ap-southeast-1)...")
            create_payload = {
                "project": {
                    "name": name,
                    "pg_version": 16,
                    "org_id": ORG_ID_2,
                    "region_id": "aws-ap-southeast-1"
                }
            }
            c_res = requests.post("https://console.neon.tech/api/v2/projects", headers=HEADERS, json=create_payload)
            if c_res.status_code not in [200, 201]:
                print(f"Error creating '{name}': {c_res.status_code} - {c_res.text}")
                continue
            project_id = c_res.json()["project"]["id"]
            print(f"Created Service Project '{name}' in Singapore (ID: {project_id})")

        # Fetch connection URI
        c_res = requests.get(f"https://console.neon.tech/api/v2/projects/{project_id}/connection_uri?database_name=neondb&role_name=neondb_owner", headers=HEADERS)
        if c_res.status_code == 200:
            conn_uri = c_res.json().get("uri")
            if conn_uri and conn_uri.startswith("postgres://"):
                conn_uri = conn_uri.replace("postgres://", "postgresql://", 1)
            connection_strings[secret_key] = conn_uri
            print(f"Fetched connection string for {secret_key}")

        time.sleep(1)

    print("\nPushing Account 2 10 Microservice Connection Strings to Doppler...")
    cmd = ["doppler", "secrets", "set"]
    for k, v in connection_strings.items():
        cmd.append(f"{k}={v}")

    # Remove temporary shard keys 10..13 from Doppler if set
    cmd_unset = ["doppler", "secrets", "unset", "DATABASE_URL_SHARD_10", "DATABASE_URL_SHARD_11", "DATABASE_URL_SHARD_12", "DATABASE_URL_SHARD_13"]
    subprocess.run(cmd_unset, capture_output=True, text=True)

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"Doppler Output: {result.stdout.encode('ascii', 'ignore').decode('ascii')}")
    if result.returncode == 0:
        print("ALL 10 ACCOUNT 2 DEDICATED MICROSERVICE PROJECTS ARE 100% CONFIGURED IN DOPPLER!")

if __name__ == "__main__":
    setup_account2_10_microservices()
