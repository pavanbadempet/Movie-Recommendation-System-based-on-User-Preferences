import os
import time
import requests
import subprocess

NEON_API_KEY_2 = "napi_22mivnfdx6b0iz2z51zxtwqpbi7o098o9hycqm7vzkl909oqxr9h56od0trewvsi"
ORG_ID_2 = "org-blue-cell-04479202"
HEADERS = {
    "Authorization": f"Bearer {NEON_API_KEY_2}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def setup_account2_vector_shards():
    print(f"Checking Account 2 projects in Singapore (Org: {ORG_ID_2})...")
    res = requests.get(f"https://console.neon.tech/api/v2/projects?org_id={ORG_ID_2}", headers=HEADERS)
    existing_projects = {p["name"]: p for p in res.json().get("projects", [])} if res.status_code == 200 else {}

    shard_connection_strings = {}

    for i in range(10, 14):
        shard_name = f"movie-shard-{i}"
        if shard_name in existing_map if 'existing_map' in locals() else shard_name in existing_projects:
            project_id = existing_projects[shard_name]["id"]
            print(f"Shard {i} ('{shard_name}') already exists with ID: {project_id}")
        else:
            print(f"Creating Neon project for Shard {i} ('{shard_name}') in Account 2 Singapore (aws-ap-southeast-1)...")
            create_payload = {
                "project": {
                    "name": shard_name,
                    "pg_version": 16,
                    "org_id": ORG_ID_2,
                    "region_id": "aws-ap-southeast-1"
                }
            }
            c_res = requests.post("https://console.neon.tech/api/v2/projects", headers=HEADERS, json=create_payload)
            if c_res.status_code not in [200, 201]:
                print(f"Error creating {shard_name}: {c_res.status_code} - {c_res.text}")
                continue
            project_id = c_res.json()["project"]["id"]
            print(f"Created {shard_name} in Singapore (ID: {project_id})")

        # Fetch connection URI
        c_res = requests.get(f"https://console.neon.tech/api/v2/projects/{project_id}/connection_uri?database_name=neondb&role_name=neondb_owner", headers=HEADERS)
        if c_res.status_code == 200:
            conn_uri = c_res.json().get("uri")
            if conn_uri and conn_uri.startswith("postgres://"):
                conn_uri = conn_uri.replace("postgres://", "postgresql://", 1)
            shard_connection_strings[f"DATABASE_URL_SHARD_{i}"] = conn_uri
            print(f"Fetched connection string for Shard {i}")

        time.sleep(1)

    print("\nPushing Account 2 Vector Shard Connection Strings to Doppler...")
    cmd = ["doppler", "secrets", "set"]
    for k, v in shard_connection_strings.items():
        cmd.append(f"{k}={v}")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"Doppler Output: {result.stdout.encode('ascii', 'ignore').decode('ascii')}")
    if result.returncode == 0:
        print("ACCOUNT 2 VECTOR SHARDS (10-13) ARE 100% CONFIGURED IN DOPPLER!")

if __name__ == "__main__":
    setup_account2_vector_shards()
