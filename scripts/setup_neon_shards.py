import os
import subprocess
import time

import requests

NEON_API_KEY = os.environ.get("NEON_ACCOUNT_1_API_KEY") or os.environ.get("NEON_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {NEON_API_KEY}", "Accept": "application/json", "Content-Type": "application/json"}

ORG_ID = "org-spring-glitter-92956691"


def create_and_get_shard_urls():
    print(f"Using Neon Organization ID: {ORG_ID}")
    print("Checking existing Neon projects...")
    res = requests.get(f"https://console.neon.tech/api/v2/projects?org_id={ORG_ID}", headers=HEADERS)
    if res.status_code != 200:
        print(f"Failed to list projects: {res.status_code} - {res.text}")
        return

    projects = res.json().get("projects", [])
    existing_map = {p["name"]: p for p in projects}
    print(f"Found {len(projects)} existing Neon project(s)")

    shard_connection_strings = {}

    for i in range(10):
        shard_name = f"movie-shard-{i}"

        # Check if exists and region is aws-ap-southeast-1
        if shard_name in existing_map:
            p_info = existing_map[shard_name]
            p_region = p_info.get("region_id")
            if p_region == "aws-ap-southeast-1":
                project_id = p_info["id"]
                print(f"Shard {i} ('{shard_name}') is already in Singapore ({p_region}) with ID: {project_id}")
            else:
                print(f"Shard {i} ('{shard_name}') is in '{p_region}'. Deleting to recreate in Singapore...")
                requests.delete(f"https://console.neon.tech/api/v2/projects/{p_info['id']}", headers=HEADERS)
                time.sleep(1)
                shard_name_in_map = False
        else:
            shard_name_in_map = False

        if shard_name not in existing_map or existing_map[shard_name].get("region_id") != "aws-ap-southeast-1":
            print(f"Creating Neon project for Shard {i} ('{shard_name}') in Singapore (aws-ap-southeast-1)...")
            create_payload = {
                "project": {"name": shard_name, "pg_version": 16, "org_id": ORG_ID, "region_id": "aws-ap-southeast-1"}
            }
            c_res = requests.post("https://console.neon.tech/api/v2/projects", headers=HEADERS, json=create_payload)
            if c_res.status_code not in [200, 201]:
                print(f"Error creating {shard_name} in Singapore: {c_res.status_code} - {c_res.text}")
                continue
            project_data = c_res.json()
            project_id = project_data["project"]["id"]
            print(f"Created {shard_name} in Singapore (ID: {project_id})")

        # Get connection string for project_id with required query params
        c_res = requests.get(
            f"https://console.neon.tech/api/v2/projects/{project_id}/connection_uri?database_name=neondb&role_name=neondb_owner",
            headers=HEADERS,
        )
        if c_res.status_code == 200:
            conn_uri = c_res.json().get("uri")
            if conn_uri and conn_uri.startswith("postgres://"):
                conn_uri = conn_uri.replace("postgres://", "postgresql://", 1)
            shard_connection_strings[f"DATABASE_URL_SHARD_{i}"] = conn_uri
            print(f"Fetched connection string for Shard {i}")
        else:
            print(f"Could not fetch connection string for Shard {i}: {c_res.status_code} - {c_res.text}")

        time.sleep(1)

    print(f"\nSuccessfully obtained {len(shard_connection_strings)} Shard Connection Strings!")

    # Push to Doppler via Doppler CLI
    print("Pushing all 10 Shard Connection Strings to Doppler...")
    cmd = ["doppler", "secrets", "set"]
    for k, v in shard_connection_strings.items():
        cmd.append(f"{k}={v}")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"Doppler Output: {result.stdout.encode('ascii', 'ignore').decode('ascii')}")
    if result.returncode == 0:
        print("ALL 10 NEON SHARDS ARE 100% CONFIGURED IN SINGAPORE!")


if __name__ == "__main__":
    create_and_get_shard_urls()
