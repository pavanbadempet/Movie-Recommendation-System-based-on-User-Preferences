# Databricks notebook source
# MAGIC %md
# MAGIC # Centralized Doppler Secret & Environment Manager
# MAGIC
# MAGIC ## 📖 System Design
# MAGIC Provides single-source-of-truth secret resolution for the APEX Data Lakehouse.
# MAGIC Fetches all enterprise environment secrets (Kaggle API keys, Database URLs, Databricks tokens)
# MAGIC from Doppler REST API in a single encrypted payload and injects them into `os.environ`.

# COMMAND ----------
import logging
import os

import requests

logger = logging.getLogger("DopplerConfig")


def resolve_doppler_token(dbutils=None, env="dev"):
    """
    Centralized 3-Tier Zero-Trust Doppler Secret Token Resolution.
    Tier 1: Unity Catalog Volume Storage (/Volumes/apex/default/secrets/{env}_doppler_token.txt)
    Tier 2: Databricks Secret Scope (dbutils.secrets.get("apex_secrets", "doppler_token"))
    Tier 3: Databricks Job Parameter Widget (dbutils.widgets.get("DOPPLER_TOKEN"))
    """
    doppler_token = None

    # Tier 1: Unity Catalog Volume Storage
    for token_name in [f"{env}_doppler_token.txt", "doppler_token.txt"]:
        try:
            token_path = f"/Volumes/apex/default/secrets/{token_name}"
            if dbutils:
                doppler_token = dbutils.fs.head(token_path).strip()
            elif os.path.exists(token_path):
                with open(token_path) as f:
                    doppler_token = f.read().strip()
            if doppler_token:
                print(f"Loaded Doppler Token from Volume: {token_path}")
                break
        except Exception:
            pass

    # Tier 2: Databricks Secret Scope (Key Vault / KMS Backed)
    if not doppler_token and dbutils:
        try:
            doppler_token = dbutils.secrets.get(scope="apex_secrets", key="doppler_token").strip()
            if doppler_token:
                print("Loaded Doppler Token from Databricks Secret Scope: apex_secrets/doppler_token")
        except Exception:
            pass

    # Tier 3: Job Parameter Widget Fallback
    if not doppler_token and dbutils:
        try:
            doppler_token = dbutils.widgets.get("DOPPLER_TOKEN").strip()
        except Exception:
            pass

    return doppler_token


def load_centralized_doppler_secrets(dbutils=None, env="dev"):
    """
    Centralized Doppler Secret Loader.
    Fetches all environment secrets from Doppler API in a single HTTP payload,
    injects them into os.environ, and returns a dictionary of active secrets.
    """
    doppler_token = resolve_doppler_token(dbutils=dbutils, env=env)

    if not doppler_token:
        raise ValueError(
            f"DOPPLER_TOKEN is missing! Please upload '{env}_doppler_token.txt' to "
            f"/Volumes/apex/default/secrets/ or configure secret scope 'apex_secrets'."
        )

    try:
        response = requests.get(
            "https://api.doppler.com/v3/configs/config/secrets",
            headers={"Authorization": f"Bearer {doppler_token}", "Accept": "application/json"},
        )
        response.raise_for_status()
        raw_secrets = response.json().get("secrets", {})

        extracted_secrets = {}
        for key, val in raw_secrets.items():
            computed_val = val.get("computed", "")
            extracted_secrets[key] = computed_val
            os.environ[key] = computed_val

        print(f"Successfully loaded {len(extracted_secrets)} centralized secrets from Doppler!")
        return extracted_secrets

    except Exception as e:
        sanitized_err = str(e).replace(str(doppler_token), "***REDACTED***") if doppler_token else str(e)
        raise ValueError(f"Failed to load centralized secrets from Doppler API: {sanitized_err}")
