# Automation Guide: Movie Recommendation System

This document explains the automated data pipeline for the Movie Recommendation System.

## 🔄 Daily Data Refresh
The system is configured to **automatically update** every day at **3:00 AM**.

### What happens daily?
1.  **Download:** Fetches the latest movie dataset from Kaggle (`alanvourch/tmdb-movies-daily-updates`).
2.  **ETL (Extract, Transform, Load):** 
    -   Reads the CSV (1M+ movies).
    -   Filters for high-quality movies (Vote Count ≥ 50).
    -   Generates AI Embeddings (SBERT) for the new movies.
3.  **Index:** Rebuilds the FAISS Search Index.
4.  **Result:** When you wake up, the recommendation engine has the latest movies (like "Avatar: Fire and Ash").

---

## 🛠 Manual Execution
If you want to trigger an update **right now** (e.g., to fix a data issue or test changes):

### 1. Reliable Mode (Recommended)
This uses **Pandas**, which is rock-solid on Windows. It takes about **20-30 minutes**.
```powershell
python refresh.py
```
*(No arguments needed - it defaults to Pandas)*

### 2. Fast/Experimental Mode (PySpark)
This uses **Spark** (local cluster). It is faster (~10-15 mins) but requires careful memory management.
```powershell
python refresh.py --spark
```
*Note: If Spark fails, the system will automatically fall back to Pandas.*

---

## 📂 Logs & Monitoring
All activity is logged to the `logs/` directory.

-   **Check the latest run:**
    ```powershell
    Get-Content logs/refresh_LATEST.log -Tail 20
    ```
-   **Troubleshooting:**
    -   `refresh_*.log`: Main orchestrator logs.
    -   `pipeline_*.log`: Pandas ETL specific logs.

---

## ⚙️ Configuration
-   **Schedule Script:** `schedule_refresh.ps1` (Run as Admin to re-register task).
-   **Orchestrator:** `refresh.py`.
-   **Pipeline Code:** `etl/pipeline.py`.
