# Automation Guide

How the daily data refresh works.

## Daily Schedule

The system updates at 3:00 AM daily:
1. Downloads latest TMDB data from Kaggle
2. Filters for quality movies (vote count >= 50)
3. Generates embeddings for new movies
4. Rebuilds the FAISS index

## Manual refresh

Run the refresh script directly:

```powershell
# Standard mode (Pandas, ~20-30 min)
python daily_refresh.py

# Fast mode (PySpark, ~10-15 min, may need more memory)
python daily_refresh.py --spark
```

## Logs

Check `logs/` directory:

```powershell
Get-Content logs/refresh_LATEST.log -Tail 20
```

## Files

- `schedule_refresh.ps1` - Windows Task Scheduler setup
- `daily_refresh.py` - Main refresh script
- `etl/pipeline.py` - ETL logic
