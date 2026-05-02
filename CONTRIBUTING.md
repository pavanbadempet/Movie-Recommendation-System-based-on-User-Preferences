# Contributing

First off, thanks for looking at the code. This project started as a personal tool, but I'm always open to ideas that make it smarter or faster.

## Philosophy

I try to keep things simple.

* **Pragmatism > Purity**: If a hack works and is well-commented, it's better than an over-engineered abstraction.
* **Comments Matter**: I write comments for *why* something is done, not just what it does. Please do the same.
* **Keep it Fast**: The search needs to feel instant. Heavy computations should stay in the ETL pipeline, not the request path.

## Local Setup

1. Fork & Clone.
2. `python manage.py setup` - This should handle the venv and deps.
3. `python manage.py run` - Spins up everything you need.

## The ETL Pipeline

If you're messing with the data processing (`etl/` folder), be aware that generating embeddings takes time.

* Use a subset of data for testing if you can.
* Check `etl/pandas_etl.py` - that's where the logic lives. The Spark stuff (`etl/spark_etl.py`) is mostly for showing off scalarability, but Pandas is the daily driver.

## Submitting PRs

* Don't worry about perfect commit messages.
* If you're adding a feature, just drop a line in the PR description about why it's cool.
* Please verify that `python manage.py test` still passes.

Thanks! 🚀
