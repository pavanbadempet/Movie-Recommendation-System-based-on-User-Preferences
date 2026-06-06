## What
<!-- One-line summary of the change. -->

## Why
<!-- The problem this solves or the improvement it makes. -->

## How
<!-- Key implementation decisions. Link to ADR if applicable. -->

## Testing
<!-- What tests were added or modified. -->
- [ ] Unit tests added / updated
- [ ] Property-based tests added / updated (Hypothesis)
- [ ] API integration test added / updated

## Checklist
- [ ] `python -m ruff check backend/ tests/ scripts/ etl/` passes
- [ ] `python -m ruff format --check backend/ tests/ scripts/ etl/` passes
- [ ] `python -m mypy backend/` passes
- [ ] `python -m pytest tests/ backend/tests/ --cov=backend --cov-fail-under=80` passes
- [ ] No binary artifacts, model weights, or `.env` files added
- [ ] `docs/` updated if public API or architecture changed
- [ ] Targets `develop`, not `main`
