# Repository Agent Instructions

These instructions apply to automated and human-assisted code changes.

- Preserve unrelated working-tree changes.
- Add or update focused tests before changing behavior.
- Do not describe planned, simulated, fallback, or synthetic behavior as production-ready.
- Training and evaluation artifacts must include real-data provenance; missing evidence must fail explicitly.
- Never commit secrets, local databases, generated model binaries, caches, or dependency directories.
- Run the narrowest relevant tests first, then the full affected test suite.
- Run `git diff --check` before handing off changes.
- Use `CONTRIBUTING.md` for environment setup, coding standards, and pull-request requirements.
