# Bun migration notes — frontend

This document describes a safe, incremental approach to adopt Bun as the package manager for the `frontend` project.

Goals
- Use Bun for faster installs and caching while keeping existing build/test tooling (Vite, Vitest, Playwright).
- Minimise risk: start with package manager migration only, then evaluate replacing tooling.

Quickstart (developer)
1. Install Bun (Windows PowerShell):

```powershell
iwr -useb https://bun.sh/install | iex
```

2. From the `frontend/` folder run:

```bash
bun install
bun run dev
# build
bun run build
# run unit tests
bun run test
```

Notes
- `bun install` will generate `bun.lockb`.
- `bun run <script>` will execute scripts defined in `package.json`.
- Keep `vite`, `vitest` and `playwright` for now to avoid feature-coverage gaps.

CI example (GitHub Actions)
- See `/.github/workflows/frontend-bun.yml` for a working example. Key points:
  - Use `oven-sh/setup-bun` to install Bun in CI.
  - Run `bun install` and `bun run build`/`bun run test` as appropriate.

Docker
- If you build frontend inside Docker, add Bun install to the Dockerfile or use an image with Bun preinstalled. Example snippet:

```dockerfile
# install bun
RUN curl -fsSL https://bun.sh/install | bash -s -- --prefix /usr/local \
  && export PATH="/root/.bun/bin:$PATH"
```

Compatibility & Risks
- OS: Bun supports Windows and Linux (verify current compatibility for our CI runners and developer environments).
- Native modules: `canvas` and other native modules may require binaries and could behave differently. Verify they work under Bun.
- Playwright: browser installation and Playwright CLI steps are unchanged — keep Playwright toolchain as-is.
- Lockfiles: decide whether to keep `package-lock.json` or fully migrate to `bun.lockb`. For now, add `bun.lockb` and keep `package-lock.json` in the repo until migration is validated.

Rollback plan
- If issues appear, revert CI and developer docs to use `npm install`/`pnpm` and delete `bun.lockb` from commits.

Next steps recommended
1. Add `bun.lockb` to `.gitignore` only if you prefer not to commit it; otherwise commit it for reproducible installs.
2. Update `frontend/README.md` to reference `bun install` and `bun run`.
3. Add Dockerfile and CI updates (I can create patches for these next).

Reference
- https://bun.sh/
