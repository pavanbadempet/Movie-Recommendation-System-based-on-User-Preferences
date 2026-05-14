# Nova React Frontend

Static React UI for the Nova recommendation platform.

## Local

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Free Hosting Settings

Cloudflare Pages, Vercel, or Netlify can deploy this as a static Vite app.

- Recommended Cloudflare setup:
  - Root directory: `frontend`
  - Build command: `npm ci && npm run build`
  - Output directory: `dist`
- If Cloudflare root directory stays as repository root `/`:
  - Build command: `cd frontend && npm ci && npm run build`
  - Output directory: `frontend/dist`
- Node version: `24` or current LTS

Optional environment variables:

- `VITE_API_URL`: primary API gateway, default Hugging Face Space
- `VITE_BACKUP_API_URL`: backup API, default Render gateway
- `VITE_TMDB_IMAGE_BASE`: poster image base URL

The UI has request-level backend failover, so a sleeping free host should not break the whole app.

## GitHub Pages

The repository includes `.github/workflows/frontend-pages.yml` for a zero-cost static deployment from `frontend/dist`.

- In repository settings, set Pages source to `GitHub Actions`.
- Push changes under `frontend/**` or run `Deploy Frontend to GitHub Pages` manually.
- The workflow builds with `VITE_BASE_PATH` set to the repository path, so Vite assets resolve correctly on `https://<user>.github.io/<repo>/`.
- If Pages is not enabled yet, the workflow still validates the frontend build and skips only the publish step.
