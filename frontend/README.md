# APEX React Frontend

![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue)
![React](https://img.shields.io/badge/React-19-61dafb)
![Vitest](https://img.shields.io/badge/tested%20with-Vitest-6e9f18)

Static React + TypeScript UI for the APEX recommendation platform. Built with Vite 7, React 19, D3.js, and Vitest.

---

## Pages

| Page | Route | Description |
|------|-------|-------------|
| **Home** | `/` | Hero showcase, movie search, AI discovery |
| **Search** | `/search` | Title and semantic search with recommendations |
| **Dashboard** | `/dashboard` | Live hardware profile, serving tier, SLO metrics |
| **Knowledge Graph** | `/knowledge-graph` | D3 force-directed graph of movie connections |
| **Evaluation** | `/evaluation` | Semantic benchmark, recommendation benchmark, offline metrics |
| **Profile** | `/profile` | Personalized recommendations, behavior stats, watch history |
| **Admin** | `/admin` | Ensemble weight management (requires auth) |

---

## Quick Start

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server on port 5173 |
| `npm run build` | Type-check + production build to `dist/` |
| `npm run preview` | Preview production build locally |
| `npm run test` | Run all tests once |
| `npm run test:watch` | Run tests in watch mode |
| `npm run test:coverage` | Run tests with coverage report (80% threshold) |
| `npm run type-check` | TypeScript type check without emitting |
| `npm run lint` | TypeScript strict type check (acts as linter) |

---

## Testing

```bash
# Run all 128 tests
npm run test

# Run with coverage (enforces 80% lines/functions, 75% branches)
npm run test:coverage
```

Test files live in `src/test/`:

| File | Tests | What it covers |
|------|-------|----------------|
| `api.test.ts` | 17 | API client: backendLabel, apiGet, apiPost, loginUser |
| `api-extended.test.ts` | 18 | API wrappers: platformStatus, searchMovies, recordEvent, etc. |
| `hooks.test.ts` | 15 | useHealth, useSlo, useKnowledgeGraph — real implementations |
| `components.test.tsx` | 26 | AuthPage, MovieCard, ErrorBanner, LoadingSpinner |
| `pages.test.tsx` | 36 | Dashboard, KnowledgeGraph, Evaluation, UserProfile, AdminPanel |
| `userprofile-extended.test.tsx` | 8 | Watch history, error states, behavior card edge cases |
| `accessibility.test.tsx` | 8 | WCAG 2.0 A/AA via jest-axe for all 5 pages |

---

## Environment Variables

Create a `.env` file (see `.env.example`):

```ini
VITE_API_URL=https://your-api.onrender.com
VITE_BACKUP_API_URL=https://your-backup.hf.space
VITE_TMDB_IMAGE_BASE=https://image.tmdb.org/t/p/w500
```

The UI has request-level backend failover — if the primary API is sleeping, it automatically retries the backup.

---

## Deployment

### Cloudflare Pages (recommended)

- Root directory: `frontend`
- Build command: `npm ci && npm run build`
- Output directory: `dist`
- Node version: `24`

### GitHub Pages

The repository includes `.github/workflows/frontend-pages.yml` for zero-cost static deployment.

- Set Pages source to `GitHub Actions` in repository settings.
- Push changes under `frontend/**` or run the workflow manually.

### Docker

```bash
docker build -t apex-frontend ./frontend
docker run -p 5173:5173 apex-frontend
```

---

## Architecture

```
src/
├── main.tsx          # App shell, routing, state management
├── AuthPage.tsx      # Login / register form
├── api.ts            # API client with multi-backend failover
├── types.ts          # TypeScript type definitions
├── styles.css        # Global dark theme styles
├── hooks/
│   ├── useHealth.ts          # /health endpoint hook
│   ├── useSlo.ts             # /v1/platform/slo hook
│   └── useKnowledgeGraph.ts  # Knowledge graph data hook
├── pages/
│   ├── Dashboard.tsx         # System health & SLO dashboard
│   ├── KnowledgeGraph.tsx    # D3 force-directed graph
│   ├── Evaluation.tsx        # Benchmark metrics tables
│   ├── UserProfile.tsx       # User profile & recommendations
│   └── AdminPanel.tsx        # Admin controls
└── test/
    ├── setup.ts
    ├── api.test.ts
    ├── api-extended.test.ts
    ├── hooks.test.ts
    ├── components.test.tsx
    ├── pages.test.tsx
    ├── userprofile-extended.test.tsx
    └── accessibility.test.tsx
```
