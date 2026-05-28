# Deployment Guide

How to deploy the backend and frontend.

## Prerequisites

- GitHub account with this repo pushed
- Render account (render.com) 
- Streamlit Cloud account (share.streamlit.io)

## Backend (Render)

1. Go to Render dashboard
2. New -> Blueprint -> Connect your repo and let Render read `render.yaml`
3. The blueprint uses the root `Dockerfile`, which has access to `requirements.txt`, `backend/`, `frontend/`, `models/`, and `data/processed/`
4. Keep the default free-plan serving profile as `NOVA_SERVING_PROFILE=lite` and `NOVA_HEALTH_LOAD_RECOMMENDER=false`
5. Add secret env vars as needed, especially `DATABASE_URL`, `REDIS_URL`, `TMDB_API_KEY`, and `OPENROUTER_API_KEY`
6. Deploy

Takes 5-10 min. Copy the URL when done (e.g. `https://movie-recs-api.onrender.com`).

## Frontend (Cloudflare Pages)

Use the React frontend as the primary free static UI.

1. Go to Cloudflare Pages
2. Connect this repository
3. Recommended settings:
   - Root directory: `frontend`
   - Build command: `npm ci && npm run build`
   - Build output directory: `dist`
4. If you keep root directory as `/`, use:
   - Build command: `cd frontend && npm ci && npm run build`
   - Build output directory: `frontend/dist`
5. Add optional build variables only when overriding defaults:
   - `VITE_API_URL` = primary API gateway
   - `VITE_BACKUP_API_URL` = backup API gateway

The React frontend performs request-level API failover, so the UI can continue through one sleeping or slow free host.

## Frontend Backup (Streamlit Cloud)

1. Go to share.streamlit.io
2. New App -> Select your repo
3. Main file: `streamlit_app.py`
4. In Advanced Settings, add secret:
   ```toml
   API_URL = "https://movie-recs-api.onrender.com"
   TMDB_API_KEY = "your_tmdb_key"
   ```
5. Deploy

## Verify

- Primary frontend: Cloudflare Pages URL
- Backup frontend: `https://<your-app>.streamlit.app`
- Backend docs: `https://<your-api>.onrender.com/docs`
