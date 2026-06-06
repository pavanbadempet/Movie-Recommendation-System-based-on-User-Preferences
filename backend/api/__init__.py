"""
API route handlers sub-package for APEX.

All FastAPI router factories live here. Each module exposes a `create_*_router`
factory function that accepts injected dependencies — keeping routes testable
without requiring a live application context.

Routers:
    recommendation_routes   — Core recommendations, multi-modal, knowledge graph
    browse_routes           — Catalog browsing and filtering
    catalog_routes          — Catalog upload and management
    evaluation_routes       — Offline evaluation, benchmarks, SLO
    experiment_routes       — A/B experiment management
    artifact_routes         — Artifact health, reload, and refresh
    admin_routes            — Admin operations (token-protected)
    auth_routes             — JWT auth and user registration
    billing_routes          — Stripe Checkout, Portal, Webhook, usage
    chat                    — LLM chat generation helper
"""
