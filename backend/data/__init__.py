"""
Data layer sub-package for APEX.

Handles persistence, multi-tenancy, auth, external integrations,
and catalog management.

Modules:
    database.py          — SQLAlchemy engine, session factory, Base
    auth.py              — JWT token handling, tenant context resolution
    billing.py           — Stripe billing integration
    catalogs.py          — Catalog upload, profiling, and persistence
    experiments.py       — A/B experiment assignment and metrics
    usage.py             — Per-tenant API usage tracking
    remote_recommender.py — HTTP proxy to remote APEX instances
    frontend_failover.py  — Multi-frontend health and routing
"""
