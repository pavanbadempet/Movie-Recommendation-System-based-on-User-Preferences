-- =============================================================================
-- Migration V1: Initial schema
-- Flyway-compatible versioned migration.
-- Idempotent: safe to run on a fresh database.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ---------------------------------------------------------------------------
-- 1. Tenant dimension (B2B SaaS multi-tenancy)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_tenant (
    tenant_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_name VARCHAR(255) NOT NULL,
    plan_tier   VARCHAR(50)  NOT NULL DEFAULT 'free',
    is_active   BOOLEAN      DEFAULT TRUE,
    created_at  TIMESTAMPTZ  DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMPTZ  DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- 2. API key dimension
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_api_key (
    api_key_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id     UUID NOT NULL REFERENCES dim_tenant(tenant_id),
    api_key_hash  VARCHAR(255) NOT NULL UNIQUE,
    key_prefix    VARCHAR(10)  NOT NULL,
    rate_limit_rpm INTEGER     DEFAULT 60,
    created_at    TIMESTAMPTZ  DEFAULT CURRENT_TIMESTAMP,
    expires_at    TIMESTAMPTZ,
    is_revoked    BOOLEAN      DEFAULT FALSE
);

-- ---------------------------------------------------------------------------
-- 3. User dimension
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_user (
    user_sk          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id        UUID NOT NULL REFERENCES dim_tenant(tenant_id),
    external_user_id VARCHAR(255) NOT NULL,
    email            VARCHAR(255),
    password_hash    VARCHAR(255),
    created_at       TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (tenant_id, external_user_id)
);

-- ---------------------------------------------------------------------------
-- 4. Movie dimension (SCD Type 2)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_movie (
    movie_sk     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    movie_id     BIGINT       NOT NULL,
    title        VARCHAR(500) NOT NULL,
    genres       VARCHAR(500),
    release_date DATE,
    vote_average NUMERIC(4, 2),
    vote_count   BIGINT,
    popularity   NUMERIC(10, 3),
    is_current   BOOLEAN      DEFAULT TRUE,
    valid_from   TIMESTAMPTZ  DEFAULT CURRENT_TIMESTAMP,
    valid_to     TIMESTAMPTZ  DEFAULT '9999-12-31 23:59:59+00'
);

CREATE INDEX IF NOT EXISTS idx_dim_movie_current ON dim_movie (movie_id) WHERE is_current = TRUE;

-- ---------------------------------------------------------------------------
-- 5. Fact user event
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_user_event (
    event_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id     UUID NOT NULL REFERENCES dim_tenant(tenant_id),
    user_sk       UUID REFERENCES dim_user(user_sk),
    movie_sk      UUID REFERENCES dim_movie(movie_sk),
    event_type    VARCHAR(50)  NOT NULL,
    event_value   NUMERIC(10, 2),
    query_text    VARCHAR(500),
    context_device VARCHAR(100),
    context_os    VARCHAR(100),
    created_at    TIMESTAMPTZ  DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fact_event_time  ON fact_user_event (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fact_event_user  ON fact_user_event (user_sk, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fact_event_movie ON fact_user_event (movie_sk, event_type);

-- ---------------------------------------------------------------------------
-- Seed data
-- ---------------------------------------------------------------------------
INSERT INTO dim_tenant (tenant_id, company_name, plan_tier)
VALUES ('00000000-0000-0000-0000-000000000001', 'Default Public Tenant', 'enterprise')
ON CONFLICT DO NOTHING;
