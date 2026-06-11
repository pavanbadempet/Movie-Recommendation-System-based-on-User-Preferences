-- ==============================================================================
-- MOVIE RECOMMENDATION SYSTEM: PRODUCTION POSTGRESQL SCHEMA
-- ==============================================================================
-- This schema supports multi-tenancy, slowly changing dimensions, and event
-- tracking for the Contextual Bandit and Active Inference engines.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ------------------------------------------------------------------------------
-- 1. TENANT DIMENSION (For B2B SaaS Multi-Tenancy)
-- ------------------------------------------------------------------------------
CREATE TABLE dim_tenant (
    tenant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_name VARCHAR(255) NOT NULL,
    plan_tier VARCHAR(50) NOT NULL DEFAULT 'free', -- free, pro, enterprise
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------------------------
-- 2. API KEY DIMENSION
-- ------------------------------------------------------------------------------
CREATE TABLE dim_api_key (
    api_key_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES dim_tenant(tenant_id),
    api_key_hash VARCHAR(255) NOT NULL UNIQUE,
    key_prefix VARCHAR(10) NOT NULL,
    rate_limit_rpm INTEGER DEFAULT 60, -- Requests per minute
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_revoked BOOLEAN DEFAULT FALSE
);

-- ------------------------------------------------------------------------------
-- 3. USER DIMENSION
-- ------------------------------------------------------------------------------
CREATE TABLE dim_user (
    user_sk UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES dim_tenant(tenant_id),
    external_user_id VARCHAR(255) NOT NULL, -- ID from the tenant's system
    email VARCHAR(255),
    password_hash VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, external_user_id)
);

-- ------------------------------------------------------------------------------
-- 4. MOVIE DIMENSION (SCD Type 2)
-- ------------------------------------------------------------------------------
CREATE TABLE dim_movie (
    movie_sk UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    movie_id BIGINT NOT NULL, -- Original TMDB/MovieLens ID
    title VARCHAR(500) NOT NULL,
    genres VARCHAR(500),
    release_date DATE,
    vote_average NUMERIC(4, 2),
    vote_count BIGINT,
    popularity NUMERIC(10, 3),

    -- SCD2 Tracking Columns
    is_current BOOLEAN DEFAULT TRUE,
    valid_from TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    valid_to TIMESTAMP WITH TIME ZONE DEFAULT '9999-12-31 23:59:59+00'
);

CREATE INDEX idx_dim_movie_current ON dim_movie(movie_id) WHERE is_current = TRUE;

-- ------------------------------------------------------------------------------
-- 5. FACT USER EVENT (The Core Analytics & Bandit Feedback Loop)
-- ------------------------------------------------------------------------------
CREATE TABLE fact_user_event (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES dim_tenant(tenant_id),
    user_sk UUID REFERENCES dim_user(user_sk),
    movie_sk UUID REFERENCES dim_movie(movie_sk),

    event_type VARCHAR(50) NOT NULL, -- 'click', 'rating', 'search', 'watch'
    event_value NUMERIC(10, 2), -- e.g. rating value 4.5, watch percentage 80.0
    query_text VARCHAR(500), -- if search event

    context_device VARCHAR(100),
    context_os VARCHAR(100),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_fact_event_time ON fact_user_event(created_at DESC);
CREATE INDEX idx_fact_event_user ON fact_user_event(user_sk, created_at DESC);
CREATE INDEX idx_fact_event_movie ON fact_user_event(movie_sk, event_type);

-- ------------------------------------------------------------------------------
-- INSERT DEFAULT SEED DATA
-- ------------------------------------------------------------------------------
-- Create the default tenant (e.g. for the free demo app)
INSERT INTO dim_tenant (tenant_id, company_name, plan_tier)
VALUES ('00000000-0000-0000-0000-000000000001', 'Default Public Tenant', 'enterprise')
ON CONFLICT DO NOTHING;

-- Seed a default API key hash for internal testing (api_key: 'test_secret_key')
-- bcrypt hash of 'test_secret_key'
INSERT INTO dim_api_key (tenant_id, api_key_hash, key_prefix, rate_limit_rpm)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    '$2b$12$7k3k0XlK3E5J/A9/B6v/IuD7n4w5K8g7j8G3F2D4S1A2F3G4H5J6K',
    'test_secre',
    1000
) ON CONFLICT DO NOTHING;
