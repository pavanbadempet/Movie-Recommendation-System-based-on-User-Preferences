-- =============================================================================
-- Migration V2: Nova content events table
-- Adds the JSONL-compatible Postgres event store used by the recommendation
-- serving path (backend/events.py).
-- =============================================================================

CREATE TABLE IF NOT EXISTS nova_content_events (
    id          BIGSERIAL    PRIMARY KEY,
    event_id    UUID         NOT NULL DEFAULT uuid_generate_v4() UNIQUE,
    tenant_id   VARCHAR(255),
    catalog_id  VARCHAR(255),
    user_id     VARCHAR(255),
    session_id  VARCHAR(255),
    event_type  VARCHAR(50)  NOT NULL,
    content_id  VARCHAR(255),
    source_content_id VARCHAR(255),
    movie_id    BIGINT,
    query_text  TEXT,
    rating      NUMERIC(4, 2),
    request_id  VARCHAR(255),
    metadata    JSONB,
    event_ts    TIMESTAMPTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_nce_user_ts    ON nova_content_events (user_id, event_ts DESC);
CREATE INDEX IF NOT EXISTS idx_nce_tenant     ON nova_content_events (tenant_id, catalog_id);
CREATE INDEX IF NOT EXISTS idx_nce_event_type ON nova_content_events (event_type, event_ts DESC);
CREATE INDEX IF NOT EXISTS idx_nce_movie      ON nova_content_events (movie_id, event_type);
