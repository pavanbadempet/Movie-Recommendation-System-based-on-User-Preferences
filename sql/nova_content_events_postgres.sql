-- Durable behavior-event store for Nova.
-- Use with a Postgres-compatible free tier by setting:
-- NOVA_EVENT_STORE=postgres
-- NOVA_EVENT_DATABASE_URL=postgresql://...

CREATE TABLE IF NOT EXISTS nova_content_events (
    event_id uuid PRIMARY KEY,
    event_ts timestamptz NOT NULL,
    tenant_id text NOT NULL,
    catalog_id text NOT NULL,
    event_type text NOT NULL,
    movie_id bigint,
    content_id text,
    source_content_id text,
    user_id text,
    session_id text,
    query_text text,
    rating double precision,
    request_id text,
    source text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    raw_event jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS nova_content_events_tenant_ts_idx
    ON nova_content_events (tenant_id, catalog_id, event_ts DESC);

CREATE INDEX IF NOT EXISTS nova_content_events_type_ts_idx
    ON nova_content_events (event_type, event_ts DESC);

CREATE INDEX IF NOT EXISTS nova_content_events_user_ts_idx
    ON nova_content_events (user_id, event_ts DESC);

