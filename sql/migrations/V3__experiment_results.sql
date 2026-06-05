-- =============================================================================
-- Migration V3: Experiment results snapshot table
-- Stores periodic snapshots of A/B experiment metrics so results are
-- queryable without re-scanning the full event log.
-- =============================================================================

CREATE TABLE IF NOT EXISTS experiment_results_snapshot (
    snapshot_id   UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment    VARCHAR(255) NOT NULL,
    variant       VARCHAR(255) NOT NULL,
    snapshot_at   TIMESTAMPTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    events        INTEGER      NOT NULL DEFAULT 0,
    impressions   INTEGER      NOT NULL DEFAULT 0,
    clicks        INTEGER      NOT NULL DEFAULT 0,
    ratings       INTEGER      NOT NULL DEFAULT 0,
    avg_rating    NUMERIC(6, 4),
    ctr           NUMERIC(8, 6),
    p_value       NUMERIC(8, 6),
    significant   BOOLEAN      DEFAULT FALSE,
    metadata      JSONB
);

CREATE INDEX IF NOT EXISTS idx_exp_snapshot_experiment
    ON experiment_results_snapshot (experiment, variant, snapshot_at DESC);
