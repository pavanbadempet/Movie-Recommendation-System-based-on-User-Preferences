-- ==============================================================================
-- V4: Add Stripe billing columns to dim_tenant
-- ==============================================================================
-- Adds stripe_customer_id and subscription_id for Stripe billing integration.
-- plan_tier is already present from V1 — Stripe webhooks will update it.
-- ==============================================================================

ALTER TABLE dim_tenant
    ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS subscription_id    VARCHAR(255);

-- Index for fast webhook lookup by Stripe customer ID
CREATE INDEX IF NOT EXISTS idx_tenant_stripe_customer
    ON dim_tenant (stripe_customer_id)
    WHERE stripe_customer_id IS NOT NULL;

COMMENT ON COLUMN dim_tenant.stripe_customer_id IS
    'Stripe Customer ID (cus_...) — populated on first checkout.session.completed webhook.';

COMMENT ON COLUMN dim_tenant.subscription_id IS
    'Stripe Subscription ID (sub_...) — populated on checkout.session.completed.';
