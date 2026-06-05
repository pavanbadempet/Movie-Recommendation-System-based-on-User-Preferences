/**
 * Pricing page — three-tier plan cards with Stripe Checkout integration.
 * Route: /pricing
 */

import React from "react";
import { Check, Zap, Shield, Cpu } from "lucide-react";
import { apiPost } from "../api";

type PlanName = "free" | "pro" | "enterprise";

interface PlanCard {
  name: string;
  tier: PlanName;
  price: string;
  period: string;
  tagline: string;
  requestsPerDay: string;
  servingTier: string;
  latency: string;
  support: string;
  features: string[];
  cta: string;
  highlighted: boolean;
  icon: React.ReactNode;
}

const PLANS: PlanCard[] = [
  {
    name: "Free",
    tier: "free",
    price: "$0",
    period: "forever",
    tagline: "Try APEX with no commitment",
    requestsPerDay: "100 requests / day",
    servingTier: "Tier 3",
    latency: "800 – 2,000 ms",
    support: "Community",
    features: [
      "FAISS vector search",
      "TF-IDF semantic matching",
      "52-endpoint REST API",
      "OpenAPI / Swagger docs",
      "Community support",
    ],
    cta: "Get started free",
    highlighted: false,
    icon: <Cpu size={24} />,
  },
  {
    name: "Pro",
    tier: "pro",
    price: "$299",
    period: "/ month",
    tagline: "Full ensemble — production ready",
    requestsPerDay: "10,000 requests / day",
    servingTier: "Tier 2",
    latency: "200 – 800 ms",
    support: "Email (48 h SLA)",
    features: [
      "Everything in Free",
      "6-model ONNX ensemble",
      "Multi-modal (CLIP + SBERT)",
      "Knowledge graph reasoning",
      "LLM explanations (GPT-4o)",
      "Online learning loop",
      "Differential privacy (ε-DP)",
      "Prometheus metrics export",
      "Email support",
    ],
    cta: "Start Pro trial",
    highlighted: true,
    icon: <Zap size={24} />,
  },
  {
    name: "Enterprise",
    tier: "enterprise",
    price: "Custom",
    period: "",
    tagline: "GPU inference + SLA",
    requestsPerDay: "Unlimited",
    servingTier: "Tier 1",
    latency: "50 – 200 ms",
    support: "Dedicated + 4 h SLA",
    features: [
      "Everything in Pro",
      "GPU ensemble (full precision)",
      "torch.compile optimisation",
      "Active inference engine",
      "Multi-tenant isolation",
      "Custom catalog ingestion",
      "Dedicated infrastructure",
      "4-hour SLA",
      "Slack / Teams channel",
    ],
    cta: "Contact sales",
    highlighted: false,
    icon: <Shield size={24} />,
  },
];

interface PricingProps {
  onNavigate?: (page: string) => void;
}

export function PricingPage({ onNavigate }: PricingProps) {
  const [loading, setLoading] = React.useState<PlanName | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  async function handleCta(plan: PlanCard) {
    if (plan.tier === "free") {
      onNavigate?.("signup");
      return;
    }
    if (plan.tier === "enterprise") {
      window.location.href = "mailto:sales@apexrecommend.ai?subject=Enterprise%20Plan%20Inquiry";
      return;
    }

    setLoading(plan.tier);
    setError(null);
    try {
      const { data } = await apiPost<{ checkout_url: string }>(
        "/v1/billing/checkout",
        {
          plan: plan.tier,
          success_url: `${window.location.origin}/dashboard?upgrade=success`,
          cancel_url: `${window.location.origin}/pricing`,
        },
      );
      window.location.href = data.checkout_url;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start checkout. Please try again.");
    } finally {
      setLoading(null);
    }
  }

  return (
    <main className="pricing-page" aria-labelledby="pricing-heading">
      {/* Header */}
      <header className="pricing-header">
        <h1 id="pricing-heading">Simple, transparent pricing</h1>
        <p>
          Netflix-quality recommendations — pick the plan that fits your scale.
          All plans include the full API surface and OpenAPI docs.
        </p>
      </header>

      {/* Benchmark proof */}
      <div className="pricing-proof" role="region" aria-label="Performance benchmarks">
        <div className="proof-stat">
          <strong>0.785</strong>
          <span>HR@10</span>
        </div>
        <div className="proof-stat">
          <strong>0.542</strong>
          <span>NDCG@10</span>
        </div>
        <div className="proof-stat">
          <strong>1.000</strong>
          <span>Semantic HR@10</span>
        </div>
        <div className="proof-stat">
          <strong>52</strong>
          <span>API endpoints</span>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="pricing-error" role="alert" aria-live="polite">
          {error}
        </div>
      )}

      {/* Plan cards */}
      <div className="pricing-cards" role="list">
        {PLANS.map((plan) => (
          <article
            key={plan.tier}
            className={`pricing-card ${plan.highlighted ? "highlighted" : ""}`}
            role="listitem"
            aria-label={`${plan.name} plan`}
          >
            {plan.highlighted && (
              <div className="most-popular-badge" aria-label="Most popular plan">
                Most popular
              </div>
            )}

            <div className="plan-icon" aria-hidden="true">
              {plan.icon}
            </div>

            <h2 className="plan-name">{plan.name}</h2>
            <p className="plan-tagline">{plan.tagline}</p>

            <div className="plan-price" aria-label={`Price: ${plan.price}${plan.period ? " " + plan.period : ""}`}>
              <span className="price-amount">{plan.price}</span>
              {plan.period && <span className="price-period">{plan.period}</span>}
            </div>

            {/* Key specs */}
            <dl className="plan-specs">
              <div className="spec-row">
                <dt>Requests</dt>
                <dd>{plan.requestsPerDay}</dd>
              </div>
              <div className="spec-row">
                <dt>Serving tier</dt>
                <dd>{plan.servingTier}</dd>
              </div>
              <div className="spec-row">
                <dt>Latency</dt>
                <dd>{plan.latency}</dd>
              </div>
              <div className="spec-row">
                <dt>Support</dt>
                <dd>{plan.support}</dd>
              </div>
            </dl>

            {/* Feature list */}
            <ul className="plan-features" aria-label={`${plan.name} plan features`}>
              {plan.features.map((feature) => (
                <li key={feature}>
                  <Check size={14} aria-hidden="true" />
                  <span>{feature}</span>
                </li>
              ))}
            </ul>

            {/* CTA */}
            <button
              className={`plan-cta ${plan.highlighted ? "primary" : "secondary"}`}
              type="button"
              onClick={() => handleCta(plan)}
              disabled={loading !== null}
              aria-busy={loading === plan.tier}
              aria-label={`${plan.cta} — ${plan.name} plan`}
            >
              {loading === plan.tier ? "Redirecting…" : plan.cta}
            </button>
          </article>
        ))}
      </div>

      {/* FAQ */}
      <section className="pricing-faq" aria-labelledby="faq-heading">
        <h2 id="faq-heading">Common questions</h2>
        <dl className="faq-list">
          <div className="faq-item">
            <dt>Can I start on Free and upgrade later?</dt>
            <dd>
              Yes. Upgrade any time from your dashboard. Your API key stays the
              same — only the daily limit and serving tier change.
            </dd>
          </div>
          <div className="faq-item">
            <dt>What happens if I exceed my daily limit?</dt>
            <dd>
              Requests over the limit return HTTP 429 with an upgrade URL.
              No data is lost — your catalog and history are preserved.
            </dd>
          </div>
          <div className="faq-item">
            <dt>Do you offer a trial for Pro?</dt>
            <dd>
              Yes — the first 14 days of Pro are free. No credit card required
              to start. Cancel any time before the trial ends.
            </dd>
          </div>
          <div className="faq-item">
            <dt>Is my data used to train shared models?</dt>
            <dd>
              No. Each tenant&apos;s data is isolated. APEX uses differential privacy
              (ε-DP) on embeddings and never shares interaction data across tenants.
            </dd>
          </div>
        </dl>
      </section>
    </main>
  );
}
