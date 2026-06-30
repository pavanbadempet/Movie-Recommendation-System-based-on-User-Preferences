/**
 * Landing page — public-facing hero, social proof, how-it-works, feature
 * highlights, pricing teaser, and footer.
 * Route: / (root, unauthenticated)
 */

import React from "react";
import {
  BarChart3,
  Brain,
  CheckCircle2,
  ChevronRight,
  Database,
  Eye,
  GitBranch,
  Lock,
  MessageSquare,
  Network,
  Shield,
  Sparkles,
  Zap,
} from "lucide-react";

interface LandingProps {
  onNavigate: (page: string) => void;
}

// ---------------------------------------------------------------------------
// Feature highlight cards
// ---------------------------------------------------------------------------
const FEATURES = [
  {
    icon: <Brain size={28} />,
    title: "6-Model Ensemble",
    description:
      "SASRec, LightGCN, KAN, Hyperbolic, Diffusion, and Neural ODE — each model corrects the others' blind spots. DR-optimized IPS weights. HR@10 = 0.785.",
  },
  {
    icon: <Eye size={28} />,
    title: "Multi-Modal Search",
    description:
      "Fuses SBERT text (60%) and CLIP visual embeddings (40%) so users find content by mood, aesthetic, or concept — not just keywords.",
  },
  {
    icon: <Network size={28} />,
    title: "Knowledge Graph",
    description:
      "Multi-hop semantic reasoning: User → Liked Theme → New Movie. Surfaces non-obvious connections that collaborative filtering misses.",
  },
  {
    icon: <MessageSquare size={28} />,
    title: "LLM Explanations",
    description:
      'GPT-4o generates "Because you loved X, you\'ll enjoy Y" for every recommendation. Append ?explain=true to any endpoint.',
  },
  {
    icon: <Shield size={28} />,
    title: "Differential Privacy",
    description:
      "Laplace/Gaussian noise on user embeddings ensures ε-DP compliance. Built-in fairness auditor enforces Gini < 0.70 on every response.",
  },
  {
    icon: <Zap size={28} />,
    title: "Adaptive Serving",
    description:
      "Hardware auto-detection selects GPU ensemble (Tier 1), ONNX CPU (Tier 2), or FAISS-lite (Tier 3) at startup. One image, any infrastructure.",
  },
];

// ---------------------------------------------------------------------------
// How it works steps
// ---------------------------------------------------------------------------
const STEPS = [
  {
    number: "01",
    title: "Upload your catalog",
    description:
      "POST a CSV with item_id, title, and description to /v1/catalog/upload. APEX builds FAISS indices and SBERT embeddings automatically.",
    code: `curl -X POST /v1/catalog/upload \\
  -H "X-Nova-API-Key: YOUR_KEY" \\
  -F "file=@catalog.csv"`,
  },
  {
    number: "02",
    title: "Call the recommendation API",
    description:
      "One endpoint returns ranked results from the full 6-model ensemble — or FAISS-only on free tier. No ML code required.",
    code: `curl "/v1/recommendations/id/550?n=10" \\
  -H "X-Nova-API-Key: YOUR_KEY"`,
  },
  {
    number: "03",
    title: "Log events, improve over time",
    description:
      "Send click and rating events. The online learning loop fine-tunes LightGCN embeddings in the background — continuously improving accuracy.",
    code: `curl -X POST /v1/events \\
  -d '{"user_id":"u42","item_id":550,
       "event_type":"rating","rating":4.5}'`,
  },
];

// ---------------------------------------------------------------------------
// Social proof benchmarks
// ---------------------------------------------------------------------------
const BENCHMARKS = [
  { value: "0.785", label: "HR@10", sublabel: "Hit Rate" },
  { value: "0.542", label: "NDCG@10", sublabel: "Ranking Quality" },
  { value: "1.000", label: "Semantic HR@10", sublabel: "Intent Understanding" },
  { value: "+4.3%", label: "Ensemble lift", sublabel: "over best single model" },
  { value: "52", label: "API endpoints", sublabel: "OpenAPI 3.1 spec" },
  { value: "80%+", label: "Test coverage", sublabel: "PBT + mutation testing" },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function LandingPage({ onNavigate }: LandingProps) {
  const [activeLang, setActiveLang] = React.useState<"curl" | "python" | "js">("curl");
  const [heroTab, setHeroTab] = React.useState<"code" | "mobile">("mobile");

  const codeSnippets = {
    curl: `# Get recommendations with LLM explanation
curl "/v1/recommendations/id/550?explain=true" \\
  -H "X-Nova-API-Key: YOUR_KEY"

# Response
{
  "recommendations": [
    {
      "title": "Se7en",
      "score": 0.94,
      "explanation": "Both are David Fincher
        psychological thrillers with morally
        ambiguous protagonists and twist endings."
    }
  ]
}`,
    python: `# Fetch ensemble recommendations in Python
import requests

url = "https://api.apex.ai/v1/recommendations/id/550"
headers = {
    "X-Nova-API-Key": "YOUR_KEY"
}
params = {"explain": "true"}

response = requests.get(url, headers=headers, params=params)
recommendations = response.json()["recommendations"]
print(f"Top recommendation: {recommendations[0]['title']}")`,
    js: `// Fetch recommendations in Node.js
const url = 'https://api.apex.ai/v1/recommendations/id/550?explain=true';
const response = await fetch(url, {
  headers: { 'X-Nova-API-Key': 'YOUR_KEY' }
});

const { recommendations } = await response.json();
console.log(\`Top recommendation: \${recommendations[0].title}\`);`
  };

  return (
    <main className="landing-page" aria-label="APEX product landing page">
      {/* ------------------------------------------------------------------ */}
      {/* Hero                                                                */}
      {/* ------------------------------------------------------------------ */}
      <section className="hero-section" aria-labelledby="hero-heading">
        <div className="hero-content">
          <div className="hero-eyebrow">
            <Sparkles size={16} aria-hidden="true" />
            <span>Production-grade recommendation API</span>
          </div>

          <h1 id="hero-heading" className="hero-headline">
            Netflix-quality recommendations.
            <br />
            <span className="hero-highlight">No ML team required.</span>
          </h1>

          <p className="hero-subheadline">
            APEX is a recommendation API for streaming and media platforms. Plug in
            your catalog, get an API key, and go live in 30 minutes — powered by a
            6-model ensemble with the same architecture Netflix, YouTube, and Amazon
            use at scale.
          </p>

          <div className="hero-actions">
            <button
              className="hero-cta-primary"
              type="button"
              onClick={() => onNavigate("signup")}
              aria-label="Sign up for APEX — free tier available"
            >
              Get started free
              <ChevronRight size={18} aria-hidden="true" />
            </button>
            <button
              className="hero-cta-secondary"
              type="button"
              onClick={() => onNavigate("getting-started")}
              aria-label="View APEX quickstart documentation"
            >
              View quickstart
            </button>
          </div>

          <p className="hero-disclaimer">
            Free tier available · No credit card required · 30-minute setup
          </p>
        </div>

        {/* Dynamic Mobile Preview vs API Code Container */}
        <div className="hero-visual-wrapper" style={{ display: "flex", flexDirection: "column", gap: "12px", alignItems: "stretch", width: "100%" }}>
          <div className="hero-tab-selector" style={{ display: "flex", gap: "8px", margin: "0 auto 8px auto" }}>
            <button
              type="button"
              className={`hero-tab-btn ${heroTab === "mobile" ? "active" : ""}`}
              onClick={() => setHeroTab("mobile")}
              style={{
                background: heroTab === "mobile" ? "rgba(167, 139, 250, 0.12)" : "rgba(255, 255, 255, 0.02)",
                border: "1px solid " + (heroTab === "mobile" ? "rgba(167, 139, 250, 0.3)" : "rgba(255,255,255,0.06)"),
                borderRadius: "20px",
                padding: "6px 14px",
                fontSize: "0.8rem",
                color: heroTab === "mobile" ? "#c084fc" : "var(--muted)",
                fontWeight: "600",
                cursor: "pointer",
                transition: "all 0.2s ease"
              }}
            >
              📱 Mobile App UI
            </button>
            <button
              type="button"
              className={`hero-tab-btn ${heroTab === "code" ? "active" : ""}`}
              onClick={() => setHeroTab("code")}
              style={{
                background: heroTab === "code" ? "rgba(255, 255, 255, 0.08)" : "rgba(255, 255, 255, 0.02)",
                border: "1px solid " + (heroTab === "code" ? "rgba(255, 255, 255, 0.15)" : "rgba(255,255,255,0.06)"),
                borderRadius: "20px",
                padding: "6px 14px",
                fontSize: "0.8rem",
                color: heroTab === "code" ? "#fff" : "var(--muted)",
                fontWeight: "600",
                cursor: "pointer",
                transition: "all 0.2s ease"
              }}
            >
              💻 Developer API
            </button>
          </div>

          {heroTab === "mobile" ? (
            <div className="landing-phone-preview" aria-label="Interactive mobile app preview">
              <div className="dynamic-island" />
              <div className="landing-phone-screen">
                <div className="phone-preview-header">
                  <span className="phone-preview-title">APEX</span>
                  <span style={{ fontSize: "0.6rem", color: "var(--success)", display: "flex", alignItems: "center", gap: "4px" }}>
                    <span style={{ width: "6px", height: "6px", background: "var(--success)", borderRadius: "50%" }}></span> Live Server
                  </span>
                </div>

                <div className="phone-preview-search">
                  <Sparkles size={12} style={{ color: "#a78bfa", flexShrink: 0 }} />
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>David Fincher thrillers...</span>
                </div>

                <div style={{ display: "flex", gap: "6px", overflowX: "hidden", paddingBottom: "4px" }}>
                  <span style={{ fontSize: "0.65rem", padding: "4px 8px", background: "rgba(167, 139, 250, 0.15)", borderRadius: "20px", border: "1px solid rgba(167, 139, 250, 0.3)", color: "#c084fc", whiteSpace: "nowrap" }}>Mindbend</span>
                  <span style={{ fontSize: "0.65rem", padding: "4px 8px", background: "rgba(255,255,255,0.04)", borderRadius: "20px", border: "1px solid rgba(255,255,255,0.08)", color: "var(--muted)", whiteSpace: "nowrap" }}>Noir</span>
                  <span style={{ fontSize: "0.65rem", padding: "4px 8px", background: "rgba(255,255,255,0.04)", borderRadius: "20px", border: "1px solid rgba(255,255,255,0.08)", color: "var(--muted)", whiteSpace: "nowrap" }}>Crime</span>
                </div>

                <div className="phone-preview-card">
                  <img
                    className="phone-preview-poster"
                    src="https://image.tmdb.org/t/p/w500/rPdtOFS5hgg2JMRIvGJA2IZ49aU.jpg"
                    alt="Se7en"
                  />
                  <div className="phone-preview-info">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span className="phone-preview-name">Se7en (1995)</span>
                      <span className="phone-preview-score" style={{ whiteSpace: "nowrap" }}>8.6 ★</span>
                    </div>
                    <span className="phone-preview-meta">Crime, Thriller</span>
                    <span style={{ fontSize: "0.65rem", padding: "2px 6px", background: "rgba(99, 102, 241, 0.12)", color: "#818cf8", borderRadius: "4px", width: "fit-content", fontWeight: "700" }}>
                      Vector recall (94% match)
                    </span>
                  </div>
                </div>

                <div style={{ background: "rgba(167, 139, 250, 0.08)", border: "1px solid rgba(167, 139, 250, 0.15)", borderRadius: "14px", padding: "10px", display: "flex", flexDirection: "column", gap: "4px" }}>
                  <span style={{ fontSize: "0.7rem", fontWeight: "700", color: "#c084fc", display: "flex", alignItems: "center", gap: "4px" }}>
                    <Sparkles size={10} /> CineBot Vibe Check
                  </span>
                  <p className="phone-preview-exp" style={{ margin: 0, lineHeight: "1.3" }}>
                    Both are David Fincher psychological thrillers with morally ambiguous protagonists and twist endings.
                  </p>
                </div>

                <div style={{ marginTop: "auto", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                  <button type="button" style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "10px", padding: "8px", fontSize: "0.7rem", color: "#fff", cursor: "pointer" }}>
                    👍 Feedback
                  </button>
                  <button type="button" style={{ background: "var(--accent)", border: "none", borderRadius: "10px", padding: "8px", fontSize: "0.7rem", color: "#fff", fontWeight: "600", cursor: "pointer" }}>
                    ✨ Recommend
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="hero-code" aria-label="API example code" style={{ margin: 0 }}>
              <div className="code-window-chrome" aria-hidden="true">
                <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                  <span className="dot red" />
                  <span className="dot yellow" />
                  <span className="dot green" />
                </div>
                <div className="code-tabs" style={{ display: "flex", gap: "2px", marginLeft: "16px" }}>
                  <button
                    type="button"
                    className={`code-tab-btn ${activeLang === "curl" ? "active" : ""}`}
                    style={{
                      background: activeLang === "curl" ? "rgba(255,255,255,0.06)" : "transparent",
                      border: "none",
                      color: activeLang === "curl" ? "#e3e0f8" : "#958da1",
                      padding: "4px 10px",
                      borderRadius: "6px",
                      fontSize: "0.75rem",
                      fontWeight: "600",
                      cursor: "pointer"
                    }}
                    onClick={() => setActiveLang("curl")}
                  >
                    cURL
                  </button>
                  <button
                    type="button"
                    className={`code-tab-btn ${activeLang === "python" ? "active" : ""}`}
                    style={{
                      background: activeLang === "python" ? "rgba(255,255,255,0.06)" : "transparent",
                      border: "none",
                      color: activeLang === "python" ? "#e3e0f8" : "#958da1",
                      padding: "4px 10px",
                      borderRadius: "6px",
                      fontSize: "0.75rem",
                      fontWeight: "600",
                      cursor: "pointer"
                    }}
                    onClick={() => setActiveLang("python")}
                  >
                    Python
                  </button>
                  <button
                    type="button"
                    className={`code-tab-btn ${activeLang === "js" ? "active" : ""}`}
                    style={{
                      background: activeLang === "js" ? "rgba(255,255,255,0.06)" : "transparent",
                      border: "none",
                      color: activeLang === "js" ? "#e3e0f8" : "#958da1",
                      padding: "4px 10px",
                      borderRadius: "6px",
                      fontSize: "0.75rem",
                      fontWeight: "600",
                      cursor: "pointer"
                    }}
                    onClick={() => setActiveLang("js")}
                  >
                    Node.js
                  </button>
                </div>
              </div>
              <pre className="code-block" style={{ minHeight: "260px" }}>
                <code key={activeLang} className="fade-in-content">{codeSnippets[activeLang]}</code>
              </pre>
            </div>
          )}
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Social proof — benchmark numbers                                    */}
      {/* ------------------------------------------------------------------ */}
      <section
        className="benchmarks-section"
        aria-labelledby="benchmarks-heading"
      >
        <h2 id="benchmarks-heading" className="visually-hidden">
          Performance benchmarks
        </h2>
        <div className="benchmarks-grid" role="list">
          {BENCHMARKS.map((b) => (
            <div
              key={b.label}
              className="benchmark-card"
              role="listitem"
              aria-label={`${b.label}: ${b.value} — ${b.sublabel}`}
            >
              <strong className="benchmark-value">{b.value}</strong>
              <span className="benchmark-label">{b.label}</span>
              <span className="benchmark-sublabel">{b.sublabel}</span>
            </div>
          ))}
        </div>
        <p className="benchmarks-footnote">
          Evaluated on leave-one-out protocol, 200 users, 100 candidates per
          user. DR-optimized IPS weights. Full methodology in{" "}
          <a href="/docs/MODEL_CARDS.md" target="_blank" rel="noreferrer">
            model cards
          </a>
          .
        </p>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* How it works                                                        */}
      {/* ------------------------------------------------------------------ */}
      <section className="how-it-works" aria-labelledby="how-heading">
        <h2 id="how-heading">Up and running in 30 minutes</h2>
        <p className="section-subheading">
          No ML infrastructure. No model training. No embeddings pipeline to
          maintain. APEX handles all of it.
        </p>

        <ol className="steps-list" aria-label="Setup steps">
          {STEPS.map((step) => (
            <li key={step.number} className="step-item">
              <div className="step-number" aria-hidden="true">
                {step.number}
              </div>
              <div className="step-content">
                <h3>{step.title}</h3>
                <p>{step.description}</p>
                <pre className="step-code" aria-label={`Code example for step ${step.number}`}>
                  <code>{step.code}</code>
                </pre>
              </div>
            </li>
          ))}
        </ol>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Feature highlights                                                  */}
      {/* ------------------------------------------------------------------ */}
      <section className="features-section" aria-labelledby="features-heading">
        <h2 id="features-heading">What powers APEX</h2>
        <p className="section-subheading">
          Every feature is production-ready and accessible through the same
          REST API — no separate SDKs or services to integrate.
        </p>

        <div className="features-grid" role="list">
          {FEATURES.map((feature) => (
            <article
              key={feature.title}
              className="feature-card"
              role="listitem"
            >
              <div className="feature-icon" aria-hidden="true">
                {feature.icon}
              </div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </article>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Pricing teaser                                                      */}
      {/* ------------------------------------------------------------------ */}
      <section className="pricing-teaser" aria-labelledby="pricing-teaser-heading">
        <h2 id="pricing-teaser-heading">Start free, scale when ready</h2>

        <div className="teaser-cards" role="list">
          {/* Free */}
          <div className="teaser-card" role="listitem" aria-label="Free plan">
            <h3>Free</h3>
            <div className="teaser-price">$0</div>
            <ul aria-label="Free plan features">
              <li><CheckCircle2 size={14} aria-hidden="true" /> 100 requests / day</li>
              <li><CheckCircle2 size={14} aria-hidden="true" /> FAISS + TF-IDF</li>
              <li><CheckCircle2 size={14} aria-hidden="true" /> Full API surface</li>
            </ul>
            <button
              type="button"
              className="teaser-cta secondary"
              onClick={() => onNavigate("signup")}
              aria-label="Sign up for free plan"
            >
              Get started
            </button>
          </div>

          {/* Pro */}
          <div className="teaser-card highlighted" role="listitem" aria-label="Pro plan">
            <div className="teaser-badge">Most popular</div>
            <h3>Pro</h3>
            <div className="teaser-price">
              $299 <span>/mo</span>
            </div>
            <ul aria-label="Pro plan features">
              <li><CheckCircle2 size={14} aria-hidden="true" /> 10,000 requests / day</li>
              <li><CheckCircle2 size={14} aria-hidden="true" /> ONNX 6-model ensemble</li>
              <li><CheckCircle2 size={14} aria-hidden="true" /> LLM explanations</li>
            </ul>
            <button
              type="button"
              className="teaser-cta primary"
              onClick={() => onNavigate("pricing")}
              aria-label="View Pro plan details"
            >
              Start free trial
            </button>
          </div>

          {/* Enterprise */}
          <div className="teaser-card" role="listitem" aria-label="Enterprise plan">
            <h3>Enterprise</h3>
            <div className="teaser-price">Custom</div>
            <ul aria-label="Enterprise plan features">
              <li><CheckCircle2 size={14} aria-hidden="true" /> Unlimited requests</li>
              <li><CheckCircle2 size={14} aria-hidden="true" /> GPU ensemble (Tier 1)</li>
              <li><CheckCircle2 size={14} aria-hidden="true" /> 4-hour SLA</li>
            </ul>
            <button
              type="button"
              className="teaser-cta secondary"
              onClick={() => onNavigate("pricing")}
              aria-label="View Enterprise plan details"
            >
              Contact sales
            </button>
          </div>
        </div>

        <p className="pricing-link">
          <button
            type="button"
            className="text-link"
            onClick={() => onNavigate("pricing")}
            aria-label="View full pricing page"
          >
            View full pricing and feature comparison →
          </button>
        </p>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Trust signals                                                       */}
      {/* ------------------------------------------------------------------ */}
      <section className="trust-section" aria-labelledby="trust-heading">
        <h2 id="trust-heading" className="visually-hidden">
          Technical trust signals
        </h2>
        <div className="trust-grid" role="list">
          {[
            { icon: <Lock size={20} />, text: "GDPR-compliant differential privacy (ε-DP)" },
            { icon: <Database size={20} />, text: "Delta Lake bronze / silver / gold pipeline" },
            { icon: <BarChart3 size={20} />, text: "Prometheus + Grafana observability built-in" },
            { icon: <GitBranch size={20} />, text: "80%+ coverage · mutation testing · PBT invariants" },
          ].map((item) => (
            <div key={item.text} className="trust-item" role="listitem">
              <span aria-hidden="true">{item.icon}</span>
              <span>{item.text}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Footer                                                              */}
      {/* ------------------------------------------------------------------ */}
      <footer className="landing-footer" role="contentinfo">
        <div className="footer-brand">
          <strong>APEX</strong>
          <span>Recommendation API</span>
        </div>
        <nav className="footer-links" aria-label="Footer navigation">
          <button type="button" className="text-link" onClick={() => onNavigate("getting-started")}>
            Quickstart
          </button>
          <a href="/docs" target="_blank" rel="noreferrer">
            API docs
          </a>
          <button type="button" className="text-link" onClick={() => onNavigate("pricing")}>
            Pricing
          </button>
          <button type="button" className="text-link" onClick={() => onNavigate("status")}>
            Status
          </button>
          <a
            href="https://github.com/pavanbadempet/Movie-Recommendation-System"
            target="_blank"
            rel="noreferrer"
            aria-label="APEX source code on GitHub (opens in new tab)"
          >
            GitHub
          </a>
        </nav>
        <p className="footer-legal">
          MIT License · Built with PyTorch, FastAPI, FAISS, and React
        </p>
      </footer>
    </main>
  );
}
