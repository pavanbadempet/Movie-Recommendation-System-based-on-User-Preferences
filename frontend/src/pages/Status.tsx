/**
 * Public status page — API uptime, serving tier, and SLO metrics.
 * No authentication required. Fetches /health and /v1/platform/slo.
 * Route: /status
 */

import React from "react";
import { Activity, CheckCircle2, AlertTriangle, Loader2, RefreshCw } from "lucide-react";
import { apiGet } from "../api";

interface HealthData {
  status: string;
  movie_count?: number;
  app_version?: string;
  serving_tier?: string | null;
  tier_selection_reason?: string | null;
  hardware_profile?: {
    gpu_available?: boolean;
    ram_gb?: number;
    cpu_cores?: number;
  } | null;
}

interface SloData {
  generated_at?: string;
  window_seconds?: number;
  total_requests?: number;
  error_rate?: number;
  p95_latency_ms?: number;
  routes?: Record<string, { p95_ms?: number; error_rate?: number; count?: number }>;
}

function formatMs(ms?: number | null): string {
  if (ms == null || !Number.isFinite(ms)) return "—";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function formatPercent(rate?: number | null): string {
  if (rate == null || !Number.isFinite(rate)) return "—";
  return `${(rate * 100).toFixed(2)}%`;
}

type OverallStatus = "operational" | "degraded" | "outage" | "loading";

function statusColor(s: OverallStatus): string {
  return s === "operational" ? "green" : s === "degraded" ? "yellow" : s === "outage" ? "red" : "gray";
}

export function StatusPage() {
  const [health, setHealth] = React.useState<HealthData | null>(null);
  const [slo, setSlo] = React.useState<SloData | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [lastChecked, setLastChecked] = React.useState<string | null>(null);

  const overallStatus: OverallStatus = React.useMemo(() => {
    if (loading) return "loading";
    if (!health) return "outage";
    if (health.status !== "ok" && health.status !== "online") return "outage";
    if (slo?.error_rate != null && slo.error_rate > 0.03) return "degraded";
    if (slo?.p95_latency_ms != null && slo.p95_latency_ms > 25000) return "degraded";
    return "operational";
  }, [health, slo, loading]);

  async function fetchStatus() {
    setLoading(true);
    await Promise.allSettled([
      apiGet<HealthData>("/health", {}, 8000).then(({ data }) => setHealth(data)),
      apiGet<SloData>("/v1/platform/slo", {}, 8000).then(({ data }) => setSlo(data)),
    ]);
    setLastChecked(new Date().toLocaleTimeString());
    setLoading(false);
  }

  React.useEffect(() => {
    void fetchStatus();
    const interval = window.setInterval(fetchStatus, 60_000); // refresh every 60 s
    return () => window.clearInterval(interval);
  }, []);

  const tier = health?.serving_tier ?? "unknown";
  const tierLabel =
    tier === "tier1" ? "Tier 1 — GPU Ensemble" :
    tier === "tier2" ? "Tier 2 — ONNX CPU Ensemble" :
    tier === "tier3" ? "Tier 3 — FAISS lite" :
    "Unknown";

  return (
    <main className="status-page" aria-labelledby="status-heading">
      {/* Overall status banner */}
      <section
        className={`status-banner status-${statusColor(overallStatus)}`}
        aria-live="polite"
        aria-label="Overall API status"
      >
        <div className="status-banner-icon" aria-hidden="true">
          {loading ? (
            <Loader2 size={32} className="spin" />
          ) : overallStatus === "operational" ? (
            <CheckCircle2 size={32} />
          ) : (
            <AlertTriangle size={32} />
          )}
        </div>
        <div className="status-banner-text">
          <h1 id="status-heading">
            {loading
              ? "Checking status…"
              : overallStatus === "operational"
              ? "All systems operational"
              : overallStatus === "degraded"
              ? "Partial degradation"
              : "Service disruption"}
          </h1>
          {lastChecked && (
            <p>Last checked: {lastChecked}</p>
          )}
        </div>
        <button
          type="button"
          className="status-refresh-btn"
          onClick={() => void fetchStatus()}
          disabled={loading}
          aria-label="Refresh status"
        >
          <RefreshCw size={18} className={loading ? "spin" : undefined} aria-hidden="true" />
          <span>Refresh</span>
        </button>
      </section>

      {/* Metrics grid */}
      <section className="status-metrics" aria-labelledby="metrics-heading">
        <h2 id="metrics-heading" className="visually-hidden">Current metrics</h2>
        <div className="status-grid" role="list">
          {/* API Health */}
          <article className="status-card" role="listitem" aria-label="API health">
            <div className="status-card-icon" aria-hidden="true">
              <Activity size={20} />
            </div>
            <h3>API status</h3>
            <div
              className={`status-indicator ${health?.status === "ok" || health?.status === "online" ? "green" : "red"}`}
              aria-label={`API status: ${health?.status ?? "unknown"}`}
            >
              {health?.status ?? (loading ? "Checking…" : "Unavailable")}
            </div>
            {health?.movie_count != null && (
              <p className="status-detail">{health.movie_count.toLocaleString()} items in catalog</p>
            )}
            <div className="status-desc">
              Shows if the recommendation engine service is currently online, active, and successfully passing health probes (API status).
            </div>
          </article>

          {/* Serving tier */}
          <article className="status-card" role="listitem" aria-label="Serving tier">
            <div className="status-card-icon" aria-hidden="true">
              <Activity size={20} />
            </div>
            <h3>Serving tier</h3>
            <div
              className={`tier-badge ${tier === "tier1" ? "tier1" : tier === "tier2" ? "tier2" : "tier3"}`}
              aria-label={`Serving tier: ${tierLabel}`}
            >
              {tierLabel}
            </div>
            {health?.tier_selection_reason && (
              <p className="status-detail">{health.tier_selection_reason}</p>
            )}
            <div className="status-desc">
              The active model serving profile (Tier 1 GPU Ensemble, Tier 2 ONNX CPU, or Tier 3 FAISS) currently running in production to handle recommendations.
            </div>
          </article>

          {/* p95 latency */}
          <article className="status-card" role="listitem" aria-label="p95 latency">
            <div className="status-card-icon" aria-hidden="true">
              <Activity size={20} />
            </div>
            <h3>p95 latency</h3>
            <div
              className={`status-metric ${
                slo?.p95_latency_ms != null && slo.p95_latency_ms > 25000 ? "warn" : "good"
              }`}
              aria-label={`p95 latency: ${formatMs(slo?.p95_latency_ms)}`}
            >
              {loading ? "—" : formatMs(slo?.p95_latency_ms)}
            </div>
            <p className="status-detail">SLO: &lt;25 s</p>
            <div className="status-desc">
              The response time threshold experienced by 95% of users, meaning 95% of all requests are processed faster than this latency value.
            </div>
          </article>

          {/* Error rate */}
          <article className="status-card" role="listitem" aria-label="Error rate">
            <div className="status-card-icon" aria-hidden="true">
              <Activity size={20} />
            </div>
            <h3>Error rate</h3>
            <div
              className={`status-metric ${
                slo?.error_rate != null && slo.error_rate > 0.03 ? "warn" : "good"
              }`}
              aria-label={`Error rate: ${formatPercent(slo?.error_rate)}`}
            >
              {loading ? "—" : formatPercent(slo?.error_rate)}
            </div>
            <p className="status-detail">SLO: &lt;3%</p>
            <div className="status-desc">
              The percentage of API calls that failed or encountered server-side errors (HTTP 5xx status codes) within the rolling monitoring window.
            </div>
          </article>

          {/* Request volume */}
          <article className="status-card" role="listitem" aria-label="Request volume">
            <div className="status-card-icon" aria-hidden="true">
              <Activity size={20} />
            </div>
            <h3>Requests (window)</h3>
            <div
              className="status-metric good"
              aria-label={`Total requests in window: ${slo?.total_requests ?? "unknown"}`}
            >
              {loading ? "—" : (slo?.total_requests ?? 0).toLocaleString()}
            </div>
            {slo?.window_seconds && (
              <p className="status-detail">Last {Math.round(slo.window_seconds / 60)} min</p>
            )}
            <div className="status-desc">
              The total volume of incoming requests processed by the recommendation engine during the current active metrics window.
            </div>
          </article>

          {/* Hardware */}
          <article className="status-card" role="listitem" aria-label="Hardware profile">
            <div className="status-card-icon" aria-hidden="true">
              <Activity size={20} />
            </div>
            <h3>Hardware</h3>
            {health?.hardware_profile ? (
              <dl className="hardware-dl">
                <div>
                  <dt>GPU</dt>
                  <dd aria-label={`GPU available: ${health.hardware_profile.gpu_available ? "yes" : "no"}`}>
                    {health.hardware_profile.gpu_available ? "✓" : "✗"}
                  </dd>
                </div>
                <div>
                  <dt>RAM</dt>
                  <dd>{health.hardware_profile.ram_gb?.toFixed(1) ?? "—"} GB</dd>
                </div>
                <div>
                  <dt>CPUs</dt>
                  <dd>{health.hardware_profile.cpu_cores ?? "—"}</dd>
                </div>
              </dl>
            ) : (
              <div className="status-metric" aria-label="Hardware profile not available">
                {loading ? "—" : "N/A"}
              </div>
            )}
            <div className="status-desc">
              The physical system resources (CPU threads, RAM capacity, and GPU core acceleration) currently allocated to run our neural retrieval and search models.
            </div>
          </article>
        </div>
      </section>

      {/* Per-route SLO table */}
      {slo?.routes && Object.keys(slo.routes).length > 0 && (
        <section className="status-routes" aria-labelledby="routes-heading">
          <h2 id="routes-heading">Per-endpoint SLO</h2>
          <div className="table-wrapper" role="region" aria-label="Per-endpoint SLO table">
            <table className="status-table">
              <thead>
                <tr>
                  <th scope="col">Route</th>
                  <th scope="col">p95</th>
                  <th scope="col">Error rate</th>
                  <th scope="col">Requests</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(slo.routes)
                  .sort(([, a], [, b]) => (b.count ?? 0) - (a.count ?? 0))
                  .slice(0, 10)
                  .map(([route, metrics]) => (
                    <tr key={route}>
                      <td>
                        <code>{route}</code>
                      </td>
                      <td
                        className={
                          metrics.p95_ms != null && metrics.p95_ms > 25000 ? "cell-warn" : ""
                        }
                      >
                        {formatMs(metrics.p95_ms)}
                      </td>
                      <td
                        className={
                          metrics.error_rate != null && metrics.error_rate > 0.03 ? "cell-warn" : ""
                        }
                      >
                        {formatPercent(metrics.error_rate)}
                      </td>
                      <td>{(metrics.count ?? 0).toLocaleString()}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <footer className="status-footer">
        <p>
          Data sourced from <code>/health</code> and <code>/v1/platform/slo</code>.
          Auto-refreshes every 60 seconds.
        </p>
        <p>
          Version: {health?.app_version ?? "—"}
        </p>
      </footer>
    </main>
  );
}
