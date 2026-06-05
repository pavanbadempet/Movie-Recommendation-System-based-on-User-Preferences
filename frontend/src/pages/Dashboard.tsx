import React from "react";
import {
  Activity,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  Loader2,
  RefreshCw,
  Server,
  Zap,
  GitCommit,
  Package,
} from "lucide-react";
import { useHealth } from "../hooks/useHealth";
import { useSlo } from "../hooks/useSlo";
import { apiGet } from "../api";

// ─── Tier Badge ───────────────────────────────────────────────────────────────

function TierBadge({ tier }: { tier: string | null }) {
  if (!tier) return <span className="tier-badge tier-unknown" aria-label="Serving tier unknown">Unknown</span>;
  const map: Record<string, { label: string; cls: string }> = {
    tier1: { label: "Tier 1 — Enterprise", cls: "tier-1" },
    tier2: { label: "Tier 2 — Professional", cls: "tier-2" },
    tier3: { label: "Tier 3 — Starter", cls: "tier-3" },
  };
  const info = map[tier] ?? { label: tier, cls: "tier-unknown" };
  return (
    <span className={`tier-badge ${info.cls}`} aria-label={`Serving tier: ${info.label}`}>
      {info.label}
    </span>
  );
}

// ─── Hardware Card ────────────────────────────────────────────────────────────

function HardwareCard({
  gpuAvailable,
  ramGb,
  cpuCores,
}: {
  gpuAvailable: boolean;
  ramGb: number;
  cpuCores: number;
}) {
  return (
    <div className="dashboard-card" aria-label="Hardware profile">
      <h3 className="dashboard-card-title">
        <HardDrive size={16} aria-hidden="true" />
        Hardware Profile
      </h3>
      <dl className="hardware-dl">
        <div className="hardware-row">
          <dt><Zap size={14} aria-hidden="true" />GPU</dt>
          <dd>
            <span className={`chip ${gpuAvailable ? "chip-success" : "chip-muted"}`}
              aria-label={`GPU ${gpuAvailable ? "available" : "not available"}`}>
              {gpuAvailable ? "Available" : "Not available"}
            </span>
          </dd>
        </div>
        <div className="hardware-row">
          <dt><HardDrive size={14} aria-hidden="true" />RAM</dt>
          <dd aria-label={`${ramGb.toFixed(1)} gigabytes RAM`}>{ramGb.toFixed(1)} GB</dd>
        </div>
        <div className="hardware-row">
          <dt><Cpu size={14} aria-hidden="true" />CPU Cores</dt>
          <dd aria-label={`${cpuCores} CPU cores`}>{cpuCores}</dd>
        </div>
      </dl>
    </div>
  );
}

// ─── SLO Metrics ─────────────────────────────────────────────────────────────

function SloMetrics({
  p95, p99, errorRate, requestRate, uptimeSeconds,
}: {
  p95: number | null | undefined;
  p99: number | null | undefined;
  errorRate: number | null | undefined;
  requestRate: number | null | undefined;
  uptimeSeconds: number | null | undefined;
}) {
  function fmt(value: number | null | undefined, unit: string): string {
    if (value == null || !Number.isFinite(value)) return "—";
    return `${value.toFixed(1)}${unit}`;
  }
  function fmtUptime(s: number | null | undefined): string {
    if (s == null || !Number.isFinite(s)) return "—";
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return h > 0 ? `${h}h ${m}m` : `${m}m`;
  }

  const errorPct = errorRate != null ? errorRate * 100 : null;
  const errorClass = errorPct != null && errorPct > 1 ? "chip-danger" : errorPct != null && errorPct > 0.1 ? "chip-warn" : "chip-success";

  return (
    <div className="dashboard-card" aria-label="SLO metrics">
      <h3 className="dashboard-card-title">
        <Gauge size={16} aria-hidden="true" />
        SLO Metrics
      </h3>
      <dl className="hardware-dl">
        <div className="hardware-row">
          <dt>P95 Latency</dt>
          <dd aria-label={`P95 latency ${fmt(p95, " ms")}`}>{fmt(p95, " ms")}</dd>
        </div>
        <div className="hardware-row">
          <dt>P99 Latency</dt>
          <dd aria-label={`P99 latency ${fmt(p99, " ms")}`}>{fmt(p99, " ms")}</dd>
        </div>
        <div className="hardware-row">
          <dt>Error Rate</dt>
          <dd>
            <span className={`chip ${errorClass}`} aria-label={`Error rate ${fmt(errorPct, "%")}`}>
              {fmt(errorPct, "%")}
            </span>
          </dd>
        </div>
        <div className="hardware-row">
          <dt>Request Rate</dt>
          <dd aria-label={`Request rate ${fmt(requestRate, " req/s")}`}>{fmt(requestRate, " req/s")}</dd>
        </div>
        <div className="hardware-row">
          <dt>Uptime</dt>
          <dd aria-label={`Uptime ${fmtUptime(uptimeSeconds)}`}>{fmtUptime(uptimeSeconds)}</dd>
        </div>
      </dl>
    </div>
  );
}

// ─── Platform Info Card ───────────────────────────────────────────────────────

type PlatformInfo = {
  status?: string;
  movie_count?: number;
  app_version?: string;
  app_commit?: string;
  serving_tier?: string | null;
  hardware_profile?: { gpu_available: boolean; ram_gb: number; cpu_cores: number } | null;
  tier_selection_reason?: string | null;
};

function PlatformInfoCard({ info }: { info: PlatformInfo }) {
  return (
    <div className="dashboard-card" aria-label="Platform information">
      <h3 className="dashboard-card-title">
        <Database size={16} aria-hidden="true" />
        Platform Info
      </h3>
      <dl className="hardware-dl">
        <div className="hardware-row">
          <dt>API Status</dt>
          <dd>
            <span className={`chip ${info.status === "online" ? "chip-success" : "chip-muted"}`}
              aria-label={`API status: ${info.status ?? "unknown"}`}>
              {info.status ?? "Unknown"}
            </span>
          </dd>
        </div>
        {info.movie_count != null && (
          <div className="hardware-row">
            <dt><Database size={14} aria-hidden="true" />Catalog Size</dt>
            <dd aria-label={`${info.movie_count.toLocaleString()} movies`}>
              {info.movie_count.toLocaleString()} movies
            </dd>
          </div>
        )}
        {info.app_version && (
          <div className="hardware-row">
            <dt><Package size={14} aria-hidden="true" />Version</dt>
            <dd aria-label={`App version ${info.app_version}`}>{info.app_version}</dd>
          </div>
        )}
        {info.app_commit && (
          <div className="hardware-row">
            <dt><GitCommit size={14} aria-hidden="true" />Commit</dt>
            <dd aria-label={`Git commit ${info.app_commit}`}>
              <code style={{ fontSize: "0.8rem" }}>{info.app_commit.slice(0, 7)}</code>
            </dd>
          </div>
        )}
      </dl>
    </div>
  );
}

// ─── Inner Dashboard (re-mounts on refresh) ───────────────────────────────────

function DashboardInner() {
  const health = useHealth();
  const slo = useSlo();
  const [platformInfo, setPlatformInfo] = React.useState<PlatformInfo | null>(null);

  React.useEffect(() => {
    apiGet<PlatformInfo>("/health", {}, 10000)
      .then((r) => setPlatformInfo(r.data))
      .catch(() => {});
  }, []);

  return (
    <>
      {/* Degraded banner */}
      {slo.degraded && (
        <div className="degraded-banner" role="alert" aria-live="polite">
          <Activity size={16} aria-hidden="true" />
          SLO endpoint is currently unavailable. Metrics may be stale.
        </div>
      )}

      {/* Serving tier */}
      <div className="dashboard-card" aria-label="Serving tier">
        <h3 className="dashboard-card-title">
          <Server size={16} aria-hidden="true" />
          Serving Tier
        </h3>
        {health.loading ? (
          <Loader2 size={20} className="spin" aria-label="Loading serving tier" />
        ) : health.error ? (
          <p className="dashboard-error" role="alert">{health.error}</p>
        ) : (
          <div className="tier-row">
            <TierBadge tier={health.data?.serving_tier ?? null} />
            {health.data?.tier_selection_reason && (
              <span className="tier-reason" aria-label={`Reason: ${health.data.tier_selection_reason}`}>
                {health.data.tier_selection_reason.replaceAll("_", " ")}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Hardware card */}
      {health.data?.hardware_profile && (
        <HardwareCard
          gpuAvailable={health.data.hardware_profile.gpu_available}
          ramGb={health.data.hardware_profile.ram_gb}
          cpuCores={health.data.hardware_profile.cpu_cores}
        />
      )}

      {/* SLO metrics */}
      {!slo.degraded && (
        <SloMetrics
          p95={slo.data?.p95_latency_ms}
          p99={slo.data?.p99_latency_ms}
          errorRate={slo.data?.error_rate}
          requestRate={slo.data?.request_rate}
          uptimeSeconds={slo.data?.uptime_seconds}
        />
      )}

      {/* Platform info */}
      {platformInfo && <PlatformInfoCard info={platformInfo} />}
    </>
  );
}

// ─── Dashboard Page ───────────────────────────────────────────────────────────

export function Dashboard() {
  const [refreshKey, setRefreshKey] = React.useState(0);
  const [isRefreshing, setIsRefreshing] = React.useState(false);

  function refresh() {
    setIsRefreshing(true);
    setRefreshKey((k) => k + 1);
    setTimeout(() => setIsRefreshing(false), 1500);
  }

  return (
    <section className="dashboard-shell" aria-labelledby="dashboard-heading">
      <div className="dashboard-header">
        <div>
          <h2 id="dashboard-heading">System Dashboard</h2>
          <p className="dashboard-subtitle">Live hardware profile, serving tier, and SLO metrics.</p>
        </div>
        <button
          className="icon-button"
          type="button"
          onClick={refresh}
          aria-label="Refresh dashboard"
          title="Refresh"
        >
          <RefreshCw size={18} className={isRefreshing ? "spin" : undefined} aria-hidden="true" />
        </button>
      </div>
      <DashboardInner key={refreshKey} />
    </section>
  );
}
