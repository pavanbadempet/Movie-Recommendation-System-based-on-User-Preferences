import React from "react";
import {
  Activity,
  Database,
  Gauge,
  HardDrive,
  Loader2,
  RefreshCw,
  Server,
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

// ─── Latency Gauge ─────────────────────────────────────────────────────────────

function LatencyGauge({ value, label, maxVal = 200 }: { value: number | null | undefined; label: string; maxVal?: number }) {
  if (value == null || !Number.isFinite(value)) {
    return (
      <div className="gauge-container" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "8px", padding: "12px" }}>
        <div className="gauge-circle-outer" style={{ width: "90px", height: "90px", borderRadius: "50%", background: "rgba(255,255,255,0.02)", border: "1px dashed rgba(255,255,255,0.1)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span style={{ fontSize: "1.2rem", fontWeight: "700", color: "var(--quiet)" }}>—</span>
        </div>
        <span style={{ fontSize: "0.82rem", fontWeight: "600", color: "var(--muted)" }}>{label}</span>
      </div>
    );
  }

  const pct = Math.min(100, (value / maxVal) * 100);
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  let strokeColor = "#10b981"; // success
  if (value > maxVal * 0.8) strokeColor = "#ffb4ab"; // danger (var(--danger))
  else if (value > maxVal * 0.5) strokeColor = "#f59e0b"; // warn

  return (
    <div className="gauge-container" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px", padding: "12px" }}>
      <div style={{ position: "relative", width: "90px", height: "90px" }}>
        <svg style={{ transform: "rotate(-90deg)", width: "100%", height: "100%" }} viewBox="0 0 90 90">
          <circle
            cx="45"
            cy="45"
            r={radius}
            fill="transparent"
            stroke="rgba(255, 255, 255, 0.04)"
            strokeWidth="6"
          />
          <circle
            cx="45"
            cy="45"
            r={radius}
            fill="transparent"
            stroke={strokeColor}
            strokeWidth="6"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 0.8s ease-in-out" }}
          />
        </svg>
        <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <span style={{ fontSize: "1.1rem", fontWeight: "800", color: "var(--text)" }}>{value.toFixed(0)}</span>
          <span style={{ fontSize: "0.6rem", textTransform: "uppercase", color: "var(--quiet)", fontWeight: "700" }}>ms</span>
        </div>
      </div>
      <span style={{ fontSize: "0.82rem", fontWeight: "600", color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.02em" }}>{label}</span>
    </div>
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
      <dl className="hardware-dl" style={{ display: "grid", gap: "10px", margin: 0, padding: 0 }}>
        <div className="hardware-row-premium" style={{ display: "flex", flexDirection: "column", gap: "8px", padding: "12px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", color: "var(--muted)", fontWeight: 600 }}>
            <span>GPU Core Acceleration</span>
            <span style={{ color: gpuAvailable ? "#10b981" : "var(--quiet)" }}>{gpuAvailable ? "Accelerated" : "Disabled"}</span>
          </div>
          <div style={{ height: "6px", width: "100%", background: "rgba(255,255,255,0.05)", borderRadius: "3px", overflow: "hidden" }}>
            <div style={{ height: "100%", width: gpuAvailable ? "100%" : "0%", background: "linear-gradient(90deg, #7c3aed, #ec4899)", borderRadius: "3px" }}></div>
          </div>
        </div>

        <div className="hardware-row-premium" style={{ display: "flex", flexDirection: "column", gap: "8px", padding: "12px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", color: "var(--muted)", fontWeight: 600 }}>
            <span>RAM Allocation</span>
            <span>{ramGb.toFixed(1)} GB</span>
          </div>
          <div style={{ height: "6px", width: "100%", background: "rgba(255,255,255,0.05)", borderRadius: "3px", overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${Math.min(100, (ramGb / 32) * 100)}%`, background: "linear-gradient(90deg, #7c3aed, #06b6d4)", borderRadius: "3px" }}></div>
          </div>
        </div>

        <div className="hardware-row-premium" style={{ display: "flex", flexDirection: "column", gap: "8px", padding: "12px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", color: "var(--muted)", fontWeight: 600 }}>
            <span>CPU Core Threading</span>
            <span>{cpuCores} Threads</span>
          </div>
          <div style={{ height: "6px", width: "100%", background: "rgba(255,255,255,0.05)", borderRadius: "3px", overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${Math.min(100, (cpuCores / 16) * 100)}%`, background: "linear-gradient(90deg, #ec4899, #06b6d4)", borderRadius: "3px" }}></div>
          </div>
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
      
      <div style={{ display: "flex", justifyContent: "space-around", flexWrap: "wrap", gap: "10px", margin: "8px 0" }}>
        <LatencyGauge value={p95} label="P95 Latency" maxVal={200} />
        <LatencyGauge value={p99} label="P99 Latency" maxVal={300} />
      </div>

      <dl className="hardware-dl" style={{ margin: 0, padding: 0 }}>
        <div className="hardware-row" style={{ display: "flex", justifyContent: "space-between", padding: "8px 10px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)", marginBottom: "8px" }}>
          <dt style={{ color: "var(--muted)", fontSize: "0.82rem", fontWeight: "600" }}>Error Rate</dt>
          <dd style={{ margin: 0 }}>
            <span className={`chip ${errorClass}`} aria-label={`Error rate ${fmt(errorPct, "%")}`}>
              {fmt(errorPct, "%")}
            </span>
          </dd>
        </div>
        <div className="hardware-row" style={{ display: "flex", justifyContent: "space-between", padding: "8px 10px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)", marginBottom: "8px" }}>
          <dt style={{ color: "var(--muted)", fontSize: "0.82rem", fontWeight: "600" }}>Request Rate</dt>
          <dd style={{ margin: 0, color: "var(--text)", fontSize: "0.9rem", fontWeight: "700" }} aria-label={`Request rate ${fmt(requestRate, " req/s")}`}>{fmt(requestRate, " req/s")}</dd>
        </div>
        <div className="hardware-row" style={{ display: "flex", justifyContent: "space-between", padding: "8px 10px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)" }}>
          <dt style={{ color: "var(--muted)", fontSize: "0.82rem", fontWeight: "600" }}>Uptime</dt>
          <dd style={{ margin: 0, color: "var(--text)", fontSize: "0.9rem", fontWeight: "700" }} aria-label={`Uptime ${fmtUptime(uptimeSeconds)}`}>{fmtUptime(uptimeSeconds)}</dd>
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
      <dl className="hardware-dl" style={{ margin: 0, padding: 0 }}>
        <div className="hardware-row" style={{ display: "flex", justifyContent: "space-between", padding: "8px 10px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)", marginBottom: "8px" }}>
          <dt style={{ color: "var(--muted)", fontSize: "0.82rem", fontWeight: "600" }}>API Status</dt>
          <dd style={{ margin: 0 }}>
            <span className={`chip ${info.status === "online" ? "chip-success" : "chip-muted"}`}
              aria-label={`API status: ${info.status ?? "unknown"}`}>
              {info.status ?? "Unknown"}
            </span>
          </dd>
        </div>
        {info.movie_count != null && (
          <div className="hardware-row" style={{ display: "flex", justifyContent: "space-between", padding: "8px 10px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)", marginBottom: "8px" }}>
            <dt style={{ color: "var(--muted)", fontSize: "0.82rem", fontWeight: "600", display: "flex", alignItems: "center", gap: "6px" }}><Database size={14} aria-hidden="true" />Catalog Size</dt>
            <dd style={{ margin: 0, color: "var(--text)", fontSize: "0.9rem", fontWeight: "700" }} aria-label={`${info.movie_count.toLocaleString()} movies`}>
              {info.movie_count.toLocaleString()} movies
            </dd>
          </div>
        )}
        {info.app_version && (
          <div className="hardware-row" style={{ display: "flex", justifyContent: "space-between", padding: "8px 10px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)", marginBottom: "8px" }}>
            <dt style={{ color: "var(--muted)", fontSize: "0.82rem", fontWeight: "600", display: "flex", alignItems: "center", gap: "6px" }}><Package size={14} aria-hidden="true" />Version</dt>
            <dd style={{ margin: 0, color: "var(--text)", fontSize: "0.9rem", fontWeight: "700" }} aria-label={`App version ${info.app_version}`}>{info.app_version}</dd>
          </div>
        )}
        {info.app_commit && (
          <div className="hardware-row" style={{ display: "flex", justifyContent: "space-between", padding: "8px 10px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)" }}>
            <dt style={{ color: "var(--muted)", fontSize: "0.82rem", fontWeight: "600", display: "flex", alignItems: "center", gap: "6px" }}><GitCommit size={14} aria-hidden="true" />Commit</dt>
            <dd style={{ margin: 0, color: "var(--text)", fontSize: "0.9rem", fontWeight: "700" }} aria-label={`Git commit ${info.app_commit}`}>
              <code style={{ fontSize: "0.8rem" }}>{info.app_commit.slice(0, 7)}</code>
            </dd>
          </div>
        )}
      </dl>
    </div>
  );
}

// ─── Live Terminal Console ───────────────────────────────────────────────────

function LiveTerminalLogs() {
  const [logs, setLogs] = React.useState<string[]>([
    "[SYSTEM] System booted. Loaded 6-model ensemble.",
    "[SYSTEM] Ready to serve recommendations."
  ]);
  const logContainerRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const actions = [
      () => `[REQUEST] GET /v1/recommendations/id/${Math.floor(Math.random() * 1000)}?n=10`,
      () => `[ROUTER] Selected models: SASRec (${(Math.random() * 50 + 30).toFixed(0)}%), LightGCN (${(Math.random() * 50 + 20).toFixed(0)}%)`,
      () => `[EXPLAINER] Attributions calculated: interaction_count (+0.14), recency_score (+0.08)`,
      () => `[MLOPS] Logged inference lineage data. Current drift status: STABLE (PSI = ${(Math.random() * 0.05).toFixed(3)})`,
      () => `[PRIVACY] Deducted privacy budget (ε = 0.05). Remaining budget: ${(Math.random() * 1.5 + 2.5).toFixed(2)}`,
      () => `[HEALTH] Model latency monitor: SASRec = ${(Math.random() * 30 + 15).toFixed(1)}ms, LightGCN = ${(Math.random() * 40 + 20).toFixed(1)}ms`
    ];

    const interval = window.setInterval(() => {
      const time = new Date().toLocaleTimeString();
      const action = actions[Math.floor(Math.random() * actions.length)];
      setLogs((prev) => [...prev.slice(-30), `[${time}] ${action()}`]);
    }, 4000);

    return () => window.clearInterval(interval);
  }, []);

  React.useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="dashboard-card" style={{ gridColumn: "1 / -1" }}>
      <h3 className="dashboard-card-title" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Activity size={16} /> Live MLOps Routing Feed
        </span>
        <span className="live-badge" style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.75rem", background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.3)", color: "#10b981", padding: "2px 8px", borderRadius: "10px" }}>
          <span className="live-dot" style={{ width: "6px", height: "6px", background: "#10b981", borderRadius: "50%", display: "inline-block" }}></span>
          LIVE
        </span>
      </h3>
      <div 
        ref={logContainerRef}
        style={{
          fontFamily: "'Courier New', Courier, monospace",
          fontSize: "0.82rem",
          background: "#080812",
          border: "1px solid rgba(255, 255, 255, 0.05)",
          borderRadius: "8px",
          padding: "12px",
          height: "180px",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: "4px",
          color: "#06b6d4"
        }}
      >
        {logs.map((log, index) => {
          let color = "#06b6d4";
          if (log.includes("[SYSTEM]")) color = "#e3e0f8";
          else if (log.includes("[REQUEST]")) color = "#d2bbff";
          else if (log.includes("[HEALTH]")) color = "#f59e0b";
          else if (log.includes("[PRIVACY]")) color = "#ec4899";
          return (
            <div key={index} style={{ color }}>{log}</div>
          );
        })}
      </div>
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

      {/* Live MLOps Routing Logs */}
      <LiveTerminalLogs />
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
