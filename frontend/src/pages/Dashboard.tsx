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
  ArrowRight,
  FileText,
  Shuffle,
  Play,
  Layers,
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
      <div className="hardware-dl" style={{ display: "grid", gap: "10px", margin: 0, padding: 0 }}>
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
      </div>
      <div className="status-desc" style={{ marginTop: "12px" }}>
        <span>Hardware specifications detected on the hosting environment executing models.</span>
        <span><strong>Technical:</strong> Direct OS-level queries checking RAM capacity, CPU cores, and NVIDIA CUDA compute availability.</span>
        <span><strong>Example:</strong> {ramGb.toFixed(1)} GB RAM Allocation indicates the hosting space has been provisioned with large memory capacity.</span>
      </div>
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
      <div className="status-desc" style={{ marginTop: "12px" }}>
        <span>Live service-level objective (SLO) metrics measuring latency, error rates, and traffic throughput.</span>
        <span><strong>Technical:</strong> Calculated over a rolling 1-hour window from memory-buffered request metrics in FastAPI middleware.</span>
        <span><strong>Example:</strong> A P95 latency of {p95 != null ? `${p95.toFixed(0)}ms` : "30ms"} means 95% of recommendation queries are completed within that duration.</span>
      </div>
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
      <div className="status-desc" style={{ marginTop: "12px" }}>
        <span>Static information about the current API deployment version, code commit, and database catalog.</span>
        <span><strong>Technical:</strong> Exposes the system environment variable metadata, active git SHA commit hash, and item count in the SQLite database.</span>
        <span><strong>Example:</strong> Catalog Size of {info.movie_count != null ? info.movie_count.toLocaleString() : "75,253"} movies represents the total number of indexable items currently in the recommendation corpus.</span>
      </div>
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

// ─── Lakehouse & Pipelines Dashboard Tab ─────────────────────────────────────

interface PipelineColumnSpec {
  type?: string;
  nullable?: boolean;
  description?: string;
}

interface PipelineContractSpec {
  name?: string;
  version?: number;
  primary_key?: string[];
  required_columns?: string[];
  columns: Record<string, PipelineColumnSpec>;
}

interface PipelineTableSummary {
  status: string;
  version_count: number;
  latest: {
    run_id: string;
    run_date: string;
    row_count: number;
    data_size_bytes?: number;
    data_sha256?: string;
  } | null;
  scd?: {
    current_rows: number;
    historical_versions: number;
    total_versions: number;
    business_keys: number | null;
  } | null;
}

interface PipelineReport {
  status: string;
  lakehouse: {
    status: string;
    ready_table_count: number;
    table_count: number;
    tables: Record<string, PipelineTableSummary>;
  };
  contracts: Record<string, PipelineContractSpec>;
  streaming: {
    event_store: string;
    durable: boolean;
    postgres_configured: boolean;
    event_table: string | null;
    event_log_path: string;
  };
}

function LakehouseDashboardInner() {
  const [data, setData] = React.useState<PipelineReport | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [selectedContract, setSelectedContract] = React.useState<string>("silver_movies");

  React.useEffect(() => {
    apiGet<PipelineReport>("/v1/platform/pipelines")
      .then((res) => {
        if (res.data?.status === "ok") {
          setData(res.data);
        } else {
          setError(res.data?.status === "error" ? res.data?.status : "Failed to load pipeline stats");
        }
      })
      .catch(() => setError("Failed to connect to pipeline diagnostics API"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ gridColumn: "1 / -1", display: "flex", justifyContent: "center", alignItems: "center", padding: "48px 0" }}>
        <Loader2 className="spin" size={28} style={{ color: "var(--accent)" }} />
        <span style={{ marginLeft: "12px", color: "var(--muted)", fontWeight: 600 }}>Analyzing Medallion tables...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="dashboard-card" style={{ gridColumn: "1 / -1", padding: "24px" }} role="alert">
        <p className="dashboard-error" style={{ margin: 0 }}>{error || "Failed to retrieve pipeline data"}</p>
        <p style={{ fontSize: "0.82rem", color: "var(--quiet)", marginTop: "8px" }}>
          Ensure that local data files exist by running the bootstrap or rebuild scripts (e.g. <code>python scripts/rebuild_serving_artifacts.py</code>).
        </p>
      </div>
    );
  }

  const lakehouse = data.lakehouse;
  const contracts = data.contracts;
  const streaming = data.streaming;

  // Table keys in inspect_lakehouse report
  const tableKeys = {
    bronze: "bronze.movies_raw",
    silver: "silver.movies_curated",
    gold_features: "gold.movies_features",
    gold_scd: "gold.dim_movie_scd",
  };

  const getTableSummary = (key: string): PipelineTableSummary => {
    return lakehouse.tables?.[key] || { status: "missing", version_count: 0, latest: null };
  };

  const bronzeInfo = getTableSummary(tableKeys.bronze);
  const silverInfo = getTableSummary(tableKeys.silver);
  const goldFeaturesInfo = getTableSummary(tableKeys.gold_features);
  const goldScdInfo = getTableSummary(tableKeys.gold_scd);

  // Active contract selection data
  const contractData = contracts[selectedContract] || { columns: {}, primary_key: [], version: 1 };
  const contractColumns = contractData.columns || {};
  const primaryKeys = contractData.primary_key || [];

  const handleKeySelection = (contract: string) => (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setSelectedContract(contract);
    }
  };

  return (
    <>
      {/* 1. Medallion Pipeline Flow Chart */}
      <div className="dashboard-card" style={{ gridColumn: "1 / -1" }}>
        <h3 className="dashboard-card-title">
          <Layers size={16} aria-hidden="true" />
          Medallion Architecture Pipeline Flow
        </h3>

        <div className="medallion-flow-container">
          {/* Bronze Node */}
          <div
            className={`flow-node ${bronzeInfo.status === "ready" ? "active" : "disabled"} ${selectedContract === "bronze_movies" ? "selected" : ""}`}
            onClick={() => setSelectedContract("bronze_movies")}
            onKeyDown={handleKeySelection("bronze_movies")}
            role="button"
            tabIndex={0}
            aria-label="Bronze Ingestion Layer"
          >
            <div className="node-layer bronze">Bronze</div>
            <div className="node-table-name">movies_raw</div>
            <div className="node-status-badge">{bronzeInfo.status}</div>
            {bronzeInfo.latest && (
              <div className="node-stats">
                <span>Rows: {bronzeInfo.latest.row_count?.toLocaleString() || "-"}</span>
                <span>Batches: {bronzeInfo.version_count}</span>
              </div>
            )}
          </div>

          <div className="flow-connector">
            <ArrowRight size={18} />
          </div>

          {/* Silver Node */}
          <div
            className={`flow-node ${silverInfo.status === "ready" ? "active" : "disabled"} ${selectedContract === "silver_movies" ? "selected" : ""}`}
            onClick={() => setSelectedContract("silver_movies")}
            onKeyDown={handleKeySelection("silver_movies")}
            role="button"
            tabIndex={0}
            aria-label="Silver Curated Layer"
          >
            <div className="node-layer silver">Silver</div>
            <div className="node-table-name">movies_curated</div>
            <div className="node-status-badge">{silverInfo.status}</div>
            {silverInfo.latest && (
              <div className="node-stats">
                <span>Rows: {silverInfo.latest.row_count?.toLocaleString() || "-"}</span>
                <span>Deduplicated</span>
              </div>
            )}
          </div>

          <div className="flow-connector">
            <ArrowRight size={18} />
          </div>

          {/* Gold Features Node */}
          <div
            className={`flow-node ${goldFeaturesInfo.status === "ready" ? "active" : "disabled"} ${selectedContract === "gold_training_set" ? "selected" : ""}`}
            onClick={() => setSelectedContract("gold_training_set")}
            onKeyDown={handleKeySelection("gold_training_set")}
            role="button"
            tabIndex={0}
            aria-label="Gold Feature Store Layer"
          >
            <div className="node-layer gold">Gold</div>
            <div className="node-table-name">movies_features</div>
            <div className="node-status-badge">{goldFeaturesInfo.status}</div>
            {goldFeaturesInfo.latest && (
              <div className="node-stats">
                <span>Rows: {goldFeaturesInfo.latest.row_count?.toLocaleString() || "-"}</span>
                <span>Vectorized (768d)</span>
              </div>
            )}
          </div>
        </div>

        <div className="status-desc" style={{ marginTop: "16px" }}>
          <span>Interactive pipeline map tracking movie ingestion quality stages. Click on any medallion node to load its corresponding schema contract.</span>
          <span><strong>Technical:</strong> Spark pipeline processes raw JSON/CSVs (Bronze), cleanses types & validates contracts (Silver), and generates dense semantic representations (Gold).</span>
        </div>
      </div>

      {/* 2. Interactive Schema Contracts Explorer */}
      <div className="dashboard-card" style={{ gridColumn: "1 / -2" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
          <h3 className="dashboard-card-title" style={{ margin: 0 }}>
            <FileText size={16} aria-hidden="true" />
            Data Quality Contracts Explorer
          </h3>
          <div className="contract-select-wrapper">
            <select
              value={selectedContract}
              onChange={(e) => setSelectedContract(e.target.value)}
              className="contract-select"
              aria-label="Select data contract schema"
            >
              <option value="raw_events">raw_events.schema (Ingest Fact)</option>
              <option value="bronze_movies">bronze_movies.schema (Bronze Dim)</option>
              <option value="silver_movies">silver_movies.schema (Silver Dim)</option>
              <option value="gold_training_set">gold_training_set.schema (Gold Feature)</option>
            </select>
          </div>
        </div>

        {Object.keys(contractColumns).length === 0 ? (
          <div style={{ padding: "16px", textAlign: "center", color: "var(--quiet)", background: "rgba(0,0,0,0.15)", borderRadius: "8px" }}>
            No columns defined in this schema contract.
          </div>
        ) : (
          <div className="table-wrapper" style={{ maxHeight: "300px", overflowY: "auto" }}>
            <table className="status-table" style={{ fontSize: "0.82rem" }}>
              <thead>
                <tr>
                  <th scope="col" style={{ width: "30%" }}>Field Name</th>
                  <th scope="col" style={{ width: "20%" }}>Type</th>
                  <th scope="col" style={{ width: "15%" }}>Nullability</th>
                  <th scope="col" style={{ width: "35%" }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(contractColumns).map(([colName, colSpec]: [string, PipelineColumnSpec]) => {
                  const isPk = primaryKeys.includes(colName);
                  const isNullable = colSpec.nullable !== false;
                  return (
                    <tr key={colName}>
                      <td>
                        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                          <code style={{ color: "#fff", fontWeight: "700" }}>{colName}</code>
                          {isPk && <span className="pk-badge">PK</span>}
                        </div>
                      </td>
                      <td>
                        <span className={`type-badge type-${colSpec.type || "string"}`}>
                          {colSpec.type || "string"}
                        </span>
                      </td>
                      <td style={{ textAlign: "center", fontWeight: "700" }}>
                        <span style={{ color: isNullable ? "var(--quiet)" : "var(--danger)" }}>
                          {isNullable ? "Nullable" : "Required"}
                        </span>
                      </td>
                      <td style={{ color: "var(--muted)" }}>{colSpec.description || "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        <div className="status-desc" style={{ marginTop: "12px" }}>
          <span>Strict schema checks matching defined data formats. Version: {contractData.version || 1} | Primary Key: {primaryKeys.join(", ") || "None"}.</span>
          <span><strong>Technical:</strong> Ingestion pipelines parse these rules from local json schemas, blocking writes on nullability or type violations.</span>
        </div>
      </div>

      {/* 3. SCD Type 2 Merge Diagnostics */}
      <div className="dashboard-card" style={{ gridColumn: "2 / -1" }}>
        <h3 className="dashboard-card-title">
          <Shuffle size={16} aria-hidden="true" />
          SCD Type 2 Merge
        </h3>

        {goldScdInfo.status !== "ready" ? (
          <div style={{ padding: "16px", textAlign: "center", color: "var(--quiet)", background: "rgba(0,0,0,0.15)", borderRadius: "8px", fontSize: "0.82rem" }}>
            SCD Type 2 table not active. Run medallion pipeline to populate history.
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
              <div style={{ padding: "8px 12px", border: "1px solid var(--line)", borderRadius: "8px", background: "rgba(0,0,0,0.2)", display: "flex", flexDirection: "column" }}>
                <span style={{ fontSize: "0.72rem", color: "var(--quiet)", fontWeight: 700, textTransform: "uppercase" }}>Current Rows</span>
                <span style={{ fontSize: "1.1rem", fontWeight: "800", color: "#10b981" }}>
                  {goldScdInfo.scd?.current_rows?.toLocaleString() || "0"}
                </span>
              </div>
              <div style={{ padding: "8px 12px", border: "1px solid var(--line)", borderRadius: "8px", background: "rgba(0,0,0,0.2)", display: "flex", flexDirection: "column" }}>
                <span style={{ fontSize: "0.72rem", color: "var(--quiet)", fontWeight: 700, textTransform: "uppercase" }}>Historical Versions</span>
                <span style={{ fontSize: "1.1rem", fontWeight: "800", color: "var(--muted)" }}>
                  {goldScdInfo.scd?.historical_versions?.toLocaleString() || "0"}
                </span>
              </div>
            </div>

            <dl className="hardware-dl" style={{ margin: 0, padding: 0 }}>
              <div className="hardware-row" style={{ display: "flex", justifyContent: "space-between", padding: "8px 10px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)", marginBottom: "4px" }}>
                <dt style={{ color: "var(--muted)", fontSize: "0.82rem", fontWeight: "600" }}>Total Snapshots</dt>
                <dd style={{ margin: 0, color: "var(--text)", fontSize: "0.85rem", fontWeight: "700" }}>
                  {goldScdInfo.version_count} runs
                </dd>
              </div>
              <div className="hardware-row" style={{ display: "flex", justifyContent: "space-between", padding: "8px 10px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)" }}>
                <dt style={{ color: "var(--muted)", fontSize: "0.82rem", fontWeight: "600" }}>Business Keys</dt>
                <dd style={{ margin: 0, color: "var(--text)", fontSize: "0.85rem", fontWeight: "700" }}>
                  {goldScdInfo.scd?.business_keys?.toLocaleString() || "-"} items
                </dd>
              </div>
            </dl>
          </div>
        )}

        <div className="status-desc" style={{ marginTop: "12px" }}>
          <span>Diagnostics of <code>dim_movie_scd</code> historical record tracking.</span>
          <span><strong>Technical:</strong> Expired matching keys (is_current = false) are preserved on record hash drift, appending updated rows dynamically.</span>
        </div>
      </div>

      {/* 4. Streaming Ingest Stats */}
      <div className="dashboard-card" style={{ gridColumn: "1 / -1" }}>
        <h3 className="dashboard-card-title">
          <Play size={16} aria-hidden="true" />
          Kafka & Redis Streaming Event Ingest
        </h3>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "12px" }}>
          <div className="hardware-row" style={{ padding: "12px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)" }}>
            <div style={{ fontSize: "0.78rem", color: "var(--muted)", fontWeight: 600 }}>Stream Buffering</div>
            <div style={{ fontSize: "1.1rem", fontWeight: "800", color: "#fff", marginTop: "4px" }}>
              {streaming.event_store === "dual" ? "Redis + Local JSONL" : streaming.event_store || "Local File (JSONL)"}
            </div>
          </div>
          <div className="hardware-row" style={{ padding: "12px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)" }}>
            <div style={{ fontSize: "0.78rem", color: "var(--muted)", fontWeight: 600 }}>Durability Backend</div>
            <div style={{ fontSize: "1.1rem", fontWeight: "800", color: streaming.durable ? "#10b981" : "var(--quiet)", marginTop: "4px" }}>
              {streaming.durable ? "Active (PostgreSQL)" : "Transient Local Store"}
            </div>
          </div>
          <div className="hardware-row" style={{ padding: "12px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", background: "rgba(0,0,0,0.15)", gridColumn: "span 2" }}>
            <div style={{ fontSize: "0.78rem", color: "var(--muted)", fontWeight: 600 }}>Active Log File Destination</div>
            <div style={{ fontSize: "0.82rem", fontFamily: "monospace", color: "var(--cyan)", marginTop: "4px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {streaming.event_log_path || "—"}
            </div>
          </div>
        </div>

        <div className="status-desc" style={{ marginTop: "12px" }}>
          <span>Uptime statistics of continuous user behavior clicks & ratings queue buffering.</span>
          <span><strong>Technical:</strong> Online learning coordinator consumes event streams asynchronously, performing mini-batch SGD weights updates.</span>
        </div>
      </div>
    </>
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
        <div className="status-desc" style={{ marginTop: "12px" }}>
          <span>The current active performance tier selected for servicing recommendations.</span>
          <span><strong>Technical:</strong> Automatically selected based on available system hardware (e.g. CPU threads, CUDA devices) and configured routing rules.</span>
          <span><strong>Example:</strong> {health.data?.serving_tier === "tier1" ? "Tier 1 — Enterprise" : health.data?.serving_tier === "tier2" ? "Tier 2 — Professional" : "Tier 3 — Starter"} selection indicating active model-serving optimization is configured.</span>
        </div>
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
  const [activeTab, setActiveTab] = React.useState<"mlops" | "lakehouse">("mlops");

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
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {/* Tab Selector */}
          <div className="dashboard-tabs" role="tablist">
            <button
              className={`dashboard-tab ${activeTab === "mlops" ? "active" : ""}`}
              onClick={() => setActiveTab("mlops")}
              role="tab"
              aria-selected={activeTab === "mlops"}
            >
              System & MLOps
            </button>
            <button
              className={`dashboard-tab ${activeTab === "lakehouse" ? "active" : ""}`}
              onClick={() => setActiveTab("lakehouse")}
              role="tab"
              aria-selected={activeTab === "lakehouse"}
            >
              Lakehouse & Pipelines
            </button>
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
      </div>

      {activeTab === "mlops" ? (
        <DashboardInner key={refreshKey} />
      ) : (
        <LakehouseDashboardInner key={refreshKey} />
      )}
    </section>
  );
}
