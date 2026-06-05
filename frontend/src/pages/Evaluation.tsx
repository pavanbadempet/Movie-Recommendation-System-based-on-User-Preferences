import React, { useEffect, useState } from "react";
import { BarChart3, CheckCircle2, Loader2, RefreshCw, XCircle } from "lucide-react";
import { apiGet, semanticBenchmark } from "../api";
import type { SemanticBenchmark } from "../types";

// ─── Types ────────────────────────────────────────────────────────────────────

type OfflineMetrics = {
  ndcg_at_10?: number | null;
  recall_at_50?: number | null;
  ild?: number | null;
  cold_start_ndcg_at_10?: number | null;
  generated_at?: string | null;
  [key: string]: unknown;
};

type RecommendationBenchmark = {
  status?: string;
  metrics?: Record<string, number | null>;
  evaluated_case_count?: number;
  k?: number;
  [key: string]: unknown;
};

// ─── Metrics Table ────────────────────────────────────────────────────────────

type MetricRow = {
  metric: string;
  value: string;
  threshold?: string;
  pass?: boolean | null;
};

function MetricsTable({ rows, caption }: { rows: MetricRow[]; caption: string }) {
  return (
    <table className="eval-table" aria-label={caption}>
      <caption className="visually-hidden">{caption}</caption>
      <thead>
        <tr>
          <th scope="col">Metric</th>
          <th scope="col">Value</th>
          <th scope="col">Threshold</th>
          <th scope="col">Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.metric}>
            <td>{row.metric}</td>
            <td>{row.value}</td>
            <td>{row.threshold ?? "—"}</td>
            <td>
              {row.pass === true && (
                <span className="eval-chip pass" aria-label="Pass">
                  <CheckCircle2 size={13} aria-hidden="true" /> Pass
                </span>
              )}
              {row.pass === false && (
                <span className="eval-chip fail" aria-label="Fail">
                  <XCircle size={13} aria-hidden="true" /> Fail
                </span>
              )}
              {row.pass == null && <span className="eval-chip neutral">—</span>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ─── Section wrapper ──────────────────────────────────────────────────────────

function EvalSection({
  title,
  loading,
  error,
  children,
}: {
  title: string;
  loading: boolean;
  error: string | null;
  children: React.ReactNode;
}) {
  return (
    <div className="eval-section" aria-labelledby={`eval-${title.replace(/\s+/g, "-").toLowerCase()}`}>
      <h3
        id={`eval-${title.replace(/\s+/g, "-").toLowerCase()}`}
        className="eval-section-title"
      >
        {title}
      </h3>
      {loading && (
        <div className="eval-loading" role="status" aria-live="polite">
          <Loader2 size={18} className="spin" aria-hidden="true" />
          <span>Loading…</span>
        </div>
      )}
      {!loading && error && (
        <p className="dashboard-error" role="alert">{error}</p>
      )}
      {!loading && !error && children}
    </div>
  );
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function pct(v?: number | null): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function dec(v?: number | null): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(3);
}

function semanticRows(report: SemanticBenchmark): MetricRow[] {
  const m = report.metrics ?? {};
  const k = report.k ?? 10;
  return [
    { metric: `NDCG@${k}`, value: dec(m.ndcg_at_k), threshold: "≥ 0.35", pass: m.ndcg_at_k != null ? m.ndcg_at_k >= 0.35 : null },
    { metric: `MRR@${k}`, value: dec(m.mrr_at_k), threshold: "≥ 0.35", pass: m.mrr_at_k != null ? m.mrr_at_k >= 0.35 : null },
    { metric: `Hit-Rate@${k}`, value: pct(m.hit_rate_at_k), threshold: "≥ 80%", pass: m.hit_rate_at_k != null ? m.hit_rate_at_k >= 0.8 : null },
    { metric: `Bad-Match-Rate@${k}`, value: pct(m.bad_match_rate_at_k), threshold: "< 10%", pass: m.bad_match_rate_at_k != null ? m.bad_match_rate_at_k < 0.1 : null },
    { metric: "Explanation Coverage", value: pct(m.explanation_coverage), threshold: null as unknown as string, pass: null },
  ];
}

function recBenchmarkRows(report: RecommendationBenchmark): MetricRow[] {
  const m = report.metrics ?? {};
  return Object.entries(m).map(([key, val]) => ({
    metric: key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
    value: typeof val === "number" ? (key.includes("rate") || key.includes("hit") ? pct(val) : dec(val)) : "—",
    threshold: undefined,
    pass: null,
  }));
}

function offlineRows(report: OfflineMetrics): MetricRow[] {
  return [
    { metric: "NDCG@10", value: dec(report.ndcg_at_10), threshold: null as unknown as string, pass: null },
    { metric: "Recall@50", value: dec(report.recall_at_50), threshold: null as unknown as string, pass: null },
    { metric: "ILD (Diversity)", value: dec(report.ild), threshold: null as unknown as string, pass: null },
    { metric: "Cold-Start NDCG@10", value: dec(report.cold_start_ndcg_at_10), threshold: null as unknown as string, pass: null },
  ];
}

// ─── Evaluation Page ──────────────────────────────────────────────────────────

export function EvaluationPage() {
  const [semanticData, setSemanticData] = useState<SemanticBenchmark | null>(null);
  const [semanticLoading, setSemanticLoading] = useState(true);
  const [semanticError, setSemanticError] = useState<string | null>(null);

  const [recData, setRecData] = useState<RecommendationBenchmark | null>(null);
  const [recLoading, setRecLoading] = useState(true);
  const [recError, setRecError] = useState<string | null>(null);

  const [offlineData, setOfflineData] = useState<OfflineMetrics | null>(null);
  const [offlineLoading, setOfflineLoading] = useState(true);
  const [offlineError, setOfflineError] = useState<string | null>(null);

  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    setSemanticLoading(true);
    setRecLoading(true);
    setOfflineLoading(true);
    setSemanticError(null);
    setRecError(null);
    setOfflineError(null);

    const results = Promise.allSettled([
      semanticBenchmark(10),
      apiGet<RecommendationBenchmark>("/v1/evaluation/recommendation-benchmark", { k: 10 }, 45000),
      apiGet<OfflineMetrics>("/v1/evaluation/offline-metrics", {}, 15000),
    ]);

    results.then(([sem, rec, offline]) => {
      if (sem.status === "fulfilled") {
        setSemanticData(sem.value.data);
      } else {
        setSemanticError(sem.reason instanceof Error ? sem.reason.message : "Unavailable");
      }
      setSemanticLoading(false);

      if (rec.status === "fulfilled") {
        setRecData(rec.value.data);
      } else {
        setRecError(rec.reason instanceof Error ? rec.reason.message : "Unavailable");
      }
      setRecLoading(false);

      if (offline.status === "fulfilled") {
        setOfflineData(offline.value.data);
      } else {
        setOfflineError(
          offline.reason instanceof Error ? offline.reason.message : "Run scripts/run_offline_evaluation.py first",
        );
      }
      setOfflineLoading(false);
    });
  }, [refreshKey]);

  return (
    <section className="eval-shell" aria-labelledby="eval-heading">
      <div className="dashboard-header">
        <div>
          <h2 id="eval-heading">
            <BarChart3 size={22} aria-hidden="true" />
            Evaluation Metrics
          </h2>
          <p className="dashboard-subtitle">
            Semantic benchmark, recommendation benchmark, and offline evaluation results.
          </p>
        </div>
        <button
          className="icon-button"
          type="button"
          onClick={() => setRefreshKey((k) => k + 1)}
          aria-label="Refresh evaluation metrics"
          title="Refresh"
        >
          <RefreshCw
            size={18}
            className={semanticLoading || recLoading || offlineLoading ? "spin" : undefined}
            aria-hidden="true"
          />
        </button>
      </div>

      <EvalSection title="Semantic Benchmark" loading={semanticLoading} error={semanticError}>
        {semanticData && (
          <>
            <p className="eval-meta">
              {semanticData.evaluated_case_count ?? 0} cases evaluated · Status:{" "}
              <strong>{semanticData.status}</strong>
            </p>
            <MetricsTable rows={semanticRows(semanticData)} caption="Semantic benchmark metrics" />
          </>
        )}
      </EvalSection>

      <EvalSection title="Recommendation Benchmark" loading={recLoading} error={recError}>
        {recData && (
          <>
            <p className="eval-meta">
              {recData.evaluated_case_count ?? 0} cases evaluated · Status:{" "}
              <strong>{recData.status ?? "ok"}</strong>
            </p>
            <MetricsTable
              rows={recBenchmarkRows(recData)}
              caption="Recommendation benchmark metrics"
            />
          </>
        )}
      </EvalSection>

      <EvalSection title="Offline Evaluation" loading={offlineLoading} error={offlineError}>
        {offlineData && (
          <>
            {offlineData.generated_at && (
              <p className="eval-meta">Generated: {offlineData.generated_at}</p>
            )}
            <MetricsTable rows={offlineRows(offlineData)} caption="Offline evaluation metrics" />
          </>
        )}
      </EvalSection>
    </section>
  );
}
