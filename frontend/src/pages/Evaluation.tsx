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
  desc: string;
};

const recMetricDescriptions: Record<string, string> = {
  case_pass_rate: "The overall percentage of test scenarios (like sequels or collections) where the correct movie was successfully recommended in the top results (representing the ratio of target hits).",
  case_pass_count: "The total number of evaluation movie cases that successfully returned target sequel or franchise recommendations.",
  good_hit_case_rate: "The percentage of test seed movies that successfully returned at least one highly-relevant recommendation with high similarity.",
  bad_case_rate_at_k: "The percentage of test scenarios where poor or unrelated recommendations (such as duplicate copycats or low-relevance items) leaked into the top results.",
  bad_match_rate_at_k: "The total fraction of irrelevant suggestions returned across all top-K recommendation slots in the benchmark tests.",
  good_recall_at_k: "The proportion of expected sequels or franchise movies that the system successfully found and retrieved in the top-K recommendation list.",
  mrr_at_k: "The average placement of the first correct sequel recommendation (Mean Reciprocal Rank), indicating how close it is to the top recommendation spot.",
  ndcg_at_k: "The position-weighted quality score of recommendations (NDCG), giving more points if correct sequels or related movies are placed at the very top of the list.",
  good_hit_count: "The total cumulative number of highly-relevant sequels or matching movies successfully recommended across all tests.",
  bad_hit_count: "The total cumulative number of irrelevant, duplicate, or copycat recommendations returned during testing.",
  stage_distribution: "The percentage breakdown of which pipeline stages (dense vector index, sparse keyword search, or knowledge graph) produced the final recommendations.",
  explanation_coverage: "The percentage of similar movie recommendations that successfully generated and displayed a clear matching reason for the user."
};

function MetricsTable({ rows, caption }: { rows: MetricRow[]; caption: string }) {
  return (
    <table className="eval-table" aria-label={caption}>
      <caption className="visually-hidden">{caption}</caption>
      <thead>
        <tr>
          <th scope="col" style={{ width: "60%" }}>Metric</th>
          <th scope="col">Value</th>
          <th scope="col">Threshold</th>
          <th scope="col">Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.metric}>
            <td>
              <div className="metric-info">
                <span className="metric-name">{row.metric}</span>
                <span className="metric-desc">{row.desc}</span>
              </div>
            </td>
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

// Ensure good hit count formatting doesn't show raw percentages
function fmtVal(key: string, val: any): string {
  if (typeof val !== "number") return "—";
  if (key === "good_hit_count" || key === "bad_hit_count" || key === "case_pass_count") {
    return val.toString();
  }
  return key.includes("rate") || key.includes("hit") ? pct(val) : dec(val);
}

function dec(v?: number | null): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(3);
}

function semanticRows(report: SemanticBenchmark): MetricRow[] {
  const m = report.metrics ?? {};
  const k = report.k ?? 10;
  return [
    { 
      metric: `NDCG@${k}`, 
      value: dec(m.ndcg_at_k), 
      threshold: "≥ 0.35", 
      pass: m.ndcg_at_k != null ? m.ndcg_at_k >= 0.35 : null,
      desc: "Measures overall search ranking quality, favoring placing the most relevant movies at the very top of results (Normalized Discounted Cumulative Gain)."
    },
    { 
      metric: `MRR@${k}`, 
      value: dec(m.mrr_at_k), 
      threshold: "≥ 0.35", 
      pass: m.mrr_at_k != null ? m.mrr_at_k >= 0.35 : null,
      desc: "Measures search efficiency by calculating how far down the list a user must look to find the first correct match (Mean Reciprocal Rank)."
    },
    { 
      metric: `Hit-Rate@${k}`, 
      value: pct(m.hit_rate_at_k), 
      threshold: "≥ 80%", 
      pass: m.hit_rate_at_k != null ? m.hit_rate_at_k >= 0.8 : null,
      desc: "The percentage of search queries that successfully return at least one correct movie in the top 10 results."
    },
    { 
      metric: `Bad-Match-Rate@${k}`, 
      value: pct(m.bad_match_rate_at_k), 
      threshold: "< 10%", 
      pass: m.bad_match_rate_at_k != null ? m.bad_match_rate_at_k < 0.1 : null,
      desc: "The fraction of top 10 search results that are completely irrelevant or off-topic, indicating search noise."
    },
    { 
      metric: "Explanation Coverage", 
      value: pct(m.explanation_coverage), 
      threshold: undefined, 
      pass: null,
      desc: "The percentage of recommended movies that include a clear, personalized explanation of why they were suggested."
    },
  ];
}

function recBenchmarkRows(report: RecommendationBenchmark): MetricRow[] {
  const m = report.metrics ?? {};
  return Object.entries(m).map(([key, val]) => ({
    metric: key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
    value: fmtVal(key, val),
    threshold: undefined,
    pass: null,
    desc: recMetricDescriptions[key] ?? `Evaluates the '${key}' parameter across benchmark movie recommendation cases.`
  }));
}

function offlineRows(report: OfflineMetrics): MetricRow[] {
  return [
    { 
      metric: "NDCG@10", 
      value: dec(report.ndcg_at_10), 
      threshold: undefined, 
      pass: null,
      desc: "The overall ranking quality of recommendation lists evaluated against historical user rating datasets, prioritizing top positions."
    },
    { 
      metric: "Recall@50", 
      value: dec(report.recall_at_50), 
      threshold: undefined, 
      pass: null,
      desc: "The percentage of historically liked movies that the recommendation system successfully retrieved within the top 50 suggestions."
    },
    { 
      metric: "ILD (Diversity)", 
      value: dec(report.ild), 
      threshold: undefined, 
      pass: null,
      desc: "Intra-List Diversity; measures how varied the recommendations are to ensure users get a healthy mix of genres rather than just one."
    },
    { 
      metric: "Cold-Start NDCG@10", 
      value: dec(report.cold_start_ndcg_at_10), 
      threshold: undefined, 
      pass: null,
      desc: "The recommendation quality score specifically calculated for new or obscure movies that have no user interaction or rating history."
    },
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
