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

type MetricDesc = {
  simple: string;
  technical: string;
  example: string;
};

type MetricRow = {
  metric: string;
  value: string;
  threshold?: string;
  pass?: boolean | null;
  desc: MetricDesc;
};

const recMetricDescriptions: Record<string, MetricDesc> = {
  case_pass_rate: {
    simple: "The percentage of test scenarios where a correct sequel or related franchise movie was successfully recommended in the top results.",
    technical: "The ratio of test cases where at least one target sequel or franchise movie was retrieved in the top-K recommendations to the total number of seed cases evaluated.",
    example: "When evaluating the seed movie 'Toy Story', the test passes if 'Toy Story 2' or 'Toy Story 3' is successfully recommended in the top results."
  },
  case_pass_count: {
    simple: "The total absolute number of test movie cases that successfully recommended their sequels or franchise movies.",
    technical: "The count of seed cases satisfying the sequel/franchise retrieval condition in the top-K recommendations.",
    example: "32 out of 36 seed movie cases successfully recommended their target franchise movies during the run."
  },
  good_hit_case_rate: {
    simple: "The percentage of test movies that successfully returned at least one highly-relevant movie recommendation.",
    technical: "The percentage of evaluated seed movies that retrieved at least one highly-relevant item (overlap of genre, franchise, or collection >= 1) in the top-K list.",
    example: "When evaluating 'Iron Man', this measures how often the system recommends other Marvel Cinematic Universe movies."
  },
  bad_case_rate_at_k: {
    simple: "The percentage of test scenarios where poor, duplicate, or irrelevant copycat movies leaked into the top recommendations.",
    technical: "The percentage of seed cases where the number of bad matches (unrelated movies with low similarity or cheap copycat titles) is greater than zero within the top-K results.",
    example: "If the recommendations for 'Alien' contain 'Alien Abduction' (a cheap copycat movie), it counts as a bad case."
  },
  bad_match_rate_at_k: {
    simple: "The total fraction of irrelevant, duplicate, or cheap copycat movie recommendations returned across all top-K slots.",
    technical: "The total number of bad hits (copycats or low-similarity items) divided by the total number of recommendation slots evaluated (K * number of test cases).",
    example: "If the system returns 300 recommendation slots in total, and 3 of them are cheap copycats, the bad match rate is 1.0%."
  },
  good_recall_at_k: {
    simple: "The proportion of expected sequels or franchise movies that the system successfully found and retrieved in the top results.",
    technical: "The average recall score calculated as the number of retrieved ground-truth target movies divided by the total number of ground-truth target movies (sequels/franchise) for each seed.",
    example: "If a movie has 3 sequels, and the system recommends 2 of them in the top 10, the recall is 66.7%."
  },
  mrr_at_k: {
    simple: "The average placement of the first correct sequel or franchise movie in the recommendation list (indicating how close to #1 it is).",
    technical: "Mean Reciprocal Rank; the average of the reciprocal rank (1/rank) of the first target sequel/franchise movie in the top-K recommendation lists.",
    example: "If the first correct sequel appears at rank #2, the reciprocal rank is 0.5 (1/2). If it appears at rank #1, it is 1.0."
  },
  ndcg_at_k: {
    simple: "The position-weighted quality score of recommendations, giving more points if correct sequels or related movies are placed at the very top.",
    technical: "Normalized Discounted Cumulative Gain; measures the ranking quality by applying a logarithmic position-based discount to the relevance scores of retrieved sequels/franchise movies.",
    example: "Recommending 'Toy Story 2' as the #1 similar movie scores significantly higher than recommending it at rank #10."
  },
  good_hit_count: {
    simple: "The total cumulative number of highly-relevant sequels or franchise movies successfully recommended across all test cases.",
    technical: "The total count of correct target sequels or franchise movies retrieved across all evaluation runs.",
    example: "Across 36 test cases, the system successfully recommended a total of 72 correct sequel/franchise movies."
  },
  bad_hit_count: {
    simple: "The total cumulative number of irrelevant, duplicate, or copycat recommendations returned during testing.",
    technical: "The total count of bad hits (copycats, off-topic, or low-similarity items) identified across all recommendation slots.",
    example: "A bad hit count of 2 indicates that only 2 irrelevant titles leaked into the results across all benchmark runs."
  },
  stage_distribution: {
    simple: "The percentage breakdown of which pipeline stages produced the final recommended movies.",
    technical: "The distribution showing the fraction of final recommendations originating from Dense Vector, Sparse BM25, and Knowledge Graph stages.",
    example: "70% from dense vectors, 20% from sparse keyword matching, and 10% from knowledge graph traversals."
  },
  explanation_coverage: {
    simple: "The percentage of recommended movies that successfully displayed a clear explanation of why they were suggested.",
    technical: "The ratio of recommended movies that generated a valid explanation string explaining their retrieval source to the total recommendations.",
    example: "Showing 'Because you liked Sci-Fi' for 95% of the recommended movies."
  }
};

function MetricsTable({ rows, caption }: { rows: MetricRow[]; caption: string }) {
  return (
    <table className="eval-table" aria-label={caption}>
      <caption className="visually-hidden">{caption}</caption>
      <thead>
        <tr>
          <th scope="col" style={{ width: "65%" }}>Metric</th>
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
                <div className="metric-desc">
                  <span>{row.desc.simple}</span>
                  <span>
                    <strong>Technical:</strong> {row.desc.technical}
                  </span>
                  <span>
                    <strong>Example:</strong> {row.desc.example}
                  </span>
                </div>
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
      desc: {
        simple: "Measures overall search result ranking quality, rewarding the system for placing the most relevant movies at the very top of search results.",
        technical: "Normalized Discounted Cumulative Gain; sums the relevance of results discounted logarithmically by their position in the top-K list, normalized against the ideal ranking.",
        example: "For a search query 'Star Wars', showing 'A New Hope' as the #1 result scores much higher than showing it as the #10 result."
      }
    },
    { 
      metric: `MRR@${k}`, 
      value: dec(m.mrr_at_k), 
      threshold: "≥ 0.35", 
      pass: m.mrr_at_k != null ? m.mrr_at_k >= 0.35 : null,
      desc: {
        simple: "Measures search efficiency by calculating how quickly a user will find their first correct match in the search results list.",
        technical: "Mean Reciprocal Rank; the average of the reciprocal rank (1/rank) of the first highly-relevant ground-truth movie across all test queries.",
        example: "If the first relevant movie appears at spot #3, the reciprocal rank is 0.33 (1/3). If it is at spot #1, it is 1.0 (1/1)."
      }
    },
    { 
      metric: `Hit-Rate@${k}`, 
      value: pct(m.hit_rate_at_k), 
      threshold: "≥ 80%", 
      pass: m.hit_rate_at_k != null ? m.hit_rate_at_k >= 0.8 : null,
      desc: {
        simple: "The percentage of search queries that successfully return at least one correct movie within the top results.",
        technical: "The ratio of test queries where at least one correct ground-truth movie is retrieved within the top-K search results to the total number of test queries.",
        example: "If 90 out of 100 test search queries return the correct movie within the top 10 results, the Hit-Rate is 90%."
      }
    },
    { 
      metric: `Bad-Match-Rate@${k}`, 
      value: pct(m.bad_match_rate_at_k), 
      threshold: "< 10%", 
      pass: m.bad_match_rate_at_k != null ? m.bad_match_rate_at_k < 0.1 : null,
      desc: {
        simple: "The percentage of search results that are completely irrelevant, off-topic, or low-quality (search noise).",
        technical: "The fraction of top-K retrieved movies that fall below a similarity score of 0.22 and have zero keyword overlap in titles, overviews, or genres.",
        example: "A search for 'Finding Nemo' that includes a horror film or an unrelated low-quality cheap copycat in the top 10 results."
      }
    },
    { 
      metric: "Explanation Coverage", 
      value: pct(m.explanation_coverage), 
      threshold: undefined, 
      pass: null,
      desc: {
        simple: "The percentage of recommended or retrieved movies that display a clear explanation of why they match.",
        technical: "The ratio of recommendations with generated explanations (explaining the vector, keyword, or knowledge graph matching source) to the total number of recommendations.",
        example: "Displaying 'Similar themes of space exploration' for 'Interstellar' or 'Genre match: Sci-Fi' for 'The Matrix'."
      }
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
    desc: recMetricDescriptions[key] ?? {
      simple: `Evaluates the ${key.replace(/_/g, " ")} parameter across recommendation benchmark cases.`,
      technical: `Calculates metrics for ${key} using the active recommendation pipeline settings.`,
      example: `Benchmark value of ${fmtVal(key, val)} is recorded.`
    }
  }));
}

function offlineRows(report: OfflineMetrics): MetricRow[] {
  return [
    { 
      metric: "NDCG@10", 
      value: dec(report.ndcg_at_10), 
      threshold: undefined, 
      pass: null,
      desc: {
        simple: "Evaluates how well the recommendation list ranks movies according to historical user ratings, prioritizing matches placed at the very top.",
        technical: "Normalized Discounted Cumulative Gain at rank 10, utilizing historical test set user ratings (typically 1-5 stars) as relevance grades and applying a logarithmic position discount.",
        example: "If a user highly rated 'The Dark Knight' (5 stars) and 'Inception' (4 stars), placing 'The Dark Knight' at rank #1 and 'Inception' at rank #2 yields a much higher score than placing them at ranks #9 and #10."
      }
    },
    { 
      metric: "Recall@50", 
      value: dec(report.recall_at_50), 
      threshold: undefined, 
      pass: null,
      desc: {
        simple: "The percentage of historically liked movies that the recommendation system successfully found within the top 50 suggestions.",
        technical: "The ratio of the number of relevant/liked test set movies retrieved in the top 50 recommendations to the total number of relevant/liked movies in the user's test profile.",
        example: "If a user's hidden test history contains 10 liked movies, and our system retrieves 6 of those movies in the top 50 recommendation list, the Recall@50 is 60% (6/10)."
      }
    },
    { 
      metric: "ILD (Diversity)", 
      value: dec(report.ild), 
      threshold: undefined, 
      pass: null,
      desc: {
        simple: "Measures the variety of genres within the recommendation list to ensure the user gets a well-rounded set of options rather than repetitive titles.",
        technical: "Intra-List Diversity; computed as the average pairwise Jaccard distance between the genre vectors of all movie pairs within the top recommendations.",
        example: "A recommendation list containing a mix of Sci-Fi, Action, Comedy, and Drama movies will have a high diversity score (near 0.8), whereas a list of 10 generic action sequels will score low (near 0.2)."
      }
    },
    { 
      metric: "Cold-Start NDCG@10", 
      value: dec(report.cold_start_ndcg_at_10), 
      threshold: undefined, 
      pass: null,
      desc: {
        simple: "Measures the ranking quality specifically for new, obscure, or low-rating movies that lack historical user interaction data.",
        technical: "NDCG@10 calculated exclusively over test cases for 'cold-start' items (movies with fewer than 5 historical ratings in the dataset), relying heavily on content metadata and semantic features.",
        example: "Recommending relevant movies for a newly released indie documentary based purely on its textual description, director, and genre tags, without any user rating history."
      }
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
