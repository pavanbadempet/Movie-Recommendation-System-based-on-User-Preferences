import React from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bookmark,
  Calendar,
  CheckCircle2,
  Clock3,
  Database,
  Film,
  Gauge,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Server,
  Share2,
  Sparkles,
  Star,
  ThumbsDown,
  ThumbsUp,
  TrendingUp,
  WandSparkles,
  X,
  User,
  LogOut,
  Network,
} from "lucide-react";
import {
  apiGet,
  artifactHealth,
  aiSearch,
  backendLabel,
  currentBackend,
  getMovie,
  getMovieEnriched,
  getRecommendations,
  getUserRecommendations,
  loadTitles,
  pingApi,
  platformReadiness,
  platformStatus,
  recordEvent,
  searchMovies,
  getShowcaseMovies,
  semanticBenchmark,
  checkVideoCacheStatus,
} from "./api";
import type {
  ArtifactHealth,
  EventPayload,
  Movie,
  MovieTitle,
  PlatformReadiness,
  PlatformStatus,
  SemanticBenchmark,
} from "./types";
import "./styles.css";
import "./apex-product.css";
import { AuthPage } from "./AuthPage";
import { ErrorBoundary } from "./ErrorBoundary";
const Dashboard = React.lazy(() => import("./pages/Dashboard").then(m => ({ default: m.Dashboard })));
const KnowledgeGraphPage = React.lazy(() => import("./pages/KnowledgeGraph").then(m => ({ default: m.KnowledgeGraphPage })));
const EvaluationPage = React.lazy(() => import("./pages/Evaluation").then(m => ({ default: m.EvaluationPage })));
const UserProfilePage = React.lazy(() => import("./pages/UserProfile").then(m => ({ default: m.UserProfilePage })));
const AdminPanel = React.lazy(() => import("./pages/AdminPanel").then(m => ({ default: m.AdminPanel })));
const LandingPage = React.lazy(() => import("./pages/Landing").then(m => ({ default: m.LandingPage })));
const SignupPage = React.lazy(() => import("./pages/Signup").then(m => ({ default: m.SignupPage })));
const PricingPage = React.lazy(() => import("./pages/Pricing").then(m => ({ default: m.PricingPage })));
const GettingStartedPage = React.lazy(() => import("./pages/GettingStarted").then(m => ({ default: m.GettingStartedPage })));
const StatusPage = React.lazy(() => import("./pages/Status").then(m => ({ default: m.StatusPage })));
const VectorSpace = React.lazy(() => import("./VectorSpace").then(m => ({ default: m.VectorSpace })));

function SuspenseFallback() {
  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "200px", width: "100%", height: "100%" }}>
      <Loader2 className="spin" size={28} style={{ color: "var(--accent)" }} />
    </div>
  );
}


const imageBase = import.meta.env.VITE_TMDB_IMAGE_BASE || "https://image.tmdb.org/t/p/w500";
const RECENT_STORAGE_KEY = "nova_recent_movies_v2";
const SESSION_STORAGE_KEY = "nova_session_id_v1";
const TITLE_CATALOG_LIMIT = 5000;

type AppPage = "home" | "search" | "vector-space" | "profile" | "dashboard" | "knowledge-graph" | "evaluation" | "admin" | "landing" | "signup" | "pricing" | "getting-started" | "status";
type SearchMode = "title" | "semantic";
type CatalogState = "booting" | "warming" | "ready" | "error";
type ResultsKind = "idle" | "search" | "recommendations";
type SelectionSource = "title_search" | "semantic_search" | "search_result" | "recommendation_card" | "recent_pick";
type FeedbackValue = "positive" | "negative";

function posterUrl(path?: string | null): string {
  if (!path) return "https://placehold.co/500x750/141418/f8fafc?text=Movie";
  if (path.startsWith("http")) return path;
  return `${imageBase}${path}`;
}

function backdropUrl(path?: string | null): string {
  if (!path) return posterUrl(path);
  if (path.startsWith("http")) return path;
  return `https://image.tmdb.org/t/p/original${path}`;
}

function movieYear(movie: Movie): string {
  return movie.release_date?.slice(0, 4) || "TBA";
}

function movieScore(movie: Movie): string {
  const value = Number(movie.vote_average || 0);
  return value > 0 ? value.toFixed(1) : "NR";
}

function formatCount(value?: number | null): string {
  const numeric = Number(value || 0);
  return numeric > 0 ? numeric.toLocaleString() : "0";
}

function compactGenres(genres?: string | null): string {
  if (!genres) return "Catalog";
  return genres
    .split(/[,|]/)
    .map((genre) => genre.trim())
    .filter(Boolean)
    .slice(0, 2)
    .join(" / ");
}

function selectTitleLabel(movie: Movie): string {
  const year = movieYear(movie);
  const titleHasYear = /\(\d{4}\)/.test(movie.title);
  const title = titleHasYear || year === "TBA" ? movie.title : `${movie.title} (${year})`;
  const genres = compactGenres(movie.genres).replaceAll(" / ", ", ");
  return genres === "Catalog" ? title : `${title} - ${genres}`;
}

function confidence(movie: Movie): string {
  const raw = Number(movie.similarity_score);
  if (!Number.isFinite(raw) || raw <= 0) return "Ranked";
  const score = Math.max(1, Math.min(99, Math.round(raw * 100)));
  return `${score}% match`;
}

function matchLabel(stage?: string | null): string {
  const normalized = (stage || "").toLowerCase();
  if (normalized.includes("llm") || normalized.includes("rerank")) return "AI rerank";
  if (normalized.includes("hybrid")) return "Hybrid rank";
  if (normalized.includes("dense") || normalized.includes("vector") || normalized.includes("faiss")) {
    return "Vector recall";
  }
  if (normalized.includes("sparse") || normalized.includes("tfidf") || normalized.includes("lexical")) {
    return "Lexical recall";
  }
  return "Content match";
}

function cleanStage(stage?: string | null): string {
  return matchLabel(stage).replace(" rank", "").replace(" recall", "");
}

function movieReasons(movie: Movie): string[] {
  if (movie.explanation?.length) return movie.explanation.slice(0, 2);
  if (movie.explanation_text) {
    return movie.explanation_text
      .split(/(?<=[.!?])\s+/)
      .map((item) => item.trim())
      .filter(Boolean)
      .slice(0, 2);
  }
  const genres = compactGenres(movie.genres);
  return genres === "Catalog" ? [] : [`Shares ${genres.toLowerCase()} signals.`];
}

function semanticDetails(movie: Movie): Record<string, unknown> | null {
  const signals = movie.retrieval_signals || movie.semantic_signals;
  const details = signals?.semantic_twin_details;
  return details && typeof details === "object" ? (details as Record<string, unknown>) : null;
}

function semanticPercent(movie: Movie): string | null {
  const details = semanticDetails(movie);
  const raw = Number(details?.score ?? movie.retrieval_signals?.semantic_twin);
  if (!Number.isFinite(raw) || raw <= 0) return null;
  return `${Math.max(1, Math.min(99, Math.round(raw * 100)))}% semantic`;
}

function evidenceChips(movie: Movie): string[] {
  const details = semanticDetails(movie);
  if (!details) return [];
  const concepts = Array.isArray(details.shared_concepts) ? details.shared_concepts : [];
  const arcs = Array.isArray(details.shared_emotional_arcs) ? details.shared_emotional_arcs : [];
  const jobs = Array.isArray(details.shared_viewer_jobs) ? details.shared_viewer_jobs : [];
  return [...concepts, ...arcs, ...jobs]
    .map((item) => String(item).replaceAll("_", " "))
    .filter(Boolean)
    .slice(0, 4);
}

function directorLabel(movie: Movie): string {
  if (movie.director) return movie.director;
  const reason = movie.explanation?.find((item) => item.toLowerCase().startsWith("same director"));
  const match = reason?.match(/\(([^)]+)\)/);
  return match?.[1] || "";
}

function dedupeMovies(items: Movie[]): Movie[] {
  const seen = new Set<string>();
  return items.filter((movie) => {
    const key = movie.id ? `id:${movie.id}` : `title:${movie.title.toLowerCase()}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function serviceLabel(url: string): string {
  if (typeof window !== "undefined" && url.replace(/\/+$/, "") === window.location.origin.replace(/\/+$/, "")) {
    return "Same-origin API";
  }
  if (url.includes("hf.space")) return "HF Space API";
  if (url.includes("onrender.com")) return "Render API";
  return backendLabel(url);
}

function loadRecentMovies(): Movie[] {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(RECENT_STORAGE_KEY) || "[]") as Movie[];
    return Array.isArray(parsed) ? parsed.slice(0, 6) : [];
  } catch {
    return [];
  }
}

function saveRecentMovies(movies: Movie[]) {
  try {
    window.localStorage.setItem(RECENT_STORAGE_KEY, JSON.stringify(movies.slice(0, 6)));
  } catch {
    // Local storage is optional; the app still works without it.
  }
}

function createSessionId(): string {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function getSessionId(): string {
  try {
    const existing = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (existing) return existing;
    const next = createSessionId();
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, next);
    return next;
  } catch {
    return createSessionId();
  }
}

function StatusBadge({
  state,
  backend,
}: {
  state: CatalogState;
  backend: string;
}) {
  const isBusy = state === "booting" || state === "warming";
  const Icon = state === "ready" ? CheckCircle2 : state === "error" ? AlertTriangle : Loader2;
  return (
    <div className={`status-badge ${state}`} title={backendLabel(backend)}>
      <Icon size={17} className={isBusy ? "spin" : undefined} />
      <span>{state === "ready" ? "Live" : state === "warming" ? "Warming" : state === "error" ? "Retrying" : "Connecting"}</span>
      <strong>{serviceLabel(backend)}</strong>
    </div>
  );
}

function MetricTile({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="metric-tile">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function healthLabel(health: ArtifactHealth | null): string {
  if (!health) return "Pending";
  if (health.status === "ready") return "Ready";
  if (health.status === "degraded") return "Degraded";
  if (health.status === "unavailable") return "Unavailable";
  return health.status || "Unknown";
}

function checkValue(value: unknown): string {
  if (value === true) return "OK";
  if (value === false) return "Fail";
  if (value === null || value === undefined) return "N/A";
  return String(value);
}

function percentValue(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  return `${Math.round(value * 100)}%`;
}

function decimalValue(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  return value.toFixed(3);
}

function shortId(value?: string | null): string {
  if (!value) return "Session";
  return value.length > 12 ? value.slice(0, 12) : value;
}

function qualityLabel(report: SemanticBenchmark | null): string {
  if (!report) return "Pending";
  if (report.status === "unavailable") return "Unavailable";
  const hitRate = report.metrics?.hit_rate_at_k;
  return typeof hitRate === "number" ? `${Math.round(hitRate * 100)}% hit` : report.status;
}

function titleCaseStatus(status?: string | null): string {
  if (!status) return "Pending";
  return status
    .replaceAll("_", " ")
    .split(" ")
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function readinessLabel(report: PlatformReadiness | null): string {
  if (!report) return "Pending";
  return titleCaseStatus(report.status);
}

function readinessSummaryText(report: PlatformReadiness | null): string {
  if (!report) return "Waiting for strict readiness";
  const summary = report.summary || {};
  const ok = Number(summary.ok_count || 0);
  const total = Number(summary.component_count || report.components?.length || 0);
  const failed = Number(summary.failed_required_count || 0);
  if (report.status === "ready") return `${ok}/${total} checks passing`;
  if (failed > 0) return `${failed} required check${failed === 1 ? "" : "s"} failing`;
  return `${ok}/${total} checks passing`;
}

function ReadinessPanel({
  report,
  loading,
  onRefresh,
}: {
  report: PlatformReadiness | null;
  loading: boolean;
  onRefresh: () => void;
}) {
  const components = report?.components || [];
  const summary = report?.summary || {};
  const visibleComponents = [...components]
    .sort((left, right) => {
      const leftOk = left.status === "ok" ? 1 : 0;
      const rightOk = right.status === "ok" ? 1 : 0;
      if (leftOk !== rightOk) return leftOk - rightOk;
      if (left.required !== right.required) return left.required ? -1 : 1;
      return left.name.localeCompare(right.name);
    })
    .slice(0, 6);

  return (
    <div className={`readiness-panel ${report?.status || "pending"}`}>
      <div className="section-mini-title">
        <span>Platform readiness</span>
        <button className="panel-icon-button" type="button" onClick={onRefresh} title="Refresh readiness">
          <RefreshCw size={14} className={loading ? "spin" : undefined} />
        </button>
      </div>
      <div className="readiness-headline">
        <strong>{readinessLabel(report)}</strong>
        <span>{readinessSummaryText(report)}</span>
      </div>
      <div className="readiness-metrics">
        <div>
          <span>Components</span>
          <strong>
            {Number(summary.ok_count || 0)}/{Number(summary.component_count || components.length || 0)}
          </strong>
        </div>
        <div>
          <span>Required failures</span>
          <strong>{Number(summary.failed_required_count || 0)}</strong>
        </div>
      </div>
      {visibleComponents.length > 0 && (
        <div className="component-list">
          {visibleComponents.map((component) => (
            <div className={`component-row ${component.status}`} key={component.name}>
              <span>{component.name.replaceAll("_", " ")}</span>
              <strong>{titleCaseStatus(component.status)}</strong>
            </div>
          ))}
        </div>
      )}
      {report?.app?.commit && <div className="quiet-line">Revision {report.app.commit.slice(0, 7)}</div>}
    </div>
  );
}

function DiagnosticsPanel({ health }: { health: ArtifactHealth | null }) {
  const checks = health?.checks || {};
  const rows = [
    ["Catalog", checks.metadata_ready],
    ["Vectors", checks.vector_files_ready],
    ["ID map", checks.catalog_vector_aligned],
    ["Semantic twins", checks.semantic_catalog_aligned],
    ["Summary", checks.semantic_summary_aligned],
  ];
  return (
    <div className={`diagnostics-panel ${health?.status || "pending"}`}>
      <div className="section-mini-title">
        <span>Artifact health</span>
        <small>{healthLabel(health)}</small>
      </div>
      <div className="diagnostic-rows">
        {rows.map(([label, value]) => (
          <div className="diagnostic-row" key={String(label)}>
            <span>{label}</span>
            <strong>{checkValue(value)}</strong>
          </div>
        ))}
      </div>
      {health?.run_date && <div className="quiet-line">Snapshot {health.run_date}</div>}
    </div>
  );
}

function QualityPanel({ report }: { report: SemanticBenchmark | null }) {
  const metrics = report?.metrics || {};
  const rows = [
    ["Hit rate", percentValue(metrics.hit_rate_at_k)],
    ["MRR", decimalValue(metrics.mrr_at_k)],
    ["NDCG", decimalValue(metrics.ndcg_at_k)],
    ["Bad matches", percentValue(metrics.bad_match_rate_at_k)],
    ["Explanations", percentValue(metrics.explanation_coverage)],
  ];
  return (
    <div className={`quality-panel ${report?.status || "pending"}`}>
      <div className="section-mini-title">
        <span>Serving quality</span>
        <small>{report?.status || "Pending"}</small>
      </div>
      <div className="diagnostic-rows">
        {rows.map(([label, value]) => (
          <div className="diagnostic-row" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      {report?.evaluated_case_count ? <div className="quiet-line">{report.evaluated_case_count} benchmark cases</div> : null}
    </div>
  );
}

const MoviePoster = React.memo(function MoviePoster({
  movie,
  onSelect,
}: {
  movie: Movie;
  onSelect: (movie: Movie) => void;
}) {
  return (
    <button className="poster-card" type="button" onClick={() => onSelect(movie)} title={movie.title}>
      <img src={posterUrl(movie.poster_path)} alt={movie.title} loading="lazy" />
    </button>
  );
});

const RecommendationCard = React.memo(function RecommendationCard({
  movie,
  rank,
  onSelect,
  feedback,
  onFeedback,
}: {
  movie: Movie;
  rank?: number;
  onSelect: (movie: Movie) => void;
  feedback?: FeedbackValue;
  onFeedback: (movie: Movie, value: FeedbackValue) => void;
}) {
  const reasons = movieReasons(movie);
  const semantic = semanticPercent(movie);
  const chips = evidenceChips(movie);
  return (
    <article className="recommendation-card">
      <div className="card-media">
        <MoviePoster movie={movie} onSelect={onSelect} />
        {rank ? <span className="rank-pill">#{rank}</span> : null}
      </div>
      <div className="recommendation-body">
        <div className="card-title-row">
          <strong>{movie.title}</strong>
          <span>{movieYear(movie)}</span>
        </div>
        <div className="signal-row">
          <span>
            <Gauge size={14} />
            {confidence(movie)}
          </span>
          <span>
            <Star size={14} fill="currentColor" />
            {movieScore(movie)}
          </span>
          {semantic && (
            <span>
              <Sparkles size={14} />
              {semantic}
            </span>
          )}
        </div>
        <div className="genre-line">{compactGenres(movie.genres)}</div>
        {/* Retrieval stage badge — Requirements 7.1, 7.2 */}
        {movie.retrieval_stage && (
          <div className="retrieval-stage-badge" aria-label={`Retrieved via ${movie.retrieval_stage}`}>
            {cleanStage(movie.retrieval_stage)}
          </div>
        )}
        {/* Retrieval signals — Requirements 7.3 */}
        {movie.retrieval_signals && Object.keys(movie.retrieval_signals).length > 0 && (
          <dl className="retrieval-signals-dl" aria-label="Retrieval signals">
            {Object.entries(movie.retrieval_signals)
              .filter(([, v]) => v != null && typeof v !== "object")
              .slice(0, 3)
              .map(([k, v]) => (
                <div key={k} className="retrieval-signal-row">
                  <dt>{k.replaceAll("_", " ")}</dt>
                  <dd>{typeof v === "number" ? (v as number).toFixed(3) : String(v)}</dd>
                </div>
              ))}
          </dl>
        )}
        {/* Explanation text — Requirements 7.4 */}
        {movie.explanation_text && (
          <p className="card-explanation" aria-label="AI explanation">{movie.explanation_text}</p>
        )}
        {chips.length > 0 && (
          <div className="evidence-chips">
            {chips.map((chip) => (
              <span key={chip}>{chip}</span>
            ))}
          </div>
        )}
        {reasons.length > 0 && <p>{reasons[0]}</p>}
        <div className="feedback-row" role="group" aria-label={`Feedback for ${movie.title}`}>
          <button
            className={feedback === "positive" ? "active positive" : ""}
            type="button"
            title="More like this"
            onClick={() => onFeedback(movie, "positive")}
          >
            <ThumbsUp size={14} />
            <span>More</span>
          </button>
          <button
            className={feedback === "negative" ? "active negative" : ""}
            type="button"
            title="Less like this"
            onClick={() => onFeedback(movie, "negative")}
          >
            <ThumbsDown size={14} />
            <span>Less</span>
          </button>
        </div>
      </div>
    </article>
  );
});

function ResultContextBar({
  kind,
  backend,
  sourceMovie,
  requestId,
  query,
}: {
  kind: ResultsKind;
  backend: string;
  sourceMovie: Movie | null;
  requestId: string | null;
  query: string;
}) {
  const sourceLabel =
    kind === "recommendations"
      ? sourceMovie?.title || "Selected title"
      : query.trim() || "Catalog query";
  return (
    <div className="result-context" aria-label="Result context">
      <span>
        <Film size={14} />
        {sourceLabel}
      </span>
      {kind === "recommendations" && (
        <span>
          <Sparkles size={14} />
          {shortId(requestId)}
        </span>
      )}
      <span>
        <Server size={14} />
        {serviceLabel(backend)}
      </span>
    </div>
  );
}

const MovieSpotlight = React.memo(function MovieSpotlight({
  movie,
  loading,
  onRecommend,
  userId,
  sessionId,
}: {
  movie: Movie;
  loading: boolean;
  onRecommend: () => void;
  userId: string | null;
  sessionId: string;
}) {
  const [likedStatus, setLikedStatus] = React.useState<"none" | "liked" | "disliked">("none");
  const reasons = movieReasons(movie);
  const semantic = semanticPercent(movie);
  const chips = evidenceChips(movie);
  return (
    <section
      className="spotlight"
      style={{ "--poster-bg": `url(${posterUrl(movie.poster_path)})` } as React.CSSProperties}
    >
      <div className="poster-column">
        <img className="detail-poster" src={posterUrl(movie.poster_path)} alt={movie.title} />
        <div className="poster-stat">
          <Star size={16} fill="currentColor" />
          <span>{movieScore(movie)}</span>
          <small>{formatCount(movie.vote_count)} votes</small>
        </div>
      </div>
      <div className="spotlight-copy">
        <div className="eyebrow">
          <Film size={15} />
          {compactGenres(movie.genres)}
        </div>
        <h1>{movie.title}</h1>
        <div className="meta-row">
          <span>
            <Clock3 size={15} />
            {movieYear(movie)}
          </span>
          <span>
            <TrendingUp size={15} />
            Popularity {formatCount(movie.popularity)}
          </span>
          <span>
            <WandSparkles size={15} />
            {cleanStage(movie.retrieval_stage)}
          </span>
          {semantic && (
            <span>
              <Sparkles size={15} />
              {semantic}
            </span>
          )}
        </div>
        <p>{movie.overview || "No overview is available for this title."}</p>

        <div className="interaction-panel">
          <button
            className={`interaction-btn ${likedStatus === "liked" ? "active-like" : ""}`}
            title="Like this movie"
            onClick={async () => {
              try {
                await recordEvent({
                  user_id: userId ?? undefined,
                  event_type: "rating",
                  movie_id: movie.id,
                  rating: 5.0,
                  session_id: sessionId,
                });
                setLikedStatus("liked");
              } catch (e) {
                console.error("Failed to rate", e);
              }
            }}
          >
            <ThumbsUp size={16} fill={likedStatus === "liked" ? "currentColor" : "none"} /> Like
          </button>
          <button
            className={`interaction-btn ${likedStatus === "disliked" ? "active-dislike" : ""}`}
            title="Dislike this movie"
            onClick={async () => {
              try {
                await recordEvent({
                  user_id: userId ?? undefined,
                  event_type: "rating",
                  movie_id: movie.id,
                  rating: 1.0,
                  session_id: sessionId,
                });
                setLikedStatus("disliked");
              } catch (e) {
                console.error("Failed to rate", e);
              }
            }}
          >
            <ThumbsDown size={16} fill={likedStatus === "disliked" ? "currentColor" : "none"} /> Dislike
          </button>
        </div>

        {reasons.length > 0 && (
          <div className="reason-panel">
            <strong>Why it matched</strong>
            {reasons.map((reason) => (
              <span key={reason}>{reason}</span>
            ))}
            {chips.length > 0 && (
              <div className="evidence-chips wide">
                {chips.map((chip) => (
                  <span key={chip}>{chip}</span>
                ))}
              </div>
            )}
          </div>
        )}
        <div className="action-row">
          <button className="primary-action" type="button" onClick={onRecommend} disabled={loading}>
            {loading ? <Loader2 size={18} className="spin" /> : <Sparkles size={18} />}
            {loading ? "Getting similar recommendations" : "Get Similar Recommendations"}
          </button>
          {movie.trailer_key && (
            <a className="ghost-action" href={`https://www.youtube.com/watch?v=${movie.trailer_key}`} target="_blank" rel="noreferrer">
              <Play size={18} />
              Trailer
            </a>
          )}
        </div>
      </div>
    </section>
  );
});

function TrailerFrame({ movie }: { movie: Movie }) {
  const [playing, setPlaying] = React.useState(true);
  const [trailerKey, setTrailerKey] = React.useState<string | null>(movie.trailer_key || null);
  const [videoError, setVideoError] = React.useState(false);
  const [isCached, setIsCached] = React.useState<boolean | null>(null);
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const [isMobile, setIsMobile] = React.useState(() => typeof window !== "undefined" ? window.innerWidth <= 768 : false);
  const [showFallbackIframe, setShowFallbackIframe] = React.useState(false);
  const [isTabVisible, setIsTabVisible] = React.useState(true);

  React.useEffect(() => {
    if (typeof document === "undefined") return;
    const handleVisibility = () => {
      const visible = document.visibilityState === "visible";
      setIsTabVisible(visible);
      if (!visible && videoRef.current) {
        videoRef.current.pause();
      }
    };
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, []);

  React.useEffect(() => {
    const currentVideo = videoRef.current;
    return () => {
      if (currentVideo) {
        currentVideo.pause();
      }
    };
  }, []);

  React.useEffect(() => {
    setShowFallbackIframe(false);
    if (trailerKey) {
      const timer = setTimeout(() => {
        setShowFallbackIframe(true);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [movie.id, trailerKey]);

  React.useEffect(() => {
    setTrailerKey(movie.trailer_key || null);
    if (!movie.trailer_key) {
      getMovieEnriched(movie.id)
        .then((res) => {
          if (res.data.trailer_key) setTrailerKey(res.data.trailer_key);
        })
        .catch(() => {});
    }
  }, [movie.id, movie.trailer_key]);

  // Check if trailer is cached when trailerKey changes
  React.useEffect(() => {
    if (!trailerKey) {
      setIsCached(null);
      return;
    }
    setIsCached(null); // Reset when key changes
    checkVideoCacheStatus(trailerKey)
      .then((res) => {
        setIsCached(res.data.cached);
      })
      .catch(() => {
        setIsCached(false); // Default to false (immediate iframe) if endpoint fails
      });
  }, [trailerKey]);

  // Handle window resizing to toggle video playback on mobile
  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Restart playback and reload video when movie, trailerKey, or isCached changes
  React.useEffect(() => {
    setPlaying(true);
    setVideoError(false); // Reset error state when movie changes
    if (isCached && videoRef.current && !isMobile) {
      videoRef.current.load();
      videoRef.current.play().catch(() => {});
    }
  }, [movie.id, trailerKey, isCached, isMobile]);

  function togglePlayback() {
    const nextPlaying = !playing;
    if (videoRef.current) {
      if (nextPlaying) {
        videoRef.current.play().catch(() => {});
      } else {
        videoRef.current.pause();
      }
    }
    setPlaying(nextPlaying);
  }

  const backendUrl = currentBackend();
  const videoSrc = trailerKey ? `${backendUrl}/v1/videos/stream/${trailerKey}` : "";

  return (
    <div className="trailer-frame">
      {!isTabVisible ? (
        <>
          <img src={backdropUrl(movie.poster_path)} alt="" style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />
          <div className="trailer-overlay" />
        </>
      ) : trailerKey ? (
        isMobile ? (
          /* Mobile: YouTube iframe embed with proper widescreen fitting */
          <div style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden", background: "#000" }}>
            <iframe
              src={`https://www.youtube.com/embed/${trailerKey}?autoplay=1&mute=1&controls=1&modestbranding=1&rel=0&playsinline=1`}
              title={`${movie.title} Trailer`}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
                display: "block",
                border: "none",
                zIndex: 3,
              }}
            />
          </div>
        ) : (
          /* Desktop: cached video first, YouTube iframe fallback */
          isCached === null ? (
            <div className="trailer-loading" style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#94a3b8" }}>
              <Loader2 className="spin" size={24} />
            </div>
          ) : (!isCached || videoError) ? (
            <div style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden", background: "#000" }}>
              <iframe
                src={`https://www.youtube.com/embed/${trailerKey}?autoplay=1&mute=1&controls=1&modestbranding=1&rel=0&playsinline=1`}
                title={`${movie.title} Trailer`}
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  display: "block",
                  border: "none",
                  zIndex: 3,
                }}
              />
            </div>
          ) : (
            <>
              <video
                ref={videoRef}
                src={videoSrc}
                autoPlay
                muted
                loop
                playsInline
                poster={backdropUrl(movie.poster_path)}
                onError={() => setVideoError(true)}
                style={{ width: "100%", height: "100%", objectFit: "cover", display: "block", pointerEvents: "none" }}
              />
              <div className="trailer-overlay" />
            </>
          )
        )
      ) : (
        /* No trailer key — static poster */
        <>
          <img src={backdropUrl(movie.poster_path)} alt="" style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />
          <div className="trailer-overlay" />
        </>
      )}
      {trailerKey && isCached && !videoError && !isMobile && (
        <button className="video-toggle" type="button" onClick={togglePlayback} aria-label={playing ? "Pause trailer" : "Play trailer"}>
          <span className="visually-hidden">{playing ? "Pause trailer" : "Play trailer"}</span>
        </button>
      )}
    </div>
  );
}

function RatingCircle({ score }: { score: string }) {
  const numScore = score === "NR" ? 0 : Number(score) || 0;
  const percent = numScore * 10;
  const radius = 18;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percent / 100) * circumference;

  let strokeColor = "var(--danger)";
  if (numScore >= 7) strokeColor = "var(--success)";
  else if (numScore >= 5) strokeColor = "var(--warn)";

  return (
    <div className="modern-rating-badge" aria-label={score === "NR" ? "Not Rated" : `Rating ${score} out of 10`}>
      <svg className="rating-svg" viewBox="0 0 44 44">
        <circle
          className="rating-track"
          cx="22"
          cy="22"
          r={radius}
          fill="transparent"
          stroke="rgba(255, 255, 255, 0.08)"
          strokeWidth="3"
        />
        {score !== "NR" && (
          <circle
            className="rating-fill"
            cx="22"
            cy="22"
            r={radius}
            fill="transparent"
            stroke={strokeColor}
            strokeWidth="3"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform="rotate(-90 22 22)"
          />
        )}
      </svg>
      <div className="rating-value">
        <span>{score}</span>
      </div>
    </div>
  );
}

function formatDate(dateStr?: string | null): string {
  if (!dateStr) return "";
  try {
    const parts = dateStr.split("-");
    if (parts.length === 3) {
      const year = parts[0];
      const monthIndex = parseInt(parts[1], 10) - 1;
      const day = parseInt(parts[2], 10);
      const months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
      ];
      if (monthIndex >= 0 && monthIndex < 12) {
        return `${months[monthIndex]} ${day}, ${year}`;
      }
    }
    const date = new Date(dateStr);
    if (!isNaN(date.getTime())) {
      return date.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
    }
  } catch {
    // fallback
  }
  return dateStr;
}

export const MovieDialog = React.memo(function MovieDialog({
  movie,
  onClose,
  feedback,
  onFeedback,
  onRating,
}: {
  movie: Movie;
  onClose: () => void;
  feedback?: "positive" | "negative";
  onFeedback?: (movie: Movie, value: "positive" | "negative") => void;
  onRating?: (movie: Movie, stars: number) => void;
}) {
  const director = directorLabel(movie);
  const cast = movie.cast || "";
  const genreList = movie.genres
    ? movie.genres.split(/[,|]/).map((g) => g.trim()).filter(Boolean)
    : [];
  const runtime = movie.runtime ? `${movie.runtime} min` : "";
  const overview = movie.overview || "No overview is available for this title.";
  const explanation = movie.explanation_text || movieReasons(movie).join(" | ");
  const rating = movieScore(movie);

  // Masterpiece States
  const [activeTab, setActiveTab] = React.useState<"overview" | "credits" | "insights">("overview");
  const [userRating, setUserRating] = React.useState(0);
  const [hoverRating, setHoverRating] = React.useState(0);
  const [inWatchlist, setInWatchlist] = React.useState(false);
  const [toast, setToast] = React.useState("");

  const dialogRef = React.useRef<HTMLElement>(null);
  const previousFocusRef = React.useRef<HTMLElement | null>(null);

  const showToast = React.useCallback((msg: string) => {
    setToast(msg);
  }, []);

  React.useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(""), 2200);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  React.useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    previousFocusRef.current = document.activeElement as HTMLElement | null;
    document.body.classList.add("modal-open");
    window.addEventListener("keydown", onKeyDown);
    const firstFocusable = dialogRef.current?.querySelector<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    firstFocusable?.focus();
    return () => {
      document.body.classList.remove("modal-open");
      window.removeEventListener("keydown", onKeyDown);
      previousFocusRef.current?.focus();
    };
  }, [onClose]);

  return (
    <div
      className="movie-dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="dialog-glow-aura" style={{ '--movie-backdrop': `url(${backdropUrl(movie.poster_path)})` } as React.CSSProperties} />

      <section ref={dialogRef} className="movie-dialog" role="dialog" aria-modal="true" aria-label={`${movie.title} details`}>
        <div className="mobile-sheet-handle" style={{ width: "36px", height: "5px", background: "rgba(255, 255, 255, 0.15)", borderRadius: "10px", margin: "12px auto 0 auto", display: "none" }} />
        <button className="dialog-close" type="button" aria-label="Close movie details" onClick={onClose}>
          <X size={20} />
        </button>

        <div className="dialog-media">
          <TrailerFrame movie={movie} />
        </div>

        <div className="dialog-content">
          <div className="dialog-tabs-header">
            <button
              type="button"
              className={`dialog-tab-btn ${activeTab === "overview" ? "active" : ""}`}
              onClick={() => setActiveTab("overview")}
            >
              Overview
            </button>
            <button
              type="button"
              className={`dialog-tab-btn ${activeTab === "credits" ? "active" : ""}`}
              onClick={() => setActiveTab("credits")}
            >
              Details & Cast
            </button>
            <button
              type="button"
              className={`dialog-tab-btn ${activeTab === "insights" ? "active" : ""}`}
              onClick={() => setActiveTab("insights")}
            >
              AI & Match
            </button>
          </div>

          <div className="dialog-grid">
            <div className="dialog-main">
              <div className="dialog-title-row">
                <h2>{movie.title}</h2>
                <RatingCircle score={rating} />
              </div>

              <div className="dialog-meta-row">
                <span className="meta-badge">
                  <Calendar size={14} />
                  <span>{movieYear(movie)}</span>
                </span>
                {runtime && (
                  <span className="meta-badge">
                    <Clock3 size={14} />
                    <span>{runtime}</span>
                  </span>
                )}
                {genreList.map((g) => (
                  <span key={g} className="meta-badge genre">
                    <Film size={14} />
                    <span>{g}</span>
                  </span>
                ))}
              </div>

              {activeTab === "overview" && (
                <>
                  <p className="dialog-overview">{overview}</p>
                  {explanation && (
                    <div className="dialog-vibe-card">
                      <div className="vibe-header">
                        <div className="vibe-title">
                          <Sparkles size={14} className="vibe-sparkle" />
                          <span>CineBot Vibe Check</span>
                        </div>
                        <span className="vibe-tag">AI Insights</span>
                      </div>
                      <p className="vibe-text">{explanation}</p>
                    </div>
                  )}
                </>
              )}

              {activeTab === "credits" && (
                <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                  {(director || cast || movie.release_date || movie.popularity || movie.vote_count) ? (
                    <div className="credits-tab-grid" style={{ display: "grid", gridTemplateColumns: "1fr", gap: "12px" }}>
                      {director && (
                        <div className="detail-item">
                          <span className="detail-label">Director</span>
                          <span className="detail-value" style={{ fontSize: "0.9rem", color: "#fff", fontWeight: "600" }}>{director}</span>
                        </div>
                      )}
                      {cast && (
                        <div className="detail-item">
                          <span className="detail-label">Cast</span>
                          <span className="detail-value" style={{ fontSize: "0.9rem", color: "#cbd5e1" }}>{cast}</span>
                        </div>
                      )}
                      {movie.release_date && (
                        <div className="detail-item">
                          <span className="detail-label">Released</span>
                          <span className="detail-value" style={{ fontSize: "0.9rem", color: "#fff" }}>{formatDate(movie.release_date)}</span>
                        </div>
                      )}
                      {movie.popularity !== undefined && movie.popularity !== null && (
                        <div className="detail-item">
                          <span className="detail-label">Popularity Score</span>
                          <span className="detail-value" style={{ fontSize: "0.9rem", color: "#fff" }}>{Number(movie.popularity).toFixed(1)}</span>
                        </div>
                      )}
                      {movie.vote_count !== undefined && movie.vote_count !== null && movie.vote_count > 0 && (
                        <div className="detail-item">
                          <span className="detail-label">Vote Count</span>
                          <span className="detail-value" style={{ fontSize: "0.9rem", color: "#fff" }}>{movie.vote_count.toLocaleString()} votes</span>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div style={{ color: "var(--quiet)", fontSize: "0.85rem" }}>No cast or crew details are available for this catalog item.</div>
                  )}
                </div>
              )}

              {activeTab === "insights" && (
                <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                  {(movie.similarity_score !== undefined && movie.similarity_score !== null) ? (
                    <div style={{
                      padding: "20px",
                      background: "rgba(6, 182, 212, 0.04)",
                      border: "1px solid rgba(6, 182, 212, 0.15)",
                      borderRadius: "16px",
                      boxShadow: "0 8px 32px rgba(6, 182, 212, 0.04)"
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.8rem", fontWeight: "900", textTransform: "uppercase", letterSpacing: "1px", color: "var(--cyan)" }}>
                          <Activity size={16} />
                          <span>Recommendation Match Insights</span>
                        </div>
                        <span style={{ fontSize: "0.72rem", background: "rgba(6, 182, 212, 0.1)", color: "#22d3ee", padding: "4px 10px", borderRadius: "20px", fontWeight: "800", border: "1px solid rgba(6, 182, 212, 0.1)" }}>
                          {Math.max(1, Math.min(99, Math.round(Number(movie.similarity_score) <= 1 ? Number(movie.similarity_score) * 100 : Number(movie.similarity_score))))}% Match Score
                        </span>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", fontSize: "0.84rem", color: "var(--muted)" }}>
                        {movie.retrieval_stage && (
                          <div>
                            Retrieval Pipeline
                            <div style={{ color: "#fff", fontWeight: "600", fontSize: "0.9rem", marginTop: "4px" }}>{movie.retrieval_stage}</div>
                          </div>
                        )}
                        <div>
                          Quality Tier
                          <div style={{ color: "#fff", fontWeight: "600", fontSize: "0.9rem", marginTop: "4px" }}>{movie.quality_bucket || "Tier 1 High-Confidence"}</div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div style={{
                      padding: "20px",
                      background: "rgba(6, 182, 212, 0.03)",
                      border: "1px solid rgba(6, 182, 212, 0.12)",
                      borderRadius: "16px",
                      display: "flex",
                      flexDirection: "column",
                      gap: "16px"
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.8rem", fontWeight: "900", textTransform: "uppercase", letterSpacing: "1px", color: "var(--cyan)" }}>
                          <Sparkles size={16} />
                          <span>Catalog Seed & Neural Profile</span>
                        </div>
                        <span style={{ fontSize: "0.72rem", background: "rgba(16, 185, 129, 0.1)", color: "#10b981", padding: "4px 10px", borderRadius: "20px", fontWeight: "800", border: "1px solid rgba(16, 185, 129, 0.2)" }}>
                          Seed Vector Active
                        </span>
                      </div>
                      <p style={{ margin: 0, fontSize: "0.86rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
                        This title serves as the primary reference seed. The 6-model neural ensemble (SASRec, KAN, LightGCN, Diffusion, Quantum-Fluid, Hyperbolic) projects this movie into 768-D vector space to retrieve and rank similar titles in real time.
                      </p>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", fontSize: "0.82rem", color: "var(--muted)", paddingTop: "8px", borderTop: "1px solid rgba(255, 255, 255, 0.05)" }}>
                        <div>
                          Embedding Vector
                          <div style={{ color: "#fff", fontWeight: "600", fontSize: "0.88rem", marginTop: "2px" }}>768-D SBERT (L2-Norm)</div>
                        </div>
                        <div>
                          Active Ensemble
                          <div style={{ color: "#fff", fontWeight: "600", fontSize: "0.88rem", marginTop: "2px" }}>6 Neural Models + Bandits</div>
                        </div>
                        <div>
                          Quality Tier
                          <div style={{ color: "#fff", fontWeight: "600", fontSize: "0.88rem", marginTop: "2px" }}>{movie.quality_bucket || "Tier 1 High-Quality"}</div>
                        </div>
                        <div>
                          Retrieval Latency
                          <div style={{ color: "#22d3ee", fontWeight: "600", fontSize: "0.88rem", marginTop: "2px" }}>&lt; 5ms (Rust SIMD)</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="dialog-sidebar">
              {onFeedback && (
                <div className="sidebar-section">
                  <h3>My Vibe</h3>
                  <div className="dialog-feedback-actions">
                    <button
                      type="button"
                      className={`feedback-btn thumbs-up ${feedback === "positive" ? "active" : ""}`}
                      onClick={() => {
                        onFeedback(movie, "positive");
                        showToast("Marked as Liked!");
                      }}
                      aria-label="Thumbs up"
                    >
                      <ThumbsUp size={16} />
                      <span>Like</span>
                    </button>
                    <button
                      type="button"
                      className={`feedback-btn thumbs-down ${feedback === "negative" ? "active" : ""}`}
                      onClick={() => {
                        onFeedback(movie, "negative");
                        showToast("Marked as Disliked.");
                      }}
                      aria-label="Thumbs down"
                    >
                      <ThumbsDown size={16} />
                      <span>Dislike</span>
                    </button>
                  </div>
                </div>
              )}

              <div className="sidebar-section">
                <h3>My Rating</h3>
                <div className="star-rating-container">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      type="button"
                      className={`star-btn ${star <= (hoverRating || userRating) ? "filled" : ""}`}
                      onMouseEnter={() => setHoverRating(star)}
                      onMouseLeave={() => setHoverRating(0)}
                      onClick={() => {
                        setUserRating(star);
                        if (onRating) {
                          onRating(movie, star);
                        }
                        showToast(`Rated ${star} Star${star > 1 ? "s" : ""}! Real-time signal synced.`);
                      }}
                      aria-label={`Rate ${star} stars`}
                    >
                      <Star size={20} fill={star <= (hoverRating || userRating) ? "currentColor" : "none"} />
                    </button>
                  ))}
                </div>
              </div>

              <div className="sidebar-section">
                <h3>Collection</h3>
                <div className="sidebar-actions-grid">
                  <button
                    type="button"
                    className={`action-pill-btn ${inWatchlist ? "active" : ""}`}
                    onClick={() => {
                      setInWatchlist(!inWatchlist);
                      showToast(inWatchlist ? "Removed from Watchlist" : "Saved to Watchlist!");
                    }}
                  >
                    <Bookmark size={15} fill={inWatchlist ? "currentColor" : "none"} />
                    <span>{inWatchlist ? "Watchlisted" : "Watchlist"}</span>
                  </button>
                  <button
                    type="button"
                    className="action-pill-btn"
                    onClick={() => {
                      try {
                        navigator.clipboard.writeText(window.location.href);
                        showToast("Copied link to clipboard!");
                      } catch {
                        showToast("Failed to copy link.");
                      }
                    }}
                  >
                    <Share2 size={15} />
                    <span>Share</span>
                  </button>
                </div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "8px", width: "100%" }}>
                {movie.trailer_key && (
                  <a className="dialog-action-btn primary" href={`https://www.youtube.com/watch?v=${movie.trailer_key}`} target="_blank" rel="noreferrer">
                    <Play size={16} fill="currentColor" />
                    <span>Play Trailer</span>
                  </a>
                )}
                <a
                  className="dialog-action-btn secondary"
                  href={`https://www.google.com/search?q=${encodeURIComponent(movie.title + " " + movieYear(movie) + " movie")}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  <span>Search Google</span>
                </a>
              </div>
            </div>
          </div>
        </div>

        {toast && (
          <div className="dialog-toast">
            <Sparkles size={14} />
            <span>{toast}</span>
          </div>
        )}
      </section>
    </div>
  );
});



const HomePage = React.memo(function HomePage({
  movies,
  heroIndex,
  loading,
  error,
  onHeroIndex,
  onOpenMovie,
  recentMovies,
  forYouMovies,
  forYouLoading,
  latestMovies,
  latestLoading,
  homeMode,
  onToggleMode,
}: {
  movies: Movie[];
  heroIndex: number;
  loading: boolean;
  error: string;
  onHeroIndex: (index: number) => void;
  onOpenMovie: (movie: Movie) => void;
  recentMovies: Movie[];
  forYouMovies: Movie[];
  forYouLoading: boolean;
  latestMovies: Movie[];
  latestLoading: boolean;
  homeMode: "foryou" | "latest" | "trending";
  onToggleMode: (mode: "foryou" | "latest" | "trending") => void;
}) {
  const hasForYou = recentMovies.length > 0 || forYouMovies.length > 0;
  const activeMovies = homeMode === "foryou" && hasForYou ? forYouMovies : homeMode === "latest" ? latestMovies : movies;
  const hero = activeMovies[heroIndex] || activeMovies[0] || null;
  const isLoading = homeMode === "foryou" ? forYouLoading : homeMode === "latest" ? latestLoading : loading;
  const heroMeta = hero
    ? [movieScore(hero) !== "NR" ? `Rating ${movieScore(hero)}` : "", compactGenres(hero.genres), hero.runtime ? `${hero.runtime} min` : ""]
        .filter(Boolean)
        .join(" | ")
    : "";
  const heroOverview = hero?.overview
    ? hero.overview.length > 160
      ? `${hero.overview.slice(0, 160).replace(/\s+\S*$/, "")}...`
      : hero.overview
    : "";

  return (
    <main className="home-shell full-width-layout">
      {hero ? (
        <div className="home-showcase full-width">
          {/* Widescreen Hero Billboard */}
          <section className="billboard-container">
            <div className="billboard-video">
              <TrailerFrame movie={hero} />
            </div>
            <div className="billboard-info">
              <h2>{hero.title}</h2>
              <div className="bb-meta">{heroMeta}</div>
              <p>{heroOverview}</p>
              <div className="bb-credits">
                {directorLabel(hero) && (
                  <span>
                    Directed by <strong>{directorLabel(hero)}</strong>
                  </span>
                )}
              </div>
              <button className="bb-details-button" type="button" onClick={() => onOpenMovie(hero)}>
                <Play size={14} />
                View details
              </button>
            </div>
          </section>

          {/* Cinematic Scrolling Rows (Netflix/Prime Style) */}
          <div className="category-rows">
            {/* Row 1: Popular & Trending */}
            <div className="category-row">
              <h3 className="category-title">Popular & Trending</h3>
              <div className="trending-strip">
                {movies.slice(0, 10).map((movie, index) => (
                  <button
                    className={movie.id === hero.id ? "active" : ""}
                    type="button"
                    key={`trending-${movie.id}`}
                    onClick={() => {
                      onToggleMode("trending");
                      onHeroIndex(index);
                    }}
                    onDoubleClick={() => onOpenMovie(movie)}
                    title={movie.title}
                  >
                    <img src={posterUrl(movie.poster_path)} alt={movie.title} loading="lazy" />
                  </button>
                ))}
              </div>
            </div>

            {/* Row 2: Personalized For You recommendations */}
            {hasForYou && (
              <div className="category-row">
                <h3 className="category-title">Recommended For You</h3>
                <div className="trending-strip">
                  {forYouMovies.slice(0, 10).map((movie, index) => (
                    <button
                      className={movie.id === hero.id ? "active" : ""}
                      type="button"
                      key={`foryou-${movie.id}`}
                      onClick={() => {
                        onToggleMode("foryou");
                        onHeroIndex(index);
                      }}
                      onDoubleClick={() => onOpenMovie(movie)}
                      title={movie.title}
                    >
                      <img src={posterUrl(movie.poster_path)} alt={movie.title} loading="lazy" />
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Row 3: Latest Releases */}
            <div className="category-row">
              <h3 className="category-title">Latest Releases</h3>
              <div className="trending-strip">
                {latestMovies.slice(0, 10).map((movie, index) => (
                  <button
                    className={movie.id === hero.id ? "active" : ""}
                    type="button"
                    key={`latest-${movie.id}`}
                    onClick={() => {
                      onToggleMode("latest");
                      onHeroIndex(index);
                    }}
                    onDoubleClick={() => onOpenMovie(movie)}
                    title={movie.title}
                  >
                    <img src={posterUrl(movie.poster_path)} alt={movie.title} loading="lazy" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <section className="billboard-container empty">
          <div>
            <Loader2 className={isLoading ? "spin" : undefined} size={28} />
            <h2>{isLoading ? "Loading Showcase" : "Unavailable"}</h2>
            <p>{error || "The recommendation service is warming up."}</p>
          </div>
        </section>
      )}
    </main>
  );
});

// ─── Auth Modal (shared, focus-trapped) ──────────────────────────────────────

function AuthModal({
  onLogin,
  onClose,
}: {
  onLogin: (tok: string, user: string) => void;
  onClose: () => void;
}) {
  const overlayRef = React.useRef<HTMLDivElement>(null);

  // Move focus into the modal on mount; restore on unmount
  React.useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null;
    const firstFocusable = overlayRef.current?.querySelector<HTMLElement>(
      'button, input, [href], select, textarea, [tabindex]:not([tabindex="-1"])',
    );
    firstFocusable?.focus();
    return () => { previousFocus?.focus(); };
  }, []);

  function handleKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key === "Escape") { onClose(); return; }
    if (e.key !== "Tab") return;
    const modal = overlayRef.current;
    if (!modal) return;
    const focusable = Array.from(
      modal.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]), [href], select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    );
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault(); last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  }

  return (
    // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions
    <div
      ref={overlayRef}
      className="auth-modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Sign in to your account"
      tabIndex={-1}
      onKeyDown={handleKeyDown}
    >
      <AuthPage onLogin={(tok, user) => { onLogin(tok, user); }} onClose={onClose} />
    </div>
  );
}

function App() {
  const [token, setToken] = React.useState<string | null>(window.localStorage.getItem("nova_jwt_token"));
  const [username, setUsername] = React.useState<string | null>(window.localStorage.getItem("nova_username"));
  const [page, setPage] = React.useState<AppPage>("home");
  const [showAuthModal, setShowAuthModal] = React.useState(false);
  const [titles, setTitles] = React.useState<MovieTitle[]>([]);
  const [titleQuery, setTitleQuery] = React.useState("");
  const [mode, setMode] = React.useState<SearchMode>("title");
  const [selectedMovie, setSelectedMovie] = React.useState<Movie | null>(null);
  const [results, setResults] = React.useState<Movie[]>([]);
  const [resultsKind, setResultsKind] = React.useState<ResultsKind>("idle");
  const [catalogState, setCatalogState] = React.useState<CatalogState>("booting");
  const [backend, setBackend] = React.useState(currentBackend());
  const [notice, setNotice] = React.useState("Connecting to recommendation API");
  const [retryCount, setRetryCount] = React.useState(0);
  const [isSearching, setIsSearching] = React.useState(false);
  const [isSelecting, setIsSelecting] = React.useState(false);
  const [loadingRecs, setLoadingRecs] = React.useState(false);
  const [_lastUpdated, setLastUpdated] = React.useState("");
  const [platform, setPlatform] = React.useState<PlatformStatus | null>(null);
  const [readinessReport, setReadinessReport] = React.useState<PlatformReadiness | null>(null);
  const [artifactReport, setArtifactReport] = React.useState<ArtifactHealth | null>(null);
  const [qualityReport, setQualityReport] = React.useState<SemanticBenchmark | null>(null);
  const [signalsLoading, setSignalsLoading] = React.useState(false);
  const [recentMovies, setRecentMovies] = React.useState<Movie[]>(() => loadRecentMovies());
  const [feedbackByMovieId, setFeedbackByMovieId] = React.useState<Record<number, FeedbackValue>>({});
  const [feedbackNotice, setFeedbackNotice] = React.useState("");
  const [lastRecommendationRequestId, setLastRecommendationRequestId] = React.useState<string | null>(null);
  const [recommendationSource, setRecommendationSource] = React.useState<Movie | null>(null);
  const [dialogMovie, setDialogMovie] = React.useState<Movie | null>(null);
  const [titleSelectOpen, setTitleSelectOpen] = React.useState(false);
  const [homeMovies, setHomeMovies] = React.useState<Movie[]>([]);
  const [homeHeroIndex, setHomeHeroIndex] = React.useState(0);
  const [homeLoading, setHomeLoading] = React.useState(false);
  const [homeError, setHomeError] = React.useState("");
  const [forYouMovies, setForYouMovies] = React.useState<Movie[]>([]);
  const [forYouLoading, setForYouLoading] = React.useState(false);
  const [latestMovies, setLatestMovies] = React.useState<Movie[]>([]);
  const [latestLoading, setLatestLoading] = React.useState(false);
  const [homeMode, setHomeMode] = React.useState<"foryou" | "latest" | "trending">(() => loadRecentMovies().length > 0 ? "foryou" : "trending");
  const [sessionId] = React.useState(() => getSessionId());
  const [isMobileViewport, setIsMobileViewport] = React.useState(() => typeof window !== "undefined" ? window.innerWidth <= 768 : false);
  const [isMobileSimulated, setIsMobileSimulated] = React.useState(false);
  const [showMoreDrawer, setShowMoreDrawer] = React.useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [deferredPrompt, setDeferredPrompt] = React.useState<any>(null);

  React.useEffect(() => {
    function handleInstallPrompt(e: Event) {
      e.preventDefault();
      setDeferredPrompt(e);
    }
    window.addEventListener("beforeinstallprompt", handleInstallPrompt);
    return () => window.removeEventListener("beforeinstallprompt", handleInstallPrompt);
  }, []);

  React.useEffect(() => {
    function handleResize() {
      setIsMobileViewport(window.innerWidth <= 768);
    }
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const isMobileMode = isMobileViewport || isMobileSimulated;
  const titleSelectRef = React.useRef<HTMLDivElement>(null);
  const bootstrapped = React.useRef(false);
  const loadedPlatform = React.useRef(false);
  const loadedHomeShowcase = React.useRef(false);
  const _loadedForYou = React.useRef(false);
  const userStarted = React.useRef(false);
  const activeQuery = titleQuery;
  const hasTitleQuery = titleQuery.trim().length > 0;
  const selectedTitleLabel = selectedMovie ? selectTitleLabel(selectedMovie) : "";
  const isSelectedTitleQuery = Boolean(selectedMovie && titleQuery === selectedTitleLabel);
  const isEditingTitle = Boolean(selectedMovie && hasTitleQuery && !isSelectedTitleQuery);
  const showTitleSuggestions = titleSelectOpen && hasTitleQuery && !isSelectedTitleQuery;
  const showNotice = catalogState !== "ready";

  const filteredTitles = React.useMemo(() => {
    const normalized = titleQuery.trim().toLowerCase();
    if (!normalized) return titles.slice(0, 34);
    const scoreTitle = (title: string) => {
      const value = title.toLowerCase();
      const cleanValue = value.replace(/^the\s+/, "");
      const index = Math.min(
        value.includes(normalized) ? value.indexOf(normalized) : Number.POSITIVE_INFINITY,
        cleanValue.includes(normalized) ? cleanValue.indexOf(normalized) : Number.POSITIVE_INFINITY,
      );
      const prefix = value.startsWith(normalized) || cleanValue.startsWith(normalized) ? 0 : 1;
      const exact = value === normalized || cleanValue === normalized ? 0 : 1;
      return { exact, prefix, index, length: value.length, value };
    };

    return titles
      .filter((item) => item.title.toLowerCase().replace(/^the\s+/, "").includes(normalized) || item.title.toLowerCase().includes(normalized))
      .sort((a, b) => {
        const left = scoreTitle(a.title);
        const right = scoreTitle(b.title);
        return (
          left.exact - right.exact ||
          left.prefix - right.prefix ||
          left.index - right.index ||
          left.length - right.length ||
          left.value.localeCompare(right.value)
        );
      })
      .slice(0, 34);
  }, [titles, titleQuery]);

  const resultHeading =
    resultsKind === "recommendations"
      ? `Recommendations similar to ${(recommendationSource || selectedMovie)?.title || "this title"}`
      : "Search matches";

  function rememberMovie(movie: Movie) {
    const next = dedupeMovies([movie, ...recentMovies]).slice(0, 6);
    setRecentMovies(next);
    saveRecentMovies(next);
  }

  function emitBehaviorEvent(payload: EventPayload) {
    const eventPayload: EventPayload = {
      ...payload,
      session_id: payload.session_id || sessionId,
      user_id: payload.user_id || username || undefined,
      metadata: {
        client: "react",
        surface: "web",
        auth_state: username ? "authenticated" : "anonymous",
        username: username || undefined,
        ...(payload.metadata || {}),
      },
    };
    if (eventPayload.movie_id !== null && eventPayload.movie_id !== undefined && !eventPayload.content_id) {
      eventPayload.content_id = String(eventPayload.movie_id);
    }

    void recordEvent(eventPayload).catch((error) => {
      console.warn("Behavior event was not recorded", error);
    });
  }

  async function loadHomeShowcase() {
    setHomeLoading(true);
    setHomeError("");

    // Phase 1: Instant showcase from in-memory catalog (no TMDB, no ML — sub-200ms)
    try {
      const showcaseResponse = await getShowcaseMovies(8);
      if (showcaseResponse.data && showcaseResponse.data.length > 0) {
        const movies = dedupeMovies(showcaseResponse.data).slice(0, 8);
        setBackend(showcaseResponse.baseUrl);
        setHomeMovies(movies);
        setHomeHeroIndex(0);
        setCatalogState("ready");
        setHomeLoading(false);

        // Phase 2: Lazy-upgrade to real ML recommendations in background
        const seeds = [155, 27205, 157336, 680, 238];
        const seedId = seeds[Math.floor(Math.random() * seeds.length)];
        getRecommendations(seedId, 8, 8000)
          .then((recResponse) => {
            const recMovies = dedupeMovies(recResponse.data.recommendations || []).slice(0, 8);
            if (recMovies.length > 0) {
              setHomeMovies(recMovies);
              setHomeHeroIndex(0);
            }
          })
          .catch(() => { /* Showcase already visible, ML upgrade is optional */ });
        return;
      }
    } catch (e) {
      console.warn("[SHOWCASE] Fast showcase unavailable, falling back:", e);
    }

    // Phase 1b fallback: search query (still fast, just reads DB)
    try {
      const searchResponse = await searchMovies("Batman");
      if (searchResponse.data && searchResponse.data.length > 0) {
        const movies = dedupeMovies(searchResponse.data).slice(0, 8);
        setBackend(searchResponse.baseUrl);
        setHomeMovies(movies);
        setHomeHeroIndex(0);
        setCatalogState("ready");
      } else {
        setHomeError("No movies found in catalog.");
      }
    } catch (error) {
      setHomeError(error instanceof Error ? error.message : "Movies unavailable during warmup.");
    } finally {
      setHomeLoading(false);
    }
  }

  async function loadForYouShowcase(userId?: string) {
    setForYouLoading(true);
    try {
      if (userId) {
        const response = await getUserRecommendations(userId, 8);
        const movies = dedupeMovies(response.data || []).slice(0, 8);
        setForYouMovies(movies);
        setBackend(response.baseUrl);
        setHomeHeroIndex(0);
        return;
      }
      const recent = loadRecentMovies();
      if (recent.length === 0) return;
      const seedMovie = recent[0];
      if (!seedMovie.id) return;
      const response = await getRecommendations(seedMovie.id, 8);
      const movies = dedupeMovies(response.data.recommendations || []).slice(0, 8);
      setForYouMovies(movies);
      setHomeHeroIndex(0);
    } catch {
      // Fall back to trending silently
      setHomeMode("trending");
    } finally {
      setForYouLoading(false);
    }
  }

  async function loadLatestShowcase() {
    setLatestLoading(true);
    try {
      // Fetch real latest/trending movies from TMDB via backend
      const response = await apiGet<Movie[]>("/movies/latest", { limit: 8 });
      const movies = response.data || [];
      if (movies.length > 0) {
        setLatestMovies(dedupeMovies(movies).slice(0, 8));
        setHomeHeroIndex(0);
      }
      setCatalogState("ready");
    } catch { /* silent fallback */ }
    finally { setLatestLoading(false); }
  }

  function selectMovie(movie: Movie, source: SelectionSource, track = true) {
    setSelectedMovie(movie);
    rememberMovie(movie);
    if (!track) return;
    emitBehaviorEvent({
      event_type: "view",
      movie_id: movie.id,
      metadata: {
        source,
        title: movie.title,
        results_kind: resultsKind,
      },
    });
  }

  function selectResultMovie(movie: Movie) {
    if (resultsKind === "recommendations") {
      const sourceMovie = recommendationSource || selectedMovie;
      emitBehaviorEvent({
        event_type: "click",
        movie_id: movie.id,
        source_content_id: sourceMovie ? String(sourceMovie.id) : undefined,
        request_id: lastRecommendationRequestId,
        metadata: {
          title: movie.title,
          source_title: sourceMovie?.title,
          retrieval_stage: movie.retrieval_stage,
          similarity_score: movie.similarity_score,
        },
      });
      rememberMovie(movie);
      setDialogMovie(movie);
      return;
    }
    setDialogMovie(movie);
    selectMovie(movie, "search_result");
  }

  function recordFeedback(movie: Movie, value: FeedbackValue) {
    const rating = value === "positive" ? 5 : 1;
    const sourceMovie = resultsKind === "recommendations" ? recommendationSource || selectedMovie : selectedMovie;
    setFeedbackByMovieId((current) => ({ ...current, [movie.id]: value }));
    setFeedbackNotice(value === "positive" ? "Preference saved" : "Negative signal saved");
    emitBehaviorEvent({
      event_type: "rating",
      movie_id: movie.id,
      source_content_id: sourceMovie && sourceMovie.id !== movie.id ? String(sourceMovie.id) : undefined,
      rating,
      request_id: lastRecommendationRequestId,
      metadata: {
        title: movie.title,
        source_title: sourceMovie?.title,
        sentiment: value,
        results_kind: resultsKind,
        retrieval_stage: movie.retrieval_stage,
        similarity_score: movie.similarity_score,
      },
    });
  }

  function recordRating(movie: Movie, stars: number) {
    const sourceMovie = resultsKind === "recommendations" ? recommendationSource || selectedMovie : selectedMovie;
    emitBehaviorEvent({
      event_type: "rating",
      movie_id: movie.id,
      source_content_id: sourceMovie && sourceMovie.id !== movie.id ? String(sourceMovie.id) : undefined,
      rating: stars,
      request_id: lastRecommendationRequestId,
      metadata: {
        title: movie.title,
        source_title: sourceMovie?.title,
        sentiment: stars >= 4 ? "positive" : stars <= 2 ? "negative" : "neutral",
        stars,
        results_kind: resultsKind,
        retrieval_stage: movie.retrieval_stage,
        similarity_score: movie.similarity_score,
      },
    });
  }

  async function bootstrap(silent = false) {
    setCatalogState(silent ? "warming" : "booting");
    setNotice(silent ? "Checking the service again" : "Connecting to recommendation API");
    try {
      const ping = await pingApi();
      setBackend(ping.baseUrl);
      setNotice("Service online. Loading catalog.");

      const result = await loadTitles(TITLE_CATALOG_LIMIT);
      setTitles(result.data);
      setBackend(result.baseUrl);
      setCatalogState("ready");
      setRetryCount(0);
      setLastUpdated(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
      setNotice(`${result.data.length.toLocaleString()} searchable titles loaded`);
    } catch (error) {
      setCatalogState("warming");
      setRetryCount((count) => count + 1);
      setNotice(error instanceof Error ? `Service warming. ${error.message}` : "Service warming.");
    }
  }

  function loadOperationalSignals() {
    setSignalsLoading(true);

    const settle = <T,>(request: Promise<{ data: T; baseUrl: string }>, onValue: (data: T, baseUrl: string) => void) =>
      request
        .then((response) => {
          setBackend(response.baseUrl);
          onValue(response.data, response.baseUrl);
          return true;
        })
        .catch((error) => {
          console.warn("Operational signal unavailable", error);
          return false;
        });

    const requests = [
      settle(platformStatus(), (data) => setPlatform(data)),
      settle(platformReadiness(true, 10), (data) => setReadinessReport(data)),
      settle(artifactHealth(), (data) => setArtifactReport(data)),
      settle(semanticBenchmark(10), (data) => setQualityReport(data)),
    ];

    void Promise.all(requests).then((results) => {
      setSignalsLoading(false);
      if (results.every((success) => !success)) {
        loadedPlatform.current = false;
      }
    });
  }

  React.useEffect(() => {
    if (bootstrapped.current) return;
    bootstrapped.current = true;
    void bootstrap();
  }, []);

  React.useEffect(() => {
    if (loadedHomeShowcase.current) return;
    loadedHomeShowcase.current = true;
    void loadHomeShowcase();
  }, []);

  React.useEffect(() => {
    void loadForYouShowcase(username || "Guest");
  }, [username]);

  const loadedLatest = React.useRef(false);
  React.useEffect(() => {
    if (loadedLatest.current) return;
    loadedLatest.current = true;
    void loadLatestShowcase();
  }, []);

  React.useEffect(() => {
    if (catalogState !== "warming") return;
    const delay = Math.min(30000, 7000 + retryCount * 4000);
    const retry = window.setTimeout(() => void bootstrap(true), delay);
    return () => window.clearTimeout(retry);
  }, [catalogState, retryCount]);

  React.useEffect(() => {
    if (catalogState === "ready") {
      if (homeMovies.length === 0 && !homeLoading) {
        void loadHomeShowcase();
      }
      if (latestMovies.length === 0 && !latestLoading) {
        void loadLatestShowcase();
      }
    }
  }, [catalogState, homeMovies.length, homeLoading, latestMovies.length, latestLoading]);

  React.useEffect(() => {
    if (catalogState !== "ready" || loadedPlatform.current) return;
    loadedPlatform.current = true;
    const timer = window.setTimeout(() => {
      loadOperationalSignals();
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [catalogState]);

  React.useEffect(() => {
    const query = titleQuery.trim();
    if (!query) {
      setResults([]);
      setResultsKind("idle");
      return;
    }
    if (selectedMovie && query === selectTitleLabel(selectedMovie)) {
      return;
    }

    const delay = mode === "semantic" ? 1000 : 300;
    const timer = window.setTimeout(() => {
      void runSearch(mode);
    }, delay);

    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [titleQuery, mode]);

  async function chooseTitle(
    item: MovieTitle,
    options: { track?: boolean; autoRecommend?: boolean } = {},
  ) {
    setIsSelecting(true);
    setTitleQuery(item.title);
    setResults([]);
    setResultsKind("idle");
    setLastRecommendationRequestId(null);
    setRecommendationSource(null);
    setFeedbackByMovieId({});
    setFeedbackNotice("");
    setNotice("Opening title");
    try {
      const result = await getMovie(item.id);
      setBackend(result.baseUrl);
      setTitleQuery(selectTitleLabel(result.data));
      setTitleSelectOpen(false);
      selectMovie(result.data, "title_search", options.track ?? true);
      setCatalogState("ready");
      setNotice("Title ready");
      if (options.autoRecommend) {
        void recommend(result.data);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Title unavailable");
    } finally {
      setIsSelecting(false);
    }
  }

  async function runSearch(searchMode: SearchMode = mode, queryOverride?: string) {
    const query = (queryOverride ?? titleQuery).trim();
    if (!query) return;

    if (searchMode === "title" && !queryOverride && isSelectedTitleQuery && selectedMovie) {
      await recommend(selectedMovie);
      return;
    }
    setIsSearching(true);
    setNotice(searchMode === "semantic" ? "Searching by intent" : "Searching catalog");
    setResults([]);
    setResultsKind("search");
    setLastRecommendationRequestId(null);
    setRecommendationSource(null);
    setFeedbackByMovieId({});
    setFeedbackNotice("");
    emitBehaviorEvent({
      event_type: "search",
      query_text: query,
      metadata: {
        mode: searchMode,
      },
    });
    try {
      const response = searchMode === "semantic" ? await aiSearch(query) : await searchMovies(query);
      const movies = dedupeMovies(response.data);
      setBackend(response.baseUrl);
      setResults(movies);
      if (movies[0]) {
        selectMovie(movies[0], searchMode === "semantic" ? "semantic_search" : "title_search");
      } else {
        setSelectedMovie(null);
      }
      setCatalogState("ready");
      setNotice(`${movies.length} matches`);
    } catch (error) {
      setCatalogState("error");
      setNotice(error instanceof Error ? error.message : "Search unavailable");
    } finally {
      setIsSearching(false);
    }
  }

  async function recommend(movie = selectedMovie) {
    if (!movie) return;
    setLoadingRecs(true);
    setNotice("Ranking candidates");
    try {
      const response = await getRecommendations(movie.id, 16);
      const recommendations = dedupeMovies(response.data.recommendations);
      setBackend(response.baseUrl);
      setRecommendationSource(response.data.query_movie);
      selectMovie(response.data.query_movie, "search_result", false);
      setResults(recommendations);
      setResultsKind("recommendations");
      setFeedbackByMovieId({});
      setFeedbackNotice("");
      setLastRecommendationRequestId(response.data.request_id || null);
      setCatalogState("ready");
      setNotice(`${recommendations.length} recommendations ranked`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Recommendations unavailable");
    } finally {
      setLoadingRecs(false);
    }
  }

  function openHome() {
    setPage("home");
    setDialogMovie(null);
    setTitleSelectOpen(false);
    window.history.pushState({ page: "home" }, "", "/");
  }

  function openSearch(searchMode?: "title" | "semantic") {
    if (searchMode) setMode(searchMode);
    setPage("search");
    setDialogMovie(null);
    setTitleSelectOpen(false);
    window.history.pushState({ page: "search", mode: searchMode }, "", searchMode === "semantic" ? "/discover" : "/search");
    // Auto-expand semantic section and focus the right input after render
    if (searchMode === "semantic") {
      setTimeout(() => {
        const details = document.querySelector(".semantic-expander") as HTMLDetailsElement | null;
        if (details) details.open = true;
        const input = document.getElementById("semantic-search") as HTMLInputElement | null;
        if (input) input.focus();
      }, 100);
    } else if (searchMode === "title") {
      setTimeout(() => {
        const input = document.getElementById("title-search") as HTMLInputElement | null;
        if (input) input.focus();
      }, 100);
    }
  }

  // Browser back/forward button support
  React.useEffect(() => {
    function handlePopState(event: PopStateEvent) {
      const state = event.state;
      if (state?.page === "home") {
        setPage("home");
        setDialogMovie(null);
      } else if (state?.page === "search") {
        if (state.mode) setMode(state.mode);
        setPage("search");
        setDialogMovie(null);
      } else {
        setPage("home");
      }
    }
    window.addEventListener("popstate", handlePopState);
    // Set initial history state
    window.history.replaceState({ page: "home" }, "", "/");
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const catalogValue = platform?.movie_count || titles.length;
  const rankerValue = platform?.ranker?.available ? "Learned" : "Hybrid";
  const innerPages: AppPage[] = ["vector-space", "dashboard", "knowledge-graph", "evaluation", "profile", "admin"];
  const isAppPage = ["home", "search", ...innerPages].includes(page);

  // ── Full-screen marketing pages (no app shell) ───────────────────────────
  if (page === "landing") {
    return (
      <React.Suspense fallback={<SuspenseFallback />}>
        <LandingPage onNavigate={(p) => setPage(p as AppPage)} />
      </React.Suspense>
    );
  }
  if (page === "signup") {
    return (
      <React.Suspense fallback={<SuspenseFallback />}>
        <SignupPage
          onNavigate={(p) => setPage(p as AppPage)}
          onLoginSuccess={(tok, user) => { setToken(tok); setUsername(user); }}
        />
      </React.Suspense>
    );
  }
  if (page === "pricing") {
    return (
      <React.Suspense fallback={<SuspenseFallback />}>
        <PricingPage onNavigate={(p) => setPage(p as AppPage)} />
      </React.Suspense>
    );
  }
  if (page === "getting-started") {
    return (
      <React.Suspense fallback={<SuspenseFallback />}>
        <GettingStartedPage onNavigate={(p) => setPage(p as AppPage)} />
      </React.Suspense>
    );
  }
  if (page === "status") {
    return (
      <React.Suspense fallback={<SuspenseFallback />}>
        <StatusPage />
      </React.Suspense>
    );
  }

  // ── App Shell Pages (Browse, Search, Dashboard, KG, Eval, Profile, Admin) ─
  if (isAppPage && isMobileMode) {
    const mobileContent = (
      <div className="iphone-screen">
        <div className="mobile-screen-orb" />

        {/* Simplified Sticky Header */}
        <header className="topbar" style={{ position: "sticky", top: 0, width: "100%", zIndex: 100 }}>
          <button className="brand-logo" type="button" onClick={openHome} style={{ fontSize: "1.3rem" }}>NOVA</button>
          <div className="topbar-right" style={{ gap: "12px" }}>
            <StatusBadge state={catalogState} backend={backend} />
            {username ? (
              <div className="user-profile-menu" style={{ padding: "4px 12px" }}>
                <button className="profile-btn" type="button" onClick={() => setPage("profile")}>
                  <strong>{username}</strong>
                </button>
              </div>
            ) : (
              <button className="signin-nav-btn" type="button" onClick={() => setShowAuthModal(true)}>
                Sign In
              </button>
            )}
          </div>
        </header>

        {/* Content Area */}
        <div className="mobile-shell-content">
          {page === "home" && (
            <HomePage
              movies={homeMovies}
              heroIndex={homeHeroIndex}
              loading={homeLoading}
              error={homeError}
              onHeroIndex={setHomeHeroIndex}
              onOpenMovie={setDialogMovie}
              recentMovies={recentMovies}
              forYouMovies={forYouMovies}
              forYouLoading={forYouLoading}
              latestMovies={latestMovies}
              latestLoading={latestLoading}
              homeMode={homeMode}
              onToggleMode={setHomeMode}
            />
          )}

          {page === "search" && (
            <main className="search-page-layout">
              {/* Metrics strip in 2x2 grid */}
              <section className="metrics-strip" aria-label="Platform snapshot" style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "8px", marginBottom: "16px" }}>
                <MetricTile icon={<Database size={14} />} label="Catalog" value={catalogValue ? catalogValue.toLocaleString() : "Loading"} />
                <MetricTile icon={<Server size={14} />} label="Readiness" value={readinessLabel(readinessReport)} />
                <MetricTile icon={<Gauge size={14} />} label="Quality" value={qualityLabel(qualityReport)} />
                <MetricTile icon={<Activity size={14} />} label="Artifacts" value={healthLabel(artifactReport)} />
              </section>

              <section className="workspace" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                <div className="control-panel" style={{ padding: "16px", borderRadius: "16px" }}>
                  <div className="title-select" ref={titleSelectRef}>
                    <div className="search-box title-select-box" style={{ padding: "4px 8px" }}>
                      {mode === "semantic" ? <Sparkles size={16} className="mode-icon-semantic" /> : <Search size={16} />}
                      <input
                        id="title-search"
                        value={titleQuery}
                        onChange={(event) => {
                          userStarted.current = true;
                          setMode(mode);
                          setTitleQuery(event.target.value);
                          setTitleSelectOpen(true);
                          if (selectedMovie && event.target.value !== selectedTitleLabel) {
                            setSelectedMovie(null);
                            setResults([]);
                            setResultsKind("idle");
                            setRecommendationSource(null);
                            setDialogMovie(null);
                            setLastRecommendationRequestId(null);
                          }
                        }}
                        onFocus={() => setTitleSelectOpen(true)}
                        onKeyDown={(event) => {
                          if (event.key === "Escape") {
                            setTitleSelectOpen(false);
                            return;
                          }
                          if (event.key === "Enter") {
                            event.preventDefault();
                            setTitleSelectOpen(false);
                            void runSearch(mode);
                          }
                        }}
                        placeholder={mode === "title" ? "Movie title..." : "Plot, mood..."}
                        style={{ fontSize: "0.82rem" }}
                      />
                      <button
                        className={`mode-toggle-btn ${mode === "semantic" ? "semantic-active" : ""}`}
                        type="button"
                        onClick={() => setMode(mode === "title" ? "semantic" : "title")}
                        style={{ padding: "4px 8px", fontSize: "0.75rem" }}
                      >
                        {mode === "semantic" ? <Sparkles size={12} /> : <Search size={12} />}
                      </button>
                    </div>

                    {showTitleSuggestions && mode === "title" && (
                      <div className="title-list" style={{ maxHeight: "200px" }}>
                        {filteredTitles.slice(0, 8).map((item) => (
                          <button
                            type="button"
                            key={`${item.id}-${item.title}`}
                            onClick={() => void chooseTitle(item, { autoRecommend: true })}
                            style={{ fontSize: "0.82rem", padding: "8px 12px" }}
                          >
                            {item.title}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <div className="result-panel">
                  {selectedMovie && !isEditingTitle ? (
                    <MovieSpotlight
                      movie={selectedMovie}
                      loading={loadingRecs}
                      onRecommend={() => void recommend()}
                      userId={username}
                      sessionId={sessionId}
                    />
                  ) : null}

                  {isSearching || loadingRecs ? (
                    <div style={{ display: "flex", justifyContent: "center", padding: "40px" }}>
                      <Loader2 className="spin" size={28} style={{ color: "var(--accent)" }} />
                    </div>
                  ) : results.length > 0 ? (
                    <section className="results-section">
                      <div className="section-title" style={{ fontSize: "0.95rem", margin: "12px 0 6px" }}>
                        <h2>{resultsKind === "recommendations" ? "Similar Movies" : "Search Results"}</h2>
                      </div>
                      <div className="poster-grid" style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "10px" }}>
                        {results.map((movie, index) => (
                          <RecommendationCard
                            key={`${movie.id}-${movie.title}`}
                            movie={movie}
                            rank={index + 1}
                            onSelect={selectResultMovie}
                            feedback={feedbackByMovieId[movie.id]}
                            onFeedback={recordFeedback}
                          />
                        ))}
                      </div>
                    </section>
                  ) : (
                    /* Show trending keywords and popular searches on mobile search if empty */
                    !selectedMovie && (
                      <div className="search-suggestions" style={{ marginTop: "16px" }}>
                        <div style={{ marginBottom: "20px" }}>
                          <span style={{ fontSize: "0.72rem", color: "var(--accent)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>Intent filter</span>
                          <h3 style={{ fontSize: "0.95rem", color: "#fff", fontWeight: 700, margin: "2px 0 10px", fontFamily: "var(--font-headline)" }}>Trending Searches</h3>
                          <div className="search-tags" style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                            {["Batman", "Godfather", "Inception", "Sci-Fi", "Action", "Thriller"].map((tag) => (
                              <button
                                key={tag}
                                className="suggestion-tag-btn"
                                type="button"
                                onClick={() => {
                                  setTitleQuery(tag);
                                  void runSearch("title", tag);
                                }}
                              >
                                {tag}
                              </button>
                            ))}
                          </div>
                        </div>

                        {latestMovies.length > 0 && (
                          <div>
                            <div className="section-title" style={{ fontSize: "0.95rem", margin: "16px 0 10px" }}>
                              <h2>Popular Searches</h2>
                            </div>
                            <div className="poster-grid" style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "12px" }}>
                              {latestMovies.slice(0, 4).map((movie) => (
                                <button
                                  type="button"
                                  key={`mobile-popular-${movie.id}`}
                                  className="popular-search-card"
                                  onClick={() => {
                                    setTitleQuery(selectTitleLabel(movie));
                                    selectMovie(movie, "title_search");
                                    void recommend(movie);
                                  }}
                                  style={{
                                    background: "rgba(255,255,255,0.02)",
                                    border: "1px solid rgba(255,255,255,0.05)",
                                    borderRadius: "16px",
                                    padding: "10px",
                                    cursor: "pointer",
                                    display: "flex",
                                    flexDirection: "column",
                                    gap: "8px",
                                    textAlign: "left",
                                    transition: "all 0.2s ease"
                                  }}
                                >
                                  <img
                                    src={posterUrl(movie.poster_path)}
                                    alt={movie.title}
                                    style={{ width: "100%", aspectRatio: "2/3", objectFit: "cover", borderRadius: "10px" }}
                                  />
                                  <div style={{ display: "flex", flexDirection: "column", gap: "2px", overflow: "hidden" }}>
                                    <strong style={{ fontSize: "0.8rem", color: "#fff", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{movie.title}</strong>
                                    <span style={{ fontSize: "0.75rem", color: "var(--muted)" }}>{movieYear(movie)}</span>
                                  </div>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )
                  )}
                </div>
              </section>
            </main>
          )}

          <React.Suspense fallback={<SuspenseFallback />}>
            {page === "dashboard" && <main className="app-shell inner-shell"><ErrorBoundary><Dashboard /></ErrorBoundary></main>}
            {page === "knowledge-graph" && <main className="app-shell inner-shell"><ErrorBoundary><KnowledgeGraphPage titles={titles} /></ErrorBoundary></main>}
            {page === "evaluation" && <main className="app-shell inner-shell"><ErrorBoundary><EvaluationPage /></ErrorBoundary></main>}
            {page === "profile" && (
              <main className="app-shell inner-shell">
                <ErrorBoundary>
                  <UserProfilePage
                    token={token}
                    username={username}
                    onRequestLogin={() => setShowAuthModal(true)}
                    onSelectMovie={(movie) => { setDialogMovie(movie); }}
                    onNavigate={(p) => setPage(p as AppPage)}
                  />
                </ErrorBoundary>
              </main>
            )}
            {page === "admin" && <main className="app-shell inner-shell"><ErrorBoundary><AdminPanel token={token} /></ErrorBoundary></main>}
          </React.Suspense>
        </div>

        {/* Bottom Navigation Bar */}
        <nav className="mobile-nav-bar" aria-label="Mobile navigation">
          <button
            className={`mobile-nav-tab ${page === "home" ? "active" : ""}`}
            type="button"
            onClick={() => { openHome(); setShowMoreDrawer(false); }}
          >
            <Film size={20} />
            <span>Browse</span>
          </button>
          <button
            className={`mobile-nav-tab ${page === "search" ? "active" : ""}`}
            type="button"
            onClick={() => { openSearch(); setShowMoreDrawer(false); }}
          >
            <Search size={20} />
            <span>Search</span>
          </button>
          <button
            className={`mobile-nav-tab ${page === "knowledge-graph" ? "active" : ""}`}
            type="button"
            onClick={() => { setPage("knowledge-graph"); setShowMoreDrawer(false); }}
          >
            <Network size={20} />
            <span>Graph</span>
          </button>
          <button
            className={`mobile-nav-tab ${page === "evaluation" ? "active" : ""}`}
            type="button"
            onClick={() => { setPage("evaluation"); setShowMoreDrawer(false); }}
          >
            <BarChart3 size={20} />
            <span>Eval</span>
          </button>
          <button
            className={`mobile-nav-tab ${["profile", "dashboard", "admin"].includes(page) ? "active" : ""}`}
            type="button"
            onClick={() => { setPage("profile"); setShowMoreDrawer(false); }}
          >
            <User size={20} />
            <span>My Space</span>
          </button>
        </nav>

        {/* Slide-up bottom sheet drawer for "More" options */}
        {showMoreDrawer && (
          <div className="mobile-bottom-sheet-overlay" onClick={() => setShowMoreDrawer(false)} role="presentation">
            {/* eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/click-events-have-key-events */}
            <div className="mobile-bottom-sheet" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-label="More navigation options">
              <div className="sheet-handle" />
              <h3 className="sheet-title">More Options</h3>
              <div className="sheet-links">
                {username && (
                  <button
                    className="sheet-link-btn"
                    type="button"
                    onClick={() => { setPage("profile"); setShowMoreDrawer(false); }}
                  >
                    <User size={18} />
                    <span>My Profile</span>
                  </button>
                )}
                <button
                  className="sheet-link-btn"
                  type="button"
                  onClick={() => { setPage("dashboard"); setShowMoreDrawer(false); }}
                >
                  <Activity size={18} />
                  <span>System Dashboard</span>
                </button>
                {username === "admin" && (
                  <button
                    className="sheet-link-btn"
                    type="button"
                    onClick={() => { setPage("admin"); setShowMoreDrawer(false); }}
                  >
                    <Server size={18} />
                    <span>Admin Panel</span>
                  </button>
                )}
                <button
                  className="sheet-link-btn"
                  type="button"
                  onClick={() => { setPage("status"); setShowMoreDrawer(false); }}
                >
                  <Database size={18} />
                  <span>System Status</span>
                </button>
                <button
                  className="sheet-link-btn"
                  type="button"
                  onClick={async () => {
                    if (deferredPrompt) {
                      deferredPrompt.prompt();
                      const { outcome } = await deferredPrompt.userChoice;
                      if (outcome === "accepted") {
                        setDeferredPrompt(null);
                      }
                    } else {
                      window.alert("To install this app on your phone:\n\n1. Tap the Share button in your mobile browser.\n2. Select 'Add to Home Screen'.\n3. Launch Nova directly from your home screen!");
                    }
                    setShowMoreDrawer(false);
                  }}
                  style={{ background: "rgba(16, 185, 129, 0.08)", borderColor: "rgba(16, 185, 129, 0.2)", color: "#34d399" }}
                >
                  <Sparkles size={18} />
                  <span>Download / Install App</span>
                </button>

                {isMobileSimulated && !isMobileViewport && (
                  <button
                    className="sheet-link-btn"
                    type="button"
                    onClick={() => { setIsMobileSimulated(false); setShowMoreDrawer(false); }}
                    style={{ border: "1px dashed rgba(167, 139, 250, 0.4)", color: "#a78bfa" }}
                  >
                    <Play size={18} />
                    <span>Exit Mobile Simulator</span>
                  </button>
                )}

                {username ? (
                  <button
                    className="sheet-link-btn logout"
                    type="button"
                    onClick={() => {
                      window.localStorage.removeItem("nova_jwt_token");
                      window.localStorage.removeItem("nova_username");
                      setToken(null);
                      setUsername(null);
                      setShowMoreDrawer(false);
                      openHome();
                    }}
                  >
                    <LogOut size={18} />
                    <span>Sign Out</span>
                  </button>
                ) : (
                  <button
                    className="sheet-link-btn"
                    type="button"
                    onClick={() => { setShowAuthModal(true); setShowMoreDrawer(false); }}
                    style={{ background: "var(--accent)" }}
                  >
                    <User size={18} />
                    <span>Sign In</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {dialogMovie && (
          <MovieDialog
            movie={dialogMovie}
            feedback={feedbackByMovieId[dialogMovie.id]}
            onFeedback={recordFeedback}
            onRating={recordRating}
            onClose={() => setDialogMovie(null)}
          />
        )}
        {showAuthModal && (
          <AuthModal
            onLogin={(tok, user) => { setToken(tok); setUsername(user); setShowAuthModal(false); }}
            onClose={() => setShowAuthModal(false)}
          />
        )}
      </div>
    );

    if (isMobileSimulated && !isMobileViewport) {
      return (
        <div className="phone-simulator-container">
          <div className="simulator-controls">
            <span className="simulator-badge">Simulator Mode</span>
            <button className="simulator-toggle-btn" type="button" onClick={() => setIsMobileSimulated(false)}>
              <Play size={14} style={{ transform: "rotate(180deg)" }} />
              Back to Widescreen
            </button>
          </div>
          <div className="iphone-mockup">
            <div className="dynamic-island" />
            <div className="simulator-status-bar">
              <span>9:41</span>
              <div className="status-bar-right" style={{ display: "flex", gap: "4px" }}>
                <Server size={10} />
                <Activity size={10} />
                <Database size={10} />
              </div>
            </div>
            {mobileContent}
            <div className="home-indicator" />
          </div>
        </div>
      );
    }

    return mobileContent;
  }

  // ── App Shell Pages (Browse, Search, Dashboard, KG, Eval, Profile, Admin) ─
  if (isAppPage) {
    const navLinks = [
      { id: "home", label: "Browse" },
      { id: "search", label: "Search" },
      { id: "vector-space", label: "3D Galaxy" },
      { id: "dashboard", label: "Dashboard" },
      { id: "knowledge-graph", label: "Knowledge Graph" },
      { id: "evaluation", label: "Evaluation" },
      { id: "profile", label: "Profile" },
      { id: "admin", label: "Admin" },
    ];

    return (
      <div className="app-container" id="main-content">
        <a href="#main-content" className="skip-link">Skip to main content</a>

        {/* Unified Sticky top navigation header */}
        <header className="topbar">
          <div className="topbar-left">
            <button className="brand-logo" type="button" onClick={openHome} aria-label="NOVA Home">NOVA</button>
            <nav className="topbar-links" aria-label="Main navigation">
              {navLinks.map((link) => (
                <button
                  key={link.id}
                  className={`topbar-link ${page === link.id ? "active" : ""}`}
                  type="button"
                  onClick={() => {
                    if (link.id === "home") openHome();
                    else if (link.id === "search") openSearch();
                    else setPage(link.id as AppPage);
                  }}
                >
                  {link.label}
                </button>
              ))}
            </nav>
          </div>
          <div className="topbar-right">
            {/* Header Search Box */}
            <div className="topbar-search">
              {mode === "semantic" ? <Sparkles size={14} className="mode-icon-semantic" /> : <Search size={14} />}
              <input
                type="text"
                placeholder={mode === "title" ? "Search movies..." : "Search by plot/desc..."}
                value={titleQuery}
                onChange={(event) => {
                  userStarted.current = true;
                  setMode(mode);
                  setTitleQuery(event.target.value);
                  if (page !== "search") {
                    setPage("search");
                  }
                  setTitleSelectOpen(true);
                  if (selectedMovie && event.target.value !== selectedTitleLabel) {
                    setSelectedMovie(null);
                    setResults([]);
                    setResultsKind("idle");
                    setRecommendationSource(null);
                    setDialogMovie(null);
                    setLastRecommendationRequestId(null);
                  }
                }}
                onFocus={() => {
                  if (page !== "search") {
                    setPage("search");
                  }
                  setTitleSelectOpen(true);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Escape") {
                    setTitleSelectOpen(false);
                    return;
                  }
                  if (event.key === "Enter") {
                    event.preventDefault();
                    setTitleSelectOpen(false);
                    void runSearch(mode);
                  }
                }}
              />
              <button
                className={`header-mode-toggle ${mode === "semantic" ? "semantic-active" : ""}`}
                type="button"
                title={mode === "title" ? "Switch to AI/Plot search" : "Switch to Title search"}
                onClick={() => {
                  setMode(mode === "title" ? "semantic" : "title");
                }}
                style={{
                  background: "transparent",
                  border: "none",
                  color: mode === "semantic" ? "#a78bfa" : "var(--quiet)",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  padding: "4px",
                  borderRadius: "50%",
                  transition: "all 0.2s ease"
                }}
              >
                <Sparkles size={14} />
              </button>
              {titleQuery && (
                <button
                  className="clear-search-btn"
                  type="button"
                  onClick={() => {
                    userStarted.current = true;
                    setTitleQuery("");
                    setSelectedMovie(null);
                    setResults([]);
                    setResultsKind("idle");
                    setRecommendationSource(null);
                    setDialogMovie(null);
                    setLastRecommendationRequestId(null);
                    setTitleSelectOpen(true);
                  }}
                  style={{
                    background: "transparent",
                    border: "none",
                    color: "var(--quiet)",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    padding: "2px"
                  }}
                >
                  <X size={14} />
                </button>
              )}
            </div>
            <StatusBadge state={catalogState} backend={backend} />
            <button
              className="icon-button"
              type="button"
              onClick={() => {
                void bootstrap(true);
                if (catalogState === "ready") loadOperationalSignals();
              }}
              title="Refresh catalog"
            >
              <RefreshCw size={16} />
            </button>
            {!isMobileViewport && (
              <button
                className="icon-button"
                type="button"
                onClick={() => setIsMobileSimulated(!isMobileSimulated)}
                title={isMobileSimulated ? "Switch to Widescreen Dashboard" : "Simulate Mobile App UI"}
                style={{
                  borderColor: isMobileSimulated ? "var(--secondary)" : "var(--line)",
                  color: isMobileSimulated ? "var(--secondary)" : "var(--muted)",
                  background: isMobileSimulated ? "rgba(236, 72, 153, 0.08)" : "var(--panel)"
                }}
              >
                <Activity size={16} />
              </button>
            )}
            {username ? (
              <div className="user-profile-menu">
                <button className="profile-btn" type="button" onClick={() => setPage("profile")}>
                  <User size={14} aria-hidden="true" /> <span className="profile-greet">Hi, </span><strong>{username}</strong>
                </button>
                <button
                  className="logout-btn"
                  type="button"
                  onClick={() => {
                    window.localStorage.removeItem("nova_jwt_token");
                    window.localStorage.removeItem("nova_username");
                    setToken(null);
                    setUsername(null);
                    openHome();
                  }}
                  title="Logout"
                  aria-label="Logout"
                >
                  <LogOut size={14} aria-hidden="true" />
                </button>
              </div>
            ) : (
              <button className="signin-nav-btn" type="button" onClick={() => setShowAuthModal(true)}>
                Sign In
              </button>
            )}
          </div>
        </header>

        <div className="app-content-area">
          {page === "home" && (
            <HomePage
              movies={homeMovies}
              heroIndex={homeHeroIndex}
              loading={homeLoading}
              error={homeError}
              onHeroIndex={setHomeHeroIndex}
              onOpenMovie={setDialogMovie}
              recentMovies={recentMovies}
              forYouMovies={forYouMovies}
              forYouLoading={forYouLoading}
              latestMovies={latestMovies}
              latestLoading={latestLoading}
              homeMode={homeMode}
              onToggleMode={setHomeMode}
            />
          )}

          {page === "search" && (
            <main className="search-page-layout">
              <section className="metrics-strip" aria-label="Platform snapshot">
                <MetricTile icon={<Database size={18} />} label="Catalog" value={catalogValue ? catalogValue.toLocaleString() : "Loading"} />
                <MetricTile icon={<Server size={18} />} label="Readiness" value={readinessLabel(readinessReport)} />
                <MetricTile icon={<BarChart3 size={18} />} label="Ranking" value={rankerValue} />
                <MetricTile icon={<Gauge size={18} />} label="Quality" value={qualityLabel(qualityReport)} />
                <MetricTile icon={<Activity size={18} />} label="Artifacts" value={healthLabel(artifactReport)} />
              </section>

              <section className="workspace">
                <div className="control-panel">

                  <div className="control-heading">
                    <Search size={44} />
                    <h1>Search & Discover</h1>
                  </div>

                  <div className="search-mode-tabs" style={{ display: "flex", gap: "8px", marginBottom: "20px" }}>
                    <button
                      className={`search-mode-tab ${mode === "title" ? "active" : ""}`}
                      type="button"
                      onClick={() => {
                        setMode("title");
                        userStarted.current = true;
                      }}
                      style={{
                        padding: "10px 20px",
                        borderRadius: "24px",
                        border: "1px solid " + (mode === "title" ? "rgba(99, 102, 241, 0.4)" : "rgba(255, 255, 255, 0.08)"),
                        background: mode === "title" ? "rgba(99, 102, 241, 0.15)" : "rgba(255, 255, 255, 0.02)",
                        color: mode === "title" ? "#a5b4fc" : "var(--muted)",
                        fontWeight: 600,
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        transition: "all 0.2s cubic-bezier(0.16, 1, 0.3, 1)"
                      }}
                    >
                      <Search size={14} />
                      Title Search
                    </button>
                    <button
                      className={`search-mode-tab ${mode === "semantic" ? "active" : ""}`}
                      type="button"
                      onClick={() => {
                        setMode("semantic");
                        userStarted.current = true;
                      }}
                      style={{
                        padding: "10px 20px",
                        borderRadius: "24px",
                        border: "1px solid " + (mode === "semantic" ? "rgba(167, 139, 250, 0.4)" : "rgba(255, 255, 255, 0.08)"),
                        background: mode === "semantic" ? "rgba(167, 139, 250, 0.15)" : "rgba(255, 255, 255, 0.02)",
                        color: mode === "semantic" ? "#c084fc" : "var(--muted)",
                        fontWeight: 600,
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        transition: "all 0.2s cubic-bezier(0.16, 1, 0.3, 1)"
                      }}
                    >
                      <Sparkles size={14} />
                      AI Semantic Search
                    </button>
                  </div>

                  <div
                    className="title-select"
                    ref={titleSelectRef}
                    onBlur={(event) => {
                      if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                        setTitleSelectOpen(false);
                      }
                    }}
                  >
                    <div className="search-box title-select-box" style={{ display: "flex", alignItems: "center", width: "100%", position: "relative" }}>
                      {mode === "semantic" ? <Sparkles size={18} className="mode-icon-semantic" style={{ color: "#a78bfa", flexShrink: 0 }} /> : <Search size={18} style={{ color: "var(--quiet)", flexShrink: 0 }} />}
                      <input
                        id="title-search"
                        value={titleQuery}
                        onChange={(event) => {
                          userStarted.current = true;
                          setMode(mode);
                          setTitleQuery(event.target.value);
                          setTitleSelectOpen(true);
                          if (selectedMovie && event.target.value !== selectedTitleLabel) {
                            setSelectedMovie(null);
                            setResults([]);
                          }
                        }}
                        onFocus={() => setTitleSelectOpen(true)}
                        onKeyDown={(event) => {
                          if (event.key === "Escape") {
                            setTitleSelectOpen(false);
                            return;
                          }
                          if (event.key === "Enter") {
                            event.preventDefault();
                            setTitleSelectOpen(false);
                            void runSearch(mode);
                          }
                        }}
                        placeholder={mode === "title" ? "Search by title, e.g. Inception..." : "Describe a plot, mood, or genre..."}
                        style={{ marginLeft: "8px", flex: 1 }}
                      />
                      {hasTitleQuery && (
                        <button
                          className="clear-title"
                          type="button"
                          aria-label="Clear selected title"
                          onClick={() => {
                            userStarted.current = true;
                            setTitleQuery("");
                            setSelectedMovie(null);
                            setResults([]);
                            setResultsKind("idle");
                            setRecommendationSource(null);
                            setDialogMovie(null);
                            setLastRecommendationRequestId(null);
                            setTitleSelectOpen(true);
                          }}
                          style={{ background: "transparent", border: "none", color: "var(--muted)", cursor: "pointer", display: "grid", placeItems: "center", padding: "4px", marginRight: "8px" }}
                        >
                          <X size={16} />
                        </button>
                      )}
                      <button
                        className="search-btn-primary"
                        type="button"
                        onClick={() => runSearch(mode)}
                        aria-label="Search"
                        style={{
                          background: mode === "semantic" ? "linear-gradient(135deg, #a78bfa, #818cf8)" : "linear-gradient(135deg, #6366f1, #4f46e5)",
                          color: "#fff",
                          border: "none",
                          borderRadius: "8px",
                          padding: "8px 20px",
                          fontWeight: 700,
                          cursor: "pointer",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "8px",
                          boxShadow: "0 4px 12px rgba(99, 102, 241, 0.2)",
                          transition: "all 0.2s ease",
                          flexShrink: 0
                        }}
                      >
                        {mode === "semantic" ? <Sparkles size={14} /> : <Search size={14} />}
                        <span>Search</span>
                      </button>
                    </div>

                    {showTitleSuggestions && mode === "title" && (
                      <div className="title-list streamlit-title-list">
                        {titles.length === 0 && catalogState !== "ready" && <span className="quiet-line">Loading movie catalog...</span>}
                        {filteredTitles.slice(0, 12).map((item) => (
                          <button
                            type="button"
                            key={`${item.id}-${item.title}`}
                            onMouseDown={(event) => event.preventDefault()}
                            onClick={() => void chooseTitle(item, { autoRecommend: true })}
                          >
                            {item.title}
                          </button>
                        ))}
                        {titles.length > 0 && filteredTitles.length === 0 && <span className="quiet-line">No title match. Try AI search mode.</span>}
                      </div>
                    )}
                  </div>

                  {showNotice && (
                    <div className={`notice ${catalogState}`}>
                      <span>{notice}</span>
                      <button type="button" onClick={() => void bootstrap(true)}>
                        Retry now
                      </button>
                    </div>
                  )}

                  <details className="ops-details">
                    <summary>
                      <span>Platform signals</span>
                      <small>{readinessLabel(readinessReport)} readiness</small>
                    </summary>
                    <div className="ops-stack">
                      <ReadinessPanel report={readinessReport} loading={signalsLoading} onRefresh={loadOperationalSignals} />
                      <DiagnosticsPanel health={artifactReport} />
                      <QualityPanel report={qualityReport} />
                    </div>
                  </details>
                </div>

                <div className="result-panel">
                  {selectedMovie && !isEditingTitle ? (
                    <MovieSpotlight
                      movie={selectedMovie}
                      loading={loadingRecs}
                      onRecommend={() => void recommend()}
                      userId={username}
                      sessionId={sessionId}
                    />
                  ) : null}

                  {isSearching ? (
                    <section className="results-section">
                      <div className="section-title">
                        <div>
                          <span>{mode === "semantic" ? "AI Semantic Search" : "Catalog Search"}</span>
                          <h2>{mode === "semantic" ? "Searching catalog by intent..." : "Searching movie database..."}</h2>
                        </div>
                      </div>
                      <div className="poster-grid">
                        {Array.from({ length: 6 }).map((_, index) => (
                          <div key={index} className="recommendation-card skeleton-card" style={{ height: "380px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: "16px", padding: "16px", display: "flex", flexDirection: "column", gap: "12px" }}>
                            <div className="skeleton" style={{ height: "180px", borderRadius: "12px" }}></div>
                            <div className="skeleton" style={{ height: "24px", width: "80%", borderRadius: "6px" }}></div>
                            <div className="skeleton" style={{ height: "16px", width: "40%", borderRadius: "4px" }}></div>
                            <div className="skeleton" style={{ height: "48px", width: "100%", borderRadius: "8px" }}></div>
                            <div style={{ display: "flex", gap: "8px", marginTop: "auto" }}>
                              <div className="skeleton" style={{ height: "28px", width: "70px", borderRadius: "20px" }}></div>
                              <div className="skeleton" style={{ height: "28px", width: "70px", borderRadius: "20px" }}></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>
                  ) : loadingRecs ? (
                    <section className="results-section">
                      <div className="section-title">
                        <div>
                          <span>Ensemble pipeline</span>
                          <h2>Generating recommendations...</h2>
                        </div>
                      </div>
                      <div className="poster-grid">
                        {Array.from({ length: 6 }).map((_, index) => (
                          <div key={index} className="recommendation-card skeleton-card" style={{ height: "380px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: "16px", padding: "16px", display: "flex", flexDirection: "column", gap: "12px" }}>
                            <div className="skeleton" style={{ height: "180px", borderRadius: "12px" }}></div>
                            <div className="skeleton" style={{ height: "24px", width: "80%", borderRadius: "6px" }}></div>
                            <div className="skeleton" style={{ height: "16px", width: "40%", borderRadius: "4px" }}></div>
                            <div className="skeleton" style={{ height: "48px", width: "100%", borderRadius: "8px" }}></div>
                            <div style={{ display: "flex", gap: "8px", marginTop: "auto" }}>
                              <div className="skeleton" style={{ height: "28px", width: "70px", borderRadius: "20px" }}></div>
                              <div className="skeleton" style={{ height: "28px", width: "70px", borderRadius: "20px" }}></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>
                  ) : results.length > 0 ? (
                    <section className="results-section">
                      <div className="section-title">
                        <div>
                          <span>{resultsKind === "recommendations" ? "Ranked set" : "Catalog results"}</span>
                          <h2>{resultHeading}</h2>
                          {feedbackNotice && <small className="feedback-status">{feedbackNotice}</small>}
                        </div>
                        <strong>{results.length} titles</strong>
                      </div>
                      <ResultContextBar
                        kind={resultsKind}
                        backend={backend}
                        sourceMovie={recommendationSource || selectedMovie}
                        requestId={lastRecommendationRequestId}
                        query={activeQuery}
                      />
                      <div className="poster-grid">
                        {results.map((movie, index) => (
                          <RecommendationCard
                            key={`${movie.id}-${movie.title}`}
                            movie={movie}
                            rank={index + 1}
                            onSelect={selectResultMovie}
                            feedback={feedbackByMovieId[movie.id]}
                            onFeedback={recordFeedback}
                          />
                        ))}
                      </div>
                    </section>
                  ) : results.length === 0 && resultsKind === "search" && titleQuery.trim() && !isSearching && !isSelecting ? (
                    <section className="results-section no-results">
                      <div style={{ textAlign: "center", padding: "48px 24px", color: "var(--muted)" }}>
                        <Film size={48} style={{ marginBottom: "16px", opacity: 0.5 }} />
                        <h3>No matches found for &quot;{titleQuery}&quot;</h3>
                        <p style={{ fontSize: "0.9rem" }}>Try checking your spelling or describe a plot using our AI Search mode!</p>
                      </div>
                    </section>
                  ) : resultsKind === "idle" && titleQuery.trim() && !isSearching && !isSelecting ? (
                    <section className="results-section no-results">
                      <div style={{ textAlign: "center", padding: "48px 24px", color: "var(--muted)" }}>
                        <Sparkles size={48} className="mode-icon-semantic" style={{ marginBottom: "16px", opacity: 0.5, color: "var(--accent)" }} />
                        <h3>Press Enter to search by intent</h3>
                        <p style={{ fontSize: "0.9rem" }}>Type your query and press Enter, or wait a moment for AI Search to automatically run.</p>
                      </div>
                    </section>
                  ) : !titleQuery.trim() ? (
                    <section className="results-section">
                      <div className="section-title">
                        <div>
                          <span>Trending</span>
                          <h2>Popular Searches</h2>
                        </div>
                      </div>
                      <div className="poster-grid">
                        {latestMovies.slice(0, 6).map((movie) => (
                          <button
                            type="button"
                            key={`popular-${movie.id}`}
                            className="popular-search-card"
                            onClick={() => {
                              setTitleQuery(selectTitleLabel(movie));
                              selectMovie(movie, "title_search");
                              void recommend(movie);
                            }}
                            style={{
                              background: "rgba(255,255,255,0.02)",
                              border: "1px solid rgba(255,255,255,0.05)",
                              borderRadius: "16px",
                              padding: "12px",
                              cursor: "pointer",
                              display: "flex",
                              flexDirection: "column",
                              gap: "8px",
                              textAlign: "left",
                              transition: "all 0.2s ease"
                            }}
                          >
                            <img
                              src={posterUrl(movie.poster_path)}
                              alt={movie.title}
                              style={{ width: "100%", aspectRatio: "2/3", objectFit: "cover", borderRadius: "10px" }}
                            />
                            <strong style={{ fontSize: "0.9rem", color: "#fff", display: "-webkit-box", WebkitLineClamp: 1, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{movie.title}</strong>
                            <span style={{ fontSize: "0.78rem", color: "var(--muted)" }}>{movieYear(movie)}</span>
                          </button>
                        ))}
                      </div>

                      {recentMovies.length > 0 && (
                        <div className="recent-picks-section" style={{ marginTop: "24px" }}>
                          <div className="section-title" style={{ marginBottom: "12px" }}>
                            <div>
                              <span>Recent</span>
                              <h2>Recent Picks</h2>
                            </div>
                          </div>
                          <div className="recent-list" style={{ display: "flex", gap: "12px", overflowX: "auto", paddingBottom: "8px" }}>
                            {recentMovies.map((movie) => (
                              <button
                                type="button"
                                key={`recent-${movie.id}`}
                                onClick={() => {
                                  setTitleQuery(selectTitleLabel(movie));
                                  selectMovie(movie, "recent_pick");
                                  void recommend(movie);
                                }}
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "12px",
                                  padding: "8px 16px",
                                  background: "rgba(255, 255, 255, 0.02)",
                                  border: "1px solid rgba(255, 255, 255, 0.05)",
                                  borderRadius: "12px",
                                  cursor: "pointer",
                                  flex: "0 0 auto",
                                  maxWidth: "240px",
                                  textAlign: "left"
                                }}
                              >
                                <img
                                  src={posterUrl(movie.poster_path)}
                                  alt=""
                                  loading="lazy"
                                  style={{ width: "32px", aspectRatio: "2/3", objectFit: "cover", borderRadius: "4px" }}
                                />
                                <span style={{ fontSize: "0.85rem", color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{movie.title}</span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </section>
                  ) : null}
                </div>
              </section>
            </main>
          )}

          <React.Suspense fallback={<SuspenseFallback />}>
            {page === "vector-space" && <main className="app-shell inner-shell"><ErrorBoundary><VectorSpace /></ErrorBoundary></main>}
            {page === "dashboard" && <main className="app-shell inner-shell"><ErrorBoundary><Dashboard /></ErrorBoundary></main>}
            {page === "knowledge-graph" && <main className="app-shell inner-shell"><ErrorBoundary><KnowledgeGraphPage titles={titles} /></ErrorBoundary></main>}
            {page === "evaluation" && <main className="app-shell inner-shell"><ErrorBoundary><EvaluationPage /></ErrorBoundary></main>}
            {page === "profile" && (
              <main className="app-shell inner-shell">
                <ErrorBoundary>
                  <UserProfilePage
                    token={token}
                    username={username}
                    onRequestLogin={() => setShowAuthModal(true)}
                    onSelectMovie={(movie) => { setDialogMovie(movie); }}
                    onNavigate={(p) => setPage(p as AppPage)}
                  />
                </ErrorBoundary>
              </main>
            )}
            {page === "admin" && <main className="app-shell inner-shell"><ErrorBoundary><AdminPanel token={token} /></ErrorBoundary></main>}
          </React.Suspense>
        </div>

        {dialogMovie && (
          <MovieDialog
            movie={dialogMovie}
            feedback={feedbackByMovieId[dialogMovie.id]}
            onFeedback={recordFeedback}
            onRating={recordRating}
            onClose={() => setDialogMovie(null)}
          />
        )}
        {showAuthModal && (
          <AuthModal
            onLogin={(tok, user) => { setToken(tok); setUsername(user); setShowAuthModal(false); }}
            onClose={() => setShowAuthModal(false)}
          />
        )}
      </div>
    );
  }

  return null;
}

const rootElement = document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
}

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/sw.js")
      // eslint-disable-next-line no-console
      .then((reg) => console.log("Service Worker registered successfully:", reg.scope))
      .catch((err) => console.error("Service Worker registration failed:", err));
  });
}
