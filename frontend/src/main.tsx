import React from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  ChevronDown,
  CheckCircle2,
  Clock3,
  Database,
  Film,
  Gauge,
  House,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Server,
  Sparkles,
  Star,
  ThumbsDown,
  ThumbsUp,
  TrendingUp,
  WandSparkles,
  X,
} from "lucide-react";
import {
  artifactHealth,
  aiSearch,
  backendLabel,
  currentBackend,
  getMovie,
  getRecommendations,
  loadTitles,
  pingApi,
  platformReadiness,
  platformStatus,
  recordEvent,
  searchMovies,
  semanticBenchmark,
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

const imageBase = import.meta.env.VITE_TMDB_IMAGE_BASE || "https://image.tmdb.org/t/p/w500";
const RECENT_STORAGE_KEY = "nova_recent_movies_v2";
const SESSION_STORAGE_KEY = "nova_session_id_v1";
const TITLE_CATALOG_LIMIT = 5000;

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

function ratingPercent(movie: Movie): number {
  const value = Number(movie.vote_average || 0);
  return Math.max(0, Math.min(100, Math.round(value * 10)));
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

function MoviePoster({
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
}

function RecommendationCard({
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
}

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

function MovieSpotlight({
  movie,
  loading,
  onRecommend,
}: {
  movie: Movie;
  loading: boolean;
  onRecommend: () => void;
}) {
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
}

function MovieDialog({ movie, onClose }: { movie: Movie; onClose: () => void }) {
  const director = directorLabel(movie);
  const cast = movie.cast || "";
  const genres = compactGenres(movie.genres);
  const primaryGenre = genres.split("/")[0]?.trim() || "Catalog";
  const runtime = movie.runtime ? `${movie.runtime} min` : "";
  const meta = [movieYear(movie), runtime, primaryGenre].filter(Boolean).join(" • ");
  const overview = movie.overview || "No overview is available for this title.";
  const shortOverview = overview.length > 240 ? `${overview.slice(0, 240).replace(/\s+\S*$/, "")}...` : overview;
  const explanation = movie.explanation_text || movieReasons(movie).join(" | ");
  const rating = movieScore(movie);
  const scorePercent = ratingPercent(movie);
  const ratingColor = Number(movie.vote_average || 0) >= 7 ? "#21d07a" : Number(movie.vote_average || 0) >= 5 ? "#d2d531" : "#db2360";

  React.useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.body.classList.add("modal-open");
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.classList.remove("modal-open");
      window.removeEventListener("keydown", onKeyDown);
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
      <section className="movie-dialog" role="dialog" aria-modal="true" aria-label={`${movie.title} details`}>
        <button className="dialog-close" type="button" aria-label="Close movie details" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="dialog-media" aria-hidden="true">
          {movie.trailer_key ? (
            <iframe
              title={`${movie.title} trailer`}
              src={`https://www.youtube.com/embed/${movie.trailer_key}?controls=0&autoplay=1&mute=1&loop=1&playlist=${movie.trailer_key}&modestbranding=1&showinfo=0&rel=0&iv_load_policy=3&disablekb=1`}
              allow="autoplay; encrypted-media"
            />
          ) : (
            <img src={backdropUrl(movie.poster_path)} alt="" />
          )}
        </div>

        <div className="dialog-content">
          <div className="dialog-title-row">
            <h2>{movie.title}</h2>
            <div
              className="rating-circle"
              style={
                {
                  "--rating-percent": scorePercent,
                  "--rating-color": ratingColor,
                } as React.CSSProperties
              }
              aria-label={`Rating ${rating} out of 10`}
            >
              <span>{rating}</span>
            </div>
          </div>

          <div className="dialog-meta">{meta}</div>
          <p className="dialog-overview">{shortOverview}</p>

          {explanation && (
            <div className="dialog-explanation">
              <strong>CineBot Vibe Check:</strong> {explanation}
            </div>
          )}

          {(director || cast) && (
            <div className="dialog-credits">
              {director && (
                <span>
                  Directed by <strong>{director}</strong>
                </span>
              )}
              {cast && (
                <span>
                  Cast: <strong>{cast}</strong>
                </span>
              )}
            </div>
          )}

          {movie.trailer_key && (
            <a className="dialog-trailer" href={`https://www.youtube.com/watch?v=${movie.trailer_key}`} target="_blank" rel="noreferrer">
              <Play size={16} />
              Open trailer
            </a>
          )}
        </div>
      </section>
    </div>
  );
}

function App() {
  const [titles, setTitles] = React.useState<MovieTitle[]>([]);
  const [titleQuery, setTitleQuery] = React.useState("");
  const [semanticQuery, setSemanticQuery] = React.useState("");
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
  const [lastUpdated, setLastUpdated] = React.useState("");
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
  const [sessionId] = React.useState(() => getSessionId());
  const titleSelectRef = React.useRef<HTMLDivElement>(null);
  const bootstrapped = React.useRef(false);
  const loadedPlatform = React.useRef(false);
  const autoSeeded = React.useRef(false);
  const userStarted = React.useRef(false);

  const activeQuery = mode === "title" ? titleQuery : semanticQuery;
  const hasTitleQuery = titleQuery.trim().length > 0;
  const selectedTitleLabel = selectedMovie ? selectTitleLabel(selectedMovie) : "";
  const isSelectedTitleQuery = Boolean(selectedMovie && titleQuery === selectedTitleLabel);
  const isEditingTitle = Boolean(selectedMovie && hasTitleQuery && !isSelectedTitleQuery);
  const showTitleSuggestions = titleSelectOpen || (hasTitleQuery && !isSelectedTitleQuery);
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
      metadata: {
        client: "react",
        surface: "web",
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
      const starter =
        result.data.find((item) => item.title.startsWith("Avatar (2009)")) ||
        result.data.find((item) => item.title.toLowerCase().startsWith("avatar")) ||
        result.data[0];
      if (!autoSeeded.current && !userStarted.current && !selectedMovie && !titleQuery.trim() && starter) {
        autoSeeded.current = true;
        window.setTimeout(() => void chooseTitle(starter, { track: false, autoRecommend: true }), 150);
      }
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
    if (catalogState !== "warming") return;
    const delay = Math.min(30000, 7000 + retryCount * 4000);
    const retry = window.setTimeout(() => void bootstrap(true), delay);
    return () => window.clearTimeout(retry);
  }, [catalogState, retryCount]);

  React.useEffect(() => {
    if (catalogState !== "ready" || loadedPlatform.current) return;
    loadedPlatform.current = true;
    const timer = window.setTimeout(() => {
      loadOperationalSignals();
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [catalogState]);

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
    const query = (queryOverride ?? (searchMode === "title" ? titleQuery : semanticQuery)).trim();
    if (!query) return;

    if (searchMode === "title" && !queryOverride && isSelectedTitleQuery && selectedMovie) {
      await recommend(selectedMovie);
      return;
    }

    if (searchMode === "title" && !queryOverride && filteredTitles.length > 0) {
      await chooseTitle(filteredTitles[0]);
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
      setNotice(`${recommendations.length} recommendations ranked`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Recommendations unavailable");
    } finally {
      setLoadingRecs(false);
    }
  }

  const catalogValue = platform?.movie_count || titles.length;
  const rankerValue = platform?.ranker?.available ? "Learned" : "Hybrid";

  return (
    <main className="app-shell">
      <header className="topbar">
        <button
          className="home-button"
          type="button"
          onClick={() => {
            setResults([]);
            setResultsKind("idle");
            setSelectedMovie(null);
            setRecommendationSource(null);
            setDialogMovie(null);
            setLastRecommendationRequestId(null);
            setFeedbackByMovieId({});
            setFeedbackNotice("");
            setTitleQuery("");
            setSemanticQuery("");
            setMode("title");
          }}
        >
          <House size={18} />
          Home
        </button>
        <div className="topbar-actions">
          <StatusBadge state={catalogState} backend={backend} />
          <button
            className="icon-button"
            type="button"
            onClick={() => {
              void bootstrap(true);
              if (catalogState === "ready") loadOperationalSignals();
            }}
            title="Refresh catalog and readiness"
          >
            <RefreshCw size={18} />
          </button>
        </div>
      </header>

      <section className="metrics-strip" aria-label="Platform snapshot">
        <MetricTile icon={<Database size={18} />} label="Catalog" value={catalogValue ? catalogValue.toLocaleString() : "Loading"} />
        <MetricTile icon={<Server size={18} />} label="Readiness" value={readinessLabel(readinessReport)} />
        <MetricTile icon={<BarChart3 size={18} />} label="Ranking" value={rankerValue} />
        <MetricTile icon={<Gauge size={18} />} label="Quality" value={qualityLabel(qualityReport)} />
        <MetricTile icon={<Activity size={18} />} label="Artifacts" value={healthLabel(artifactReport)} />
      </section>

      <section className="workspace">
        <aside className="control-panel">
          <div className="control-heading">
            <Search size={54} />
            <h1>Search & Discover</h1>
          </div>

          <label className="field-label" htmlFor="title-search">
            Search by title
          </label>
          <div
            className="title-select"
            ref={titleSelectRef}
            onBlur={(event) => {
              if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                setTitleSelectOpen(false);
              }
            }}
          >
            <div className="search-box title-select-box">
              <input
                id="title-search"
                value={titleQuery}
                onChange={(event) => {
                  userStarted.current = true;
                  setMode("title");
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
                    if (filteredTitles[0]) void chooseTitle(filteredTitles[0]);
                    else void runSearch("title");
                  }
                }}
                placeholder="e.g. Inception, Avatar, The Dark Knight..."
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
                >
                  <X size={16} />
                </button>
              )}
              <button
                className="select-chevron"
                type="button"
                aria-label="Open movie title options"
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => setTitleSelectOpen((open) => !open)}
              >
                <ChevronDown size={19} />
              </button>
            </div>

            {showTitleSuggestions && (
              <div className="title-list streamlit-title-list">
                {titles.length === 0 && catalogState !== "ready" && <span className="quiet-line">Loading movie catalog...</span>}
                {filteredTitles.slice(0, 12).map((item) => (
                  <button
                    type="button"
                    key={`${item.id}-${item.title}`}
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => void chooseTitle(item)}
                  >
                    {item.title}
                  </button>
                ))}
                {titles.length > 0 && filteredTitles.length === 0 && <span className="quiet-line">No title match. Try semantic search below.</span>}
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

          <details className="semantic-expander">
            <summary>
              <Search size={18} />
              <span>Search by plot, genre, or description</span>
            </summary>
            <div className="semantic-body">
              <p>Can't find the title? Describe the movie by mood, plot, genre, or viewing intent.</p>
              <div className="search-box">
                <Sparkles size={18} />
                <input
                  id="semantic-search"
                  value={semanticQuery}
                  onChange={(event) => {
                    userStarted.current = true;
                    setMode("semantic");
                    setSemanticQuery(event.target.value);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void runSearch("semantic");
                  }}
                  placeholder="time travel heist, alien civilization, romantic comedy in Paris"
                />
              </div>
              <button className="secondary-action" type="button" onClick={() => void runSearch("semantic")} disabled={isSearching || isSelecting}>
                {isSearching || isSelecting ? <Loader2 size={18} className="spin" /> : <Sparkles size={18} />}
                Search by intent
              </button>
            </div>
          </details>

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

          {!hasTitleQuery && recentMovies.length > 0 && (
            <div className="recent-strip">
              <div className="section-mini-title">
                <span>Recent picks</span>
                <small>Seed</small>
              </div>
              <div className="recent-list">
                {recentMovies.map((movie) => (
                  <button type="button" key={`${movie.id}-${movie.title}`} onClick={() => selectMovie(movie, "recent_pick")}>
                    <img src={posterUrl(movie.poster_path)} alt="" loading="lazy" />
                    <span>{movie.title}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </aside>

        <section className="result-panel">
          {selectedMovie && !isEditingTitle ? (
            <MovieSpotlight
              movie={selectedMovie}
              loading={loadingRecs}
              onRecommend={() => void recommend()}
            />
          ) : null}

          {results.length > 0 && (
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
          )}
        </section>
      </section>
      {dialogMovie && <MovieDialog movie={dialogMovie} onClose={() => setDialogMovie(null)} />}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
