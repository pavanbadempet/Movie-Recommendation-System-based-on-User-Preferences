import React, { useEffect, useState } from "react";
import { Database, Film, Loader2, Star, User, Activity, Network, Settings, BarChart3 } from "lucide-react";
import { getUserRecommendations, apiGet } from "../api";
import type { Movie } from "../types";

const imageBase = "https://image.tmdb.org/t/p/w500";

function posterUrl(path?: string | null): string {
  if (!path) return "https://placehold.co/500x750/141418/f8fafc?text=Movie";
  if (path.startsWith("http")) return path;
  return `${imageBase}${path}`;
}

// ─── Login Prompt ─────────────────────────────────────────────────────────────

function LoginPrompt({ onLogin }: { onLogin: () => void }) {
  return (
    <div className="profile-login-prompt" role="status">
      <User size={40} aria-hidden="true" />
      <h3>Sign in to view your profile</h3>
      <p>Your personalized recommendations and watch history are available after signing in.</p>
      <button className="primary-action" type="button" onClick={onLogin}>
        Sign In
      </button>
    </div>
  );
}

// ─── Behavior Card ────────────────────────────────────────────────────────────

type BehaviorFeatures = {
  total_ratings?: number | null;
  avg_rating?: number | null;
  click_count?: number | null;
  view_count?: number | null;
  [key: string]: unknown;
};

function BehaviorCard({ features }: { features: BehaviorFeatures }) {
  function safeVal(v: number | null | undefined): string {
    if (v == null || !Number.isFinite(v) || v < 0) return "—";
    return v.toLocaleString();
  }

  const rows: { label: string; value: string; ariaLabel: string }[] = [
    {
      label: "Total Ratings",
      value: safeVal(features.total_ratings),
      ariaLabel: `Total ratings: ${safeVal(features.total_ratings)}`,
    },
    {
      label: "Avg Rating",
      value:
        features.avg_rating != null && Number.isFinite(features.avg_rating) && features.avg_rating >= 0
          ? features.avg_rating.toFixed(1)
          : "—",
      ariaLabel: `Average rating: ${features.avg_rating?.toFixed(1) ?? "—"}`,
    },
    {
      label: "Clicks",
      value: safeVal(features.click_count),
      ariaLabel: `Click count: ${safeVal(features.click_count)}`,
    },
    {
      label: "Views",
      value: safeVal(features.view_count),
      ariaLabel: `View count: ${safeVal(features.view_count)}`,
    },
  ];

  return (
    <div className="dashboard-card" aria-label="Behavior statistics">
      <h3 className="dashboard-card-title">
        <Star size={16} aria-hidden="true" />
        Behavior Statistics
      </h3>
      <dl className="hardware-dl">
        {rows.map((row) => (
          <div className="hardware-row" key={row.label}>
            <dt>{row.label}</dt>
            <dd aria-label={row.ariaLabel}>{row.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

// ─── Recommendation Card (mini) ───────────────────────────────────────────────

function MiniRecommendationCard({
  movie,
  onSelect,
}: {
  movie: Movie;
  onSelect: (movie: Movie) => void;
}) {
  return (
    <article className="recommendation-card" aria-label={`Recommended: ${movie.title}`}>
      <div className="card-media">
        <button
          className="poster-card"
          type="button"
          onClick={() => onSelect(movie)}
          aria-label={`View details for ${movie.title}`}
        >
          <img
            src={posterUrl(movie.poster_path)}
            alt={`Poster for ${movie.title}`}
            loading="lazy"
          />
        </button>
      </div>
      <div className="recommendation-body">
        <div className="card-title-row">
          <strong>{movie.title}</strong>
          <span>{movie.release_date?.slice(0, 4) ?? ""}</span>
        </div>
        {movie.genres && <div className="genre-line">{movie.genres.split(",")[0]?.trim()}</div>}
      </div>
    </article>
  );
}

// ─── Watch History Section ────────────────────────────────────────────────────

const RECENT_STORAGE_KEY = "nova_recent_movies_v2";

function loadRecentMovies(): Movie[] {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(RECENT_STORAGE_KEY) || "[]") as Movie[];
    return Array.isArray(parsed) ? parsed.slice(0, 6) : [];
  } catch {
    return [];
  }
}

function WatchHistorySection({ onSelectMovie }: { onSelectMovie: (movie: Movie) => void }) {
  const recent = loadRecentMovies();
  if (recent.length === 0) {
    return (
      <p className="eval-meta" role="status">
        No watch history yet. Browse and click movies to build your history.
      </p>
    );
  }
  return (
    <>
      <p className="eval-meta">Stored in your browser — {recent.length} title{recent.length !== 1 ? "s" : ""} recently viewed.</p>
      <div className="poster-grid" aria-label="Recently viewed movies" style={{ marginTop: "12px" }}>
        {recent.map((movie) => (
          <MiniRecommendationCard key={movie.id} movie={movie} onSelect={onSelectMovie} />
        ))}
      </div>
    </>
  );
}

// ─── User Profile Page ────────────────────────────────────────────────────────

export function UserProfilePage({
  token,
  username,
  onRequestLogin,
  onSelectMovie,
  onNavigate,
}: {
  token: string | null;
  username: string | null;
  onRequestLogin: () => void;
  onSelectMovie: (movie: Movie) => void;
  onNavigate?: (page: string) => void;
}) {
  const [features, setFeatures] = useState<BehaviorFeatures | null>(null);
  const [featuresLoading, setFeaturesLoading] = useState(false);
  const [featuresError, setFeaturesError] = useState<string | null>(null);

  const [recs, setRecs] = useState<Movie[]>([]);
  const [recsLoading, setRecsLoading] = useState(false);
  const [recsError, setRecsError] = useState<string | null>(null);

  useEffect(() => {
    if (!token || !username) return;

    setFeaturesLoading(true);
    setRecsLoading(true);

    apiGet<BehaviorFeatures>("/v1/events/features", {}, 15000)
      .then((res) => setFeatures(res.data))
      .catch((err) => setFeaturesError(err instanceof Error ? err.message : "Unavailable"))
      .finally(() => setFeaturesLoading(false));

    getUserRecommendations(username, 10)
      .then((res) => setRecs(res.data ?? []))
      .catch((err) => setRecsError(err instanceof Error ? err.message : "Unavailable"))
      .finally(() => setRecsLoading(false));
  }, [token, username]);

  if (!token) {
    return (
      <section className="profile-shell" aria-labelledby="profile-heading">
        <h2 id="profile-heading" className="visually-hidden">User Profile</h2>
        <LoginPrompt onLogin={onRequestLogin} />
      </section>
    );
  }

  return (
    <section className="profile-shell" aria-labelledby="profile-heading">
      <div className="dashboard-header">
        <div>
          <h2 id="profile-heading">
            <User size={20} aria-hidden="true" />
            {username ? `Hi, ${username}` : "Your Profile"}
          </h2>
          <p className="dashboard-subtitle">Your personalized recommendations and interaction history.</p>
        </div>
      </div>

      {/* Behavior stats */}
      {featuresLoading && (
        <div role="status" aria-live="polite" className="eval-loading">
          <Loader2 size={18} className="spin" aria-hidden="true" />
          <span>Loading behavior stats…</span>
        </div>
      )}
      {featuresError && <p className="dashboard-error" role="alert">{featuresError}</p>}
      {features && <BehaviorCard features={features} />}

      {/* Personalized recommendations */}
      <div className="dashboard-card" aria-label="Personalized recommendations">
        <h3 className="dashboard-card-title">
          <Film size={16} aria-hidden="true" />
          Recommended for You
        </h3>

        {recsLoading && (
          <div role="status" aria-live="polite" className="eval-loading">
            <Loader2 size={18} className="spin" aria-hidden="true" />
            <span>Loading recommendations…</span>
          </div>
        )}
        {recsError && <p className="dashboard-error" role="alert">{recsError}</p>}
        {!recsLoading && !recsError && recs.length === 0 && (
          <p role="status">No personalized recommendations yet. Rate some movies to get started.</p>
        )}
        {recs.length > 0 && (
          <div className="poster-grid" aria-label="Personalized movie recommendations">
            {recs.map((movie) => (
              <MiniRecommendationCard
                key={movie.id}
                movie={movie}
                onSelect={onSelectMovie}
              />
            ))}
          </div>
        )}
      </div>

      {/* Watch history from localStorage */}
      <div className="dashboard-card" aria-label="Watch history">
        <h3 className="dashboard-card-title">
          <Database size={16} aria-hidden="true" />
          Local Watch History
        </h3>
        <WatchHistorySection onSelectMovie={onSelectMovie} />
      </div>

      {/* Developer and platform analytics options inside My Space */}
      {onNavigate && (
        <div className="dashboard-card" aria-label="Platform and Analytics Tools">
          <h3 className="dashboard-card-title">
            <Activity size={16} aria-hidden="true" />
            Developer & Platform Tools
          </h3>
          <div className="profile-tools-grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "12px", marginTop: "12px" }}>
            <button
              type="button"
              className="tool-btn"
              onClick={() => onNavigate("dashboard")}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "8px",
                padding: "16px",
                background: "rgba(255, 255, 255, 0.02)",
                border: "1px solid rgba(255, 255, 255, 0.06)",
                borderRadius: "12px",
                color: "#fff",
                cursor: "pointer",
                transition: "all 0.2s"
              }}
            >
              <BarChart3 size={20} style={{ color: "var(--accent)" }} />
              <span style={{ fontSize: "0.8rem", fontWeight: 600 }}>Analytics</span>
            </button>
            <button
              type="button"
              className="tool-btn"
              onClick={() => onNavigate("knowledge-graph")}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "8px",
                padding: "16px",
                background: "rgba(255, 255, 255, 0.02)",
                border: "1px solid rgba(255, 255, 255, 0.06)",
                borderRadius: "12px",
                color: "#fff",
                cursor: "pointer",
                transition: "all 0.2s"
              }}
            >
              <Network size={20} style={{ color: "#a78bfa" }} />
              <span style={{ fontSize: "0.8rem", fontWeight: 600 }}>Knowledge Graph</span>
            </button>
            <button
              type="button"
              className="tool-btn"
              onClick={() => onNavigate("evaluation")}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "8px",
                padding: "16px",
                background: "rgba(255, 255, 255, 0.02)",
                border: "1px solid rgba(255, 255, 255, 0.06)",
                borderRadius: "12px",
                color: "#fff",
                cursor: "pointer",
                transition: "all 0.2s"
              }}
            >
              <Activity size={20} style={{ color: "#f43f5e" }} />
              <span style={{ fontSize: "0.8rem", fontWeight: 600 }}>Evaluation</span>
            </button>
            <button
              type="button"
              className="tool-btn"
              onClick={() => onNavigate("admin")}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "8px",
                padding: "16px",
                background: "rgba(255, 255, 255, 0.02)",
                border: "1px solid rgba(255, 255, 255, 0.06)",
                borderRadius: "12px",
                color: "#fff",
                cursor: "pointer",
                transition: "all 0.2s"
              }}
            >
              <Settings size={20} style={{ color: "#e2e8f0" }} />
              <span style={{ fontSize: "0.8rem", fontWeight: 600 }}>Admin Panel</span>
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
