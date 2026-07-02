import type {
  ApiRoot,
  ArtifactHealth,
  BackendResult,
  EventPayload,
  EventResponse,
  Movie,
  MovieTitle,
  PlatformReadiness,
  PlatformStatus,
  RecommendationResponse,
  SemanticBenchmark,
} from "./types";
import { getClientRecommendations, getClientTextSearch } from "./webgpuEngine";

function normalizeBackend(url: string | undefined): string | undefined {
  const value = url?.trim().replace(/\/+$/, "");
  return value || undefined;
}

function localDevBackend(): string | undefined {
  if (typeof window === "undefined") return undefined;
  if (["localhost", "127.0.0.1", "::1"].includes(window.location.hostname) && window.location.port !== "8000") {
    return "http://localhost:8000";
  }
  return undefined;
}

function sameOriginBackend(): string | undefined {
  if (typeof window === "undefined") return undefined;
  if (["localhost", "127.0.0.1", "::1"].includes(window.location.hostname) && window.location.port !== "8000") {
    return undefined;
  }
  return normalizeBackend(window.location.origin);
}

const isLocalhost =
  typeof window !== "undefined" &&
  ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);

const configuredBackends = [
  import.meta.env.VITE_API_URL,
  import.meta.env.VITE_BACKUP_API_URL,
  localDevBackend(),
  sameOriginBackend(),
  "http://localhost:8000",
]
  .map(normalizeBackend)
  .filter((url): url is string => Boolean(url))
  .filter((url) => {
    // If not running locally, ignore localhost backend options to avoid connection timeouts
    if (!isLocalhost && (url.includes("localhost") || url.includes("127.0.0.1") || url.includes("::1"))) {
      return false;
    }
    return true;
  });

export const API_BASES = Array.from(new Set(configuredBackends));

let activeBackend = API_BASES[0] || (isLocalhost ? "http://localhost:8000" : (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000"));

function timeoutSignal(ms: number): { signal: AbortSignal; cancel: () => void } {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), ms);
  return {
    signal: controller.signal,
    cancel: () => window.clearTimeout(timeout),
  };
}

function candidateBackends(): string[] {
  return Array.from(new Set([activeBackend, ...API_BASES].filter(Boolean)));
}

function errorMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === "AbortError") return "timed out";
  if (error instanceof Error) return error.message;
  return "request failed";
}

function buildSuffix(path: string, params: Record<string, string | number | boolean | undefined>): string {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined) query.set(key, String(value));
  });
  return query.toString() ? `${path}?${query.toString()}` : path;
}

export function currentBackend(): string {
  return activeBackend;
}

export function backendLabel(url: string): string {
  try {
    const host = new URL(url).host;
    return host.replace(/^www\./, "");
  } catch {
    return url.replace("https://", "").replace("http://", "");
  }
}

export async function apiGet<T>(
  path: string,
  params: Record<string, string | number | boolean | undefined> = {},
  timeoutMs = 15000,
): Promise<BackendResult<T>> {
  const suffix = buildSuffix(path, params);
  const errors: string[] = [];

  for (const baseUrl of candidateBackends()) {
    const timeout = timeoutSignal(timeoutMs);
    try {
      const token = typeof window !== "undefined" ? window.localStorage.getItem("nova_jwt_token") : null;
      const headers: Record<string, string> = { Accept: "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const response = await fetch(`${baseUrl}${suffix}`, {
        headers,
        signal: timeout.signal,
      });
      if (!response.ok) {
        errors.push(`${baseUrl}: ${response.status}`);
        if (response.status < 500) break;
        continue;
      }
      activeBackend = baseUrl;
      return { data: (await response.json()) as T, baseUrl };
    } catch (error) {
      errors.push(`${backendLabel(baseUrl)} ${errorMessage(error)}`);
    } finally {
      timeout.cancel();
    }
  }

  throw new Error(errors.join(" | ") || "No backend available");
}

export async function apiGetFirstSuccess<T>(
  path: string,
  params: Record<string, string | number | boolean | undefined> = {},
  timeoutMs = 20000,
): Promise<BackendResult<T>> {
  const suffix = buildSuffix(path, params);
  const errors: string[] = [];
  const controllers: AbortController[] = [];
  const requests = candidateBackends().map(async (baseUrl) => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    controllers.push(controller);
    try {
      const token = typeof window !== "undefined" ? window.localStorage.getItem("nova_jwt_token") : null;
      const headers: Record<string, string> = { Accept: "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const response = await fetch(`${baseUrl}${suffix}`, {
        headers,
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(`${response.status}`);
      }
      return { data: (await response.json()) as T, baseUrl };
    } catch (error) {
      errors.push(`${backendLabel(baseUrl)} ${errorMessage(error)}`);
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  });

  try {
    const result = await Promise.any(requests);
    activeBackend = result.baseUrl;
    return result;
  } catch {
    throw new Error(errors.join(" | ") || "No backend available");
  } finally {
    controllers.forEach((controller) => controller.abort());
  }
}

export async function apiPost<T>(path: string, body: unknown, timeoutMs = 15000): Promise<BackendResult<T>> {
  const errors: string[] = [];

  for (const baseUrl of candidateBackends()) {
    const timeout = timeoutSignal(timeoutMs);
    try {
      const token = typeof window !== "undefined" ? window.localStorage.getItem("nova_jwt_token") : null;
      const headers: Record<string, string> = {
          Accept: "application/json",
          "Content-Type": "application/json",
      };
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const response = await fetch(`${baseUrl}${path}`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: timeout.signal,
      });
      if (!response.ok) {
        errors.push(`${baseUrl}: ${response.status}`);
        if (response.status < 500) break;
        continue;
      }
      activeBackend = baseUrl;
      return { data: (await response.json()) as T, baseUrl };
    } catch (error) {
      errors.push(`${backendLabel(baseUrl)} ${errorMessage(error)}`);
    } finally {
      timeout.cancel();
    }
  }

  throw new Error(errors.join(" | ") || "No backend available");
}

export async function pingApi(): Promise<BackendResult<ApiRoot>> {
  return apiGetFirstSuccess<ApiRoot>("/", {}, 8000);
}

export async function platformStatus(): Promise<BackendResult<PlatformStatus>> {
  return apiGet<PlatformStatus>("/v1/platform/status", {}, 15000);
}

export async function platformReadiness(strict = true, k = 10): Promise<BackendResult<PlatformReadiness>> {
  return apiGet<PlatformReadiness>("/v1/platform/readiness", { strict, k }, 90000);
}

export async function artifactHealth(): Promise<BackendResult<ArtifactHealth>> {
  return apiGet<ArtifactHealth>("/v1/artifacts/health", {}, 15000);
}

export async function semanticBenchmark(k = 10): Promise<BackendResult<SemanticBenchmark>> {
  return apiGet<SemanticBenchmark>("/v1/evaluation/semantic-benchmark", { k }, 45000);
}

export async function loadTitles(limit = 100000): Promise<BackendResult<MovieTitle[]>> {
  return apiGetFirstSuccess<MovieTitle[]>("/movies/titles", { limit }, 30000);
}

export async function searchMovies(query: string): Promise<BackendResult<Movie[]>> {
  return apiGetFirstSuccess<Movie[]>("/v1/search", { q: query, limit: 40 }, 18000);
}

export async function getShowcaseMovies(limit = 8): Promise<BackendResult<Movie[]>> {
  return apiGetFirstSuccess<Movie[]>("/movies/showcase", { limit }, 3000);
}

export async function aiSearch(query: string): Promise<BackendResult<Movie[]>> {
  try {
    const clientResults = await getClientTextSearch(query, 40);
    if (clientResults && clientResults.length > 0) {
      return {
        data: clientResults,
        baseUrl: "client"
      };
    }
  } catch (e) {
    console.warn("[APEX] Client semantic search failed, falling back to server:", e);
  }
  return apiGetFirstSuccess<Movie[]>("/v1/search/ai", { q: query, limit: 40 }, 25000);
}

export async function getMovie(movieId: number): Promise<BackendResult<Movie>> {
  return apiGetFirstSuccess<Movie>(`/movie/${movieId}`, {}, 15000);
}

export async function getMovieEnriched(movieId: number): Promise<BackendResult<Movie>> {
  // 1. Try the dedicated enriched endpoint (available after backend redeploy)
  try {
    const result = await apiGet<Movie>(`/movie/${movieId}/enriched`, {}, 10000);
    if (result.data.trailer_key) return result;
  } catch { /* endpoint not deployed yet, continue */ }

  // 2. Fetch base movie + TMDB trailer in parallel
  const [baseResult, trailerKey] = await Promise.all([
    apiGetFirstSuccess<Movie>(`/movie/${movieId}`, {}, 15000),
    fetchTmdbTrailer(movieId),
  ]);
  if (trailerKey) {
    baseResult.data = { ...baseResult.data, trailer_key: trailerKey };
  }
  return baseResult;
}

async function fetchTmdbTrailer(movieId: number): Promise<string | null> {
  // Try fetching trailer from TMDB videos endpoint via the backend proxy
  try {
    const result = await apiGet<{ trailer_key?: string | null }>(`/movie/${movieId}/trailer`, {}, 8000);
    if (result.data.trailer_key) return result.data.trailer_key;
  } catch { /* no trailer proxy endpoint, try enriched recs */ }

  // Fall back: use enriched recommendations to extract trailer from first rec
  try {
    const recResult = await apiGetFirstSuccess<RecommendationResponse>(
      `/v1/recommendations/id/${movieId}/enriched`, { n: 1 }, 15000,
    );
    // The recommendations (not query_movie) have trailer_key from TMDB enrichment
    // We can't get OUR movie's trailer from recs, but query_movie might have it
    const qm = recResult.data.query_movie;
    if ((qm as Record<string, unknown>).trailer_key) {
      return (qm as Record<string, unknown>).trailer_key as string;
    }
  } catch { /* no enriched recs available */ }

  return null;
}

export async function getRecommendations(movieId: number, n = 12, timeoutMs = 60000): Promise<BackendResult<RecommendationResponse>> {
  try {
    const clientRecs = await getClientRecommendations(movieId, n);
    if (clientRecs) {
      const titlesResult = await loadTitles();
      let queryMovie: Movie = { id: movieId, title: "Query Movie", genres: "", overview: "", release_date: "", popularity: 1.0 };
      if (titlesResult && titlesResult.data) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const found = titlesResult.data.find(m => m.id === movieId) as any;
        if (found) {
          queryMovie = {
            id: found.id,
            title: found.title,
            genres: Array.isArray(found.genres) ? found.genres.join("|") : found.genres || "",
            overview: "Query movie matching client cache.",
            release_date: found.release_date || "",
            popularity: found.popularity || 1.0
          };
        }
      }
      return {
        data: {
          request_id: "client_gpu_vector_search",
          query_movie: queryMovie,
          recommendations: clientRecs
        },
        baseUrl: "client"
      };
    }
  } catch (e) {
    console.warn("[APEX] Client recommendations failed, falling back to server:", e);
  }

  try {
    return await apiGetFirstSuccess<RecommendationResponse>(`/v1/recommendations/id/${movieId}/enriched`, { n, explain: true }, timeoutMs);
  } catch {
    return apiGetFirstSuccess<RecommendationResponse>(`/v1/recommendations/id/${movieId}`, { n, explain: true }, timeoutMs);
  }
}

export async function getVisualRecommendations(movieId: number, n = 12): Promise<BackendResult<RecommendationResponse>> {
  return apiGetFirstSuccess<RecommendationResponse>(`/v1/recommendations/visually-similar/${movieId}`, { n, explain: true }, 60000);
}

export async function getKGRecommendations(movieId: number, n = 12): Promise<BackendResult<RecommendationResponse>> {
  return apiGetFirstSuccess<RecommendationResponse>(`/v1/recommendations/knowledge-graph/${movieId}`, { n }, 60000);
}

export async function getUserRecommendations(userId: string, n = 8): Promise<BackendResult<Movie[]>> {
  return apiGet<Movie[]>(`/v1/recommendations/user/${encodeURIComponent(userId)}`, { n }, 30000);
}

export interface CacheStatus {
  youtube_id: string;
  cached: boolean;
}

export async function checkVideoCacheStatus(youtubeId: string): Promise<BackendResult<CacheStatus>> {
  return apiGet<CacheStatus>(`/v1/videos/cache-status/${youtubeId}`, {}, 8000);
}

export async function recordEvent(payload: EventPayload): Promise<BackendResult<EventResponse>> {
  return apiPost<EventResponse>("/v1/events", payload, 8000);
}

export async function registerUser(username: string, password: string): Promise<BackendResult<{ detail?: string; username?: string }>> {
  return apiPost<{ detail?: string; username?: string }>("/v1/auth/register", { username, password }, 8000);
}

export async function loginUser(username: string, password: string): Promise<BackendResult<{access_token: string, token_type: string}>> {
  const errors: string[] = [];
  const body = new URLSearchParams();
  body.append("username", username);
  body.append("password", password);

  for (const baseUrl of candidateBackends()) {
    const timeout = timeoutSignal(8000);
    try {
      const response = await fetch(`${baseUrl}/v1/auth/token`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: body.toString(),
        signal: timeout.signal,
      });
      if (!response.ok) {
        errors.push(`${baseUrl}: ${response.status}`);
        continue;
      }
      return { data: await response.json(), baseUrl };
    } catch (error) {
      errors.push(`${backendLabel(baseUrl)} ${errorMessage(error)}`);
    } finally {
      timeout.cancel();
    }
  }

  throw new Error(errors.join(" | ") || "No backend available");
}

export const runAdminTestSuite = async (suite: string) => { const response = await fetch(`${activeBackend}/v1/admin/tests/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ suite }) }); return response.json(); };
