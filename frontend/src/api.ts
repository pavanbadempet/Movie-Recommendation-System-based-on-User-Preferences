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

const DEFAULT_BACKENDS = [
  "https://pavanbadempet-movie-rec-api.hf.space",
  "https://movie-recs-api-5qvy.onrender.com",
];

function sameOriginBackend(): string | undefined {
  if (typeof window === "undefined") return undefined;
  const enabledByHost = window.location.hostname.endsWith(".hf.space");
  const enabledByEnv = import.meta.env.VITE_USE_SAME_ORIGIN_API === "true";
  if (!enabledByHost && !enabledByEnv) return undefined;
  return window.location.origin.replace(/\/+$/, "");
}

const configuredBackends = [
  import.meta.env.VITE_API_URL,
  sameOriginBackend(),
  import.meta.env.VITE_BACKUP_API_URL,
  ...DEFAULT_BACKENDS,
]
  .filter(Boolean)
  .map((url) => String(url).replace(/\/+$/, ""));

export const API_BASES = Array.from(new Set(configuredBackends));

let activeBackend = API_BASES[0];

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
      const response = await fetch(`${baseUrl}${suffix}`, {
        headers: { Accept: "application/json" },
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
      const response = await fetch(`${baseUrl}${suffix}`, {
        headers: { Accept: "application/json" },
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
      const response = await fetch(`${baseUrl}${path}`, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
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

export async function aiSearch(query: string): Promise<BackendResult<Movie[]>> {
  return apiGetFirstSuccess<Movie[]>("/v1/search/ai", { q: query, limit: 40 }, 25000);
}

export async function getMovie(movieId: number): Promise<BackendResult<Movie>> {
  return apiGetFirstSuccess<Movie>(`/movie/${movieId}`, {}, 15000);
}

export async function getRecommendations(movieId: number, n = 12): Promise<BackendResult<RecommendationResponse>> {
  try {
    return await apiGetFirstSuccess<RecommendationResponse>(`/v1/recommendations/id/${movieId}/enriched`, { n }, 60000);
  } catch {
    return apiGetFirstSuccess<RecommendationResponse>(`/v1/recommendations/id/${movieId}`, { n }, 60000);
  }
}

export async function recordEvent(payload: EventPayload): Promise<BackendResult<EventResponse>> {
  return apiPost<EventResponse>("/v1/events", payload, 8000);
}
