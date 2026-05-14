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
  "https://movie-recs-api-5qvy.onrender.com",
  "https://pavanbadempet-movie-rec-api.hf.space",
];

const configuredBackends = [
  import.meta.env.VITE_API_URL,
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
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined) query.set(key, String(value));
  });
  const suffix = query.toString() ? `${path}?${query.toString()}` : path;
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
  return apiGet<ApiRoot>("/", {}, 8000);
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

export async function loadTitles(limit = 5000): Promise<BackendResult<MovieTitle[]>> {
  return apiGet<MovieTitle[]>("/movies/titles", { limit }, 30000);
}

export async function searchMovies(query: string): Promise<BackendResult<Movie[]>> {
  return apiGet<Movie[]>("/v1/search", { q: query, limit: 40 }, 18000);
}

export async function aiSearch(query: string): Promise<BackendResult<Movie[]>> {
  return apiGet<Movie[]>("/v1/search/ai", { q: query, limit: 40 }, 35000);
}

export async function getMovie(movieId: number): Promise<BackendResult<Movie>> {
  return apiGet<Movie>(`/movie/${movieId}`, {}, 15000);
}

export async function getRecommendations(movieId: number, n = 12): Promise<BackendResult<RecommendationResponse>> {
  return apiGet<RecommendationResponse>(`/v1/recommendations/id/${movieId}`, { n }, 45000);
}

export async function recordEvent(payload: EventPayload): Promise<BackendResult<EventResponse>> {
  return apiPost<EventResponse>("/v1/events", payload, 8000);
}
