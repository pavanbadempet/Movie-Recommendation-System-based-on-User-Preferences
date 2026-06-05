/**
 * Extended API tests covering branches not hit by api.test.ts:
 *  - apiGetFirstSuccess race (all fail)
 *  - apiPost 4xx stops retrying
 *  - platformStatus / platformReadiness / artifactHealth / semanticBenchmark wrappers
 *  - loadTitles / searchMovies / aiSearch / getMovie wrappers
 *  - getRecommendations fallback path
 *  - getUserRecommendations
 *  - recordEvent
 *  - registerUser
 *  - loginUser form-encoded path
 */

import "@testing-library/jest-dom";

// ─── Stub fetch globally ──────────────────────────────────────────────────────

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function okJson(data: unknown) {
  return Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve(data),
  } as Response);
}

function failResponse(status: number) {
  return Promise.resolve({
    ok: false,
    status,
    json: () => Promise.resolve({ detail: "error" }),
  } as Response);
}

import {
  platformStatus,
  platformReadiness,
  artifactHealth,
  semanticBenchmark,
  loadTitles,
  searchMovies,
  aiSearch,
  getMovie,
  getUserRecommendations,
  recordEvent,
  registerUser,
  loginUser,
  getRecommendations,
  backendLabel,
  currentBackend,
} from "../api";

beforeEach(() => {
  vi.clearAllMocks();
});

// ─── Wrapper functions ────────────────────────────────────────────────────────

describe("API wrapper functions", () => {
  it("platformStatus calls /v1/platform/status", async () => {
    mockFetch.mockResolvedValue(okJson({ status: "online" }));
    const result = await platformStatus();
    expect(result.data).toEqual({ status: "online" });
  });

  it("platformReadiness calls /v1/platform/readiness", async () => {
    mockFetch.mockResolvedValue(okJson({ status: "ready" }));
    const result = await platformReadiness(true, 10);
    expect(result.data).toEqual({ status: "ready" });
  });

  it("artifactHealth calls /v1/artifacts/health", async () => {
    mockFetch.mockResolvedValue(okJson({ status: "ready" }));
    const result = await artifactHealth();
    expect(result.data).toEqual({ status: "ready" });
  });

  it("semanticBenchmark calls /v1/evaluation/semantic-benchmark", async () => {
    mockFetch.mockResolvedValue(okJson({ status: "ok", metrics: {} }));
    const result = await semanticBenchmark(10);
    expect(result.data.status).toBe("ok");
  });

  it("loadTitles calls /movies/titles", async () => {
    mockFetch.mockResolvedValue(okJson([{ id: 1, title: "Avatar" }]));
    const result = await loadTitles(100);
    expect(result.data).toHaveLength(1);
  });

  it("searchMovies calls /v1/search", async () => {
    mockFetch.mockResolvedValue(okJson([{ id: 1, title: "Avatar" }]));
    const result = await searchMovies("avatar");
    expect(result.data[0].title).toBe("Avatar");
  });

  it("aiSearch calls /v1/search/ai", async () => {
    mockFetch.mockResolvedValue(okJson([{ id: 2, title: "Inception" }]));
    const result = await aiSearch("mind bending");
    expect(result.data[0].title).toBe("Inception");
  });

  it("getMovie calls /movie/:id", async () => {
    mockFetch.mockResolvedValue(okJson({ id: 1, title: "Avatar" }));
    const result = await getMovie(1);
    expect(result.data.id).toBe(1);
  });

  it("getUserRecommendations calls /v1/recommendations/user/:userId", async () => {
    mockFetch.mockResolvedValue(okJson([{ id: 1, title: "Avatar" }]));
    const result = await getUserRecommendations("alice", 5);
    expect(result.data).toHaveLength(1);
  });

  it("recordEvent calls /v1/events", async () => {
    mockFetch.mockResolvedValue(okJson({ status: "ok", event_id: "abc" }));
    const result = await recordEvent({ event_type: "view", movie_id: 1 });
    expect(result.data.status).toBe("ok");
  });

  it("registerUser calls /v1/auth/register", async () => {
    mockFetch.mockResolvedValue(okJson({ username: "bob" }));
    const result = await registerUser("bob", "pass");
    expect(result.data.username).toBe("bob");
  });
});

// ─── getRecommendations fallback ──────────────────────────────────────────────

describe("getRecommendations", () => {
  it("falls back to non-enriched endpoint when enriched fails", async () => {
    mockFetch
      .mockRejectedValueOnce(new Error("enriched unavailable"))
      .mockResolvedValue(okJson({
        request_id: "req-1",
        query_movie: { id: 1, title: "Avatar" },
        recommendations: [],
      }));

    const result = await getRecommendations(1, 5);
    expect(result.data.query_movie.title).toBe("Avatar");
  });
});

// ─── loginUser ────────────────────────────────────────────────────────────────

describe("loginUser extended", () => {
  it("returns token on success", async () => {
    mockFetch.mockResolvedValue(okJson({ access_token: "tok-xyz", token_type: "bearer" }));
    const result = await loginUser("alice", "secret");
    expect(result.data.access_token).toBe("tok-xyz");
  });

  it("throws when all backends return non-ok", async () => {
    mockFetch.mockResolvedValue(failResponse(401));
    await expect(loginUser("alice", "wrong")).rejects.toThrow();
  });
});

// ─── apiPost 4xx stops retrying ───────────────────────────────────────────────

describe("apiPost 4xx", () => {
  it("stops retrying on 4xx and throws", async () => {
    mockFetch.mockResolvedValue(failResponse(403));
    const { apiPost } = await import("../api");
    await expect(apiPost("/v1/admin/test", {}, 5000)).rejects.toThrow();
    // Should only have been called once (4xx = client error, no retry)
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});

// ─── currentBackend / backendLabel ───────────────────────────────────────────

describe("utility functions", () => {
  it("currentBackend returns a string", () => {
    expect(typeof currentBackend()).toBe("string");
  });

  it("backendLabel handles invalid URL", () => {
    expect(backendLabel("not-a-url")).toBe("not-a-url");
  });

  it("backendLabel strips www prefix", () => {
    expect(backendLabel("https://www.example.com")).toBe("example.com");
  });
});
