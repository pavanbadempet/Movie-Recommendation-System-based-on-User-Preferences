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

// Mock webgpuEngine
vi.mock("../webgpuEngine", () => ({
  getClientTextSearch: vi.fn(),
  getClientRecommendations: vi.fn(),
}));
import * as webgpuEngine from "../webgpuEngine";

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
  getShowcaseMovies,
  getMovieEnriched,
  getVisualRecommendations,
  getKGRecommendations,
  checkVideoCacheStatus,
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

  it("aiSearch calls /v1/search/ai when client engine returns empty", async () => {
    vi.mocked(webgpuEngine.getClientTextSearch).mockResolvedValue(null);
    mockFetch.mockResolvedValue(okJson([{ id: 2, title: "Inception" }]));
    const result = await aiSearch("mind bending");
    expect(result.data[0].title).toBe("Inception");
  });

  it("aiSearch falls back to server if client engine throws", async () => {
    vi.mocked(webgpuEngine.getClientTextSearch).mockRejectedValue(new Error("client failed"));
    mockFetch.mockResolvedValue(okJson([{ id: 3, title: "Matrix" }]));
    const result = await aiSearch("mind bending");
    expect(result.data[0].title).toBe("Matrix");
  });

  it("uses client engine when it returns results", async () => {
    vi.mocked(webgpuEngine.getClientTextSearch).mockResolvedValue([{ id: 10, title: "Client Result" } as any]);
    const result = await aiSearch("client query");
    expect(result.data[0].title).toBe("Client Result");
    expect(result.baseUrl).toBe("client");
  });

  it("getMovie calls /movie/:id", async () => {
    mockFetch.mockResolvedValue(okJson({ id: 1, title: "Avatar" }));
    const result = await getMovie(1);
    expect(result.data.id).toBe(1);
  });

  it("getShowcaseMovies calls /movies/showcase", async () => {
    mockFetch.mockResolvedValue(okJson([{ id: 1, title: "Avatar" }]));
    const result = await getShowcaseMovies(8);
    expect(result.data).toHaveLength(1);
  });

  it("getMovieEnriched returns immediately if enriched endpoint has trailer", async () => {
    mockFetch.mockResolvedValueOnce(okJson({ id: 1, trailer_key: "abc" }));
    const result = await getMovieEnriched(1);
    expect(result.data.trailer_key).toBe("abc");
  });

  it("getMovieEnriched falls back and fetches trailer via proxy if enriched fails", async () => {
    mockFetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = input.toString();
      if (url.includes("/enriched")) return failResponse(404);
      if (url.includes("/trailer")) return okJson({ trailer_key: "proxy-key" });
      if (url.includes("/movie/1")) return okJson({ id: 1, title: "Base" });
      return failResponse(404);
    });

    const result = await getMovieEnriched(1);
    expect(result.data.trailer_key).toBe("proxy-key");
    expect(result.data.title).toBe("Base");
  });

  it("getMovieEnriched falls back to recs endpoint if proxy fails", async () => {
    mockFetch.mockImplementation(async (input: RequestInfo | URL) => {
      const url = input.toString();
      if (url.endsWith("/enriched")) return failResponse(404); // the direct movie enriched
      if (url.includes("/trailer")) return failResponse(404);
      if (url.includes("/recommendations/id/1/enriched")) return okJson({ query_movie: { trailer_key: "rec-key" } });
      if (url.includes("/movie/1")) return okJson({ id: 1, title: "Base" });
      return failResponse(404);
    });

    const result = await getMovieEnriched(1);
    expect(result.data.trailer_key).toBe("rec-key");
    expect(result.data.title).toBe("Base");
  });

  it("getVisualRecommendations calls /v1/recommendations/visual/:id", async () => {
    mockFetch.mockResolvedValue(okJson({ request_id: "req", recommendations: [] }));
    const result = await getVisualRecommendations(1, 5);
    expect(result.data.request_id).toBe("req");
  });

  it("getKGRecommendations calls /v1/recommendations/kg/:id", async () => {
    mockFetch.mockResolvedValue(okJson({ request_id: "req", recommendations: [] }));
    const result = await getKGRecommendations(1, 5);
    expect(result.data.request_id).toBe("req");
  });

  it("checkVideoCacheStatus calls /v1/video/status", async () => {
    mockFetch.mockResolvedValue(okJson({ youtube_id: "abc", cached: true }));
    const result = await checkVideoCacheStatus("abc");
    expect(result.data.cached).toBe(true);
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
    vi.mocked(webgpuEngine.getClientRecommendations).mockResolvedValue(null);
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

  it("uses client engine when it returns recommendations", async () => {
    vi.mocked(webgpuEngine.getClientRecommendations).mockResolvedValue([{ id: 11, title: "Client Rec" } as any]);
    // Also need to mock loadTitles, which calls /movies/titles
    mockFetch.mockResolvedValueOnce(okJson([{ id: 1, title: "Avatar" }]));
    
    const result = await getRecommendations(1, 5);
    expect(result.data.recommendations[0].title).toBe("Client Rec");
    expect(result.data.query_movie.title).toBe("Avatar"); // from loadTitles mock
    expect(result.baseUrl).toBe("client");
  });

  it("uses client engine even if loadTitles fails to find the query movie", async () => {
    vi.mocked(webgpuEngine.getClientRecommendations).mockResolvedValue([{ id: 11, title: "Client Rec" } as any]);
    // Missing from titles
    mockFetch.mockResolvedValueOnce(okJson([{ id: 99, title: "Other Movie" }]));
    
    const result = await getRecommendations(1, 5);
    expect(result.data.query_movie.title).toBe("Query Movie");
    expect(result.baseUrl).toBe("client");
  });

  it("falls back to server if client recommendations throws", async () => {
    vi.mocked(webgpuEngine.getClientRecommendations).mockRejectedValue(new Error("client failed"));
    mockFetch.mockResolvedValue(okJson({
      request_id: "req-1",
      query_movie: { id: 1, title: "Avatar" },
      recommendations: [],
    }));
    const result = await getRecommendations(1, 5);
    expect(result.data.query_movie.title).toBe("Avatar");
    expect(result.baseUrl).not.toBe("client");
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
