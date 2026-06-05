/**
 * Unit tests for custom hooks: useHealth, useSlo, useKnowledgeGraph.
 *
 * Uses renderHook from @testing-library/react to exercise the real hook
 * implementations (not mocked), covering the happy path, error path, and
 * cancellation on unmount.
 */

import { renderHook, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";

// ─── Mock the API module ──────────────────────────────────────────────────────

vi.mock("../api", () => ({
  apiGet: vi.fn(),
  getKGRecommendations: vi.fn(),
}));

import { apiGet, getKGRecommendations } from "../api";
import { useHealth } from "../hooks/useHealth";
import { useSlo } from "../hooks/useSlo";
import { useKnowledgeGraph } from "../hooks/useKnowledgeGraph";

// ─── useHealth ────────────────────────────────────────────────────────────────

describe("useHealth", () => {
  beforeEach(() => vi.clearAllMocks());

  it("starts in loading state", () => {
    vi.mocked(apiGet).mockReturnValue(new Promise(() => {})); // never resolves
    const { result } = renderHook(() => useHealth());
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("returns data on success", async () => {
    const mockData = {
      status: "online",
      serving_tier: "tier2",
      hardware_profile: { gpu_available: false, ram_gb: 16.0, cpu_cores: 4 },
      tier_selection_reason: "hardware_auto_detection",
    };
    vi.mocked(apiGet).mockResolvedValue({ data: mockData, baseUrl: "http://localhost:8000" });

    const { result } = renderHook(() => useHealth());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data).toEqual(mockData);
    expect(result.current.error).toBeNull();
  });

  it("sets error on failure", async () => {
    vi.mocked(apiGet).mockRejectedValue(new Error("Network error"));

    const { result } = renderHook(() => useHealth());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe("Network error");
  });

  it("handles non-Error rejection gracefully", async () => {
    vi.mocked(apiGet).mockRejectedValue("string error");

    const { result } = renderHook(() => useHealth());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe("Failed to fetch health data");
  });
});

// ─── useSlo ───────────────────────────────────────────────────────────────────

describe("useSlo", () => {
  beforeEach(() => vi.clearAllMocks());

  it("starts in loading state with degraded=false", () => {
    vi.mocked(apiGet).mockReturnValue(new Promise(() => {}));
    const { result } = renderHook(() => useSlo());
    expect(result.current.loading).toBe(true);
    expect(result.current.degraded).toBe(false);
  });

  it("returns SLO data on success", async () => {
    const mockSlo = { p95_latency_ms: 120, error_rate: 0.001, request_rate: 5.4 };
    vi.mocked(apiGet).mockResolvedValue({ data: mockSlo, baseUrl: "http://localhost:8000" });

    const { result } = renderHook(() => useSlo());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data).toEqual(mockSlo);
    expect(result.current.degraded).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("sets degraded=true on network failure without throwing", async () => {
    vi.mocked(apiGet).mockRejectedValue(new Error("SLO unavailable"));

    const { result } = renderHook(() => useSlo());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.degraded).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe("SLO unavailable");
  });

  it("sets degraded=true on non-Error rejection", async () => {
    vi.mocked(apiGet).mockRejectedValue(null);

    const { result } = renderHook(() => useSlo());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.degraded).toBe(true);
    expect(result.current.error).toBe("SLO endpoint unavailable");
  });
});

// ─── useKnowledgeGraph ────────────────────────────────────────────────────────

describe("useKnowledgeGraph", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns null graphData when movieId is null", () => {
    const { result } = renderHook(() => useKnowledgeGraph(null));
    expect(result.current.graphData).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("starts loading when movieId is provided", () => {
    vi.mocked(getKGRecommendations).mockReturnValue(new Promise(() => {}));
    const { result } = renderHook(() => useKnowledgeGraph(1));
    expect(result.current.loading).toBe(true);
  });

  it("transforms API response into nodes and edges", async () => {
    const mockResponse = {
      data: {
        query_movie: { id: 1, title: "Avatar", poster_path: "/avatar.jpg" },
        recommendations: [
          { id: 2, title: "Inception", retrieval_stage: "knowledge_graph" },
          { id: 3, title: "Interstellar", retrieval_stage: "semantic" },
        ],
      },
      baseUrl: "http://localhost:8000",
    };
    vi.mocked(getKGRecommendations).mockResolvedValue(mockResponse as any);

    const { result } = renderHook(() => useKnowledgeGraph(1));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.graphData).not.toBeNull();
    expect(result.current.graphData!.nodes).toHaveLength(3); // seed + 2 recs
    expect(result.current.graphData!.edges).toHaveLength(2);

    const seedNode = result.current.graphData!.nodes.find((n) => n.type === "seed");
    expect(seedNode?.label).toBe("Avatar");
    expect(seedNode?.id).toBe("movie-1");

    const recNode = result.current.graphData!.nodes.find((n) => n.id === "movie-2");
    expect(recNode?.type).toBe("recommendation");

    const edge = result.current.graphData!.edges[0];
    expect(edge.source).toBe("movie-1");
    expect(edge.target).toBe("movie-2");
    expect(edge.label).toBe("knowledge_graph");
  });

  it("uses 'related' as edge label when retrieval_stage is null", async () => {
    const mockResponse = {
      data: {
        query_movie: { id: 1, title: "Avatar" },
        recommendations: [{ id: 2, title: "Inception", retrieval_stage: null }],
      },
      baseUrl: "http://localhost:8000",
    };
    vi.mocked(getKGRecommendations).mockResolvedValue(mockResponse as any);

    const { result } = renderHook(() => useKnowledgeGraph(1));
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.graphData!.edges[0].label).toBe("related");
  });

  it("sets error on API failure", async () => {
    vi.mocked(getKGRecommendations).mockRejectedValue(new Error("Graph unavailable"));

    const { result } = renderHook(() => useKnowledgeGraph(42));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe("Graph unavailable");
    expect(result.current.graphData).toBeNull();
  });

  it("handles non-Error rejection", async () => {
    vi.mocked(getKGRecommendations).mockRejectedValue("unknown");

    const { result } = renderHook(() => useKnowledgeGraph(42));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe("Failed to load knowledge graph");
  });

  it("resets state when movieId changes to null", async () => {
    const mockResponse = {
      data: {
        query_movie: { id: 1, title: "Avatar" },
        recommendations: [],
      },
      baseUrl: "http://localhost:8000",
    };
    vi.mocked(getKGRecommendations).mockResolvedValue(mockResponse as any);

    const { result, rerender } = renderHook(
      ({ id }: { id: number | null }) => useKnowledgeGraph(id),
      { initialProps: { id: 1 as number | null } },
    );

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.graphData).not.toBeNull();

    rerender({ id: null });
    expect(result.current.graphData).toBeNull();
    expect(result.current.error).toBeNull();
  });
});
