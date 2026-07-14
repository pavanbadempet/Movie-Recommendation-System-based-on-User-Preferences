import React from "react";
import { render, screen, waitFor, act } from "@testing-library/react";
import "@testing-library/jest-dom";
import { StatusPage } from "../pages/Status";
import { apiGet } from "../api";

vi.mock("../api", () => ({
  apiGet: vi.fn(),
}));

describe("StatusPage Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders outage state if health endpoint fails", async () => {
    vi.mocked(apiGet).mockRejectedValue(new Error("Network Error"));
    await act(async () => {
      render(<StatusPage />);
    });
    await waitFor(() => {
      expect(screen.getByText(/Service disruption/i)).toBeInTheDocument();
    });
  });

  it("renders degraded state if error rate is high", async () => {
    vi.mocked(apiGet).mockImplementation(async (url: string) => {
      if (url === "/health") return { data: { status: "ok", serving_tier: "tier1" } };
      if (url === "/v1/platform/slo") return { data: { error_rate: 0.05, p95_latency_ms: 100 } };
      return { data: {} };
    });
    
    await act(async () => {
      render(<StatusPage />);
    });
    await waitFor(() => {
      expect(screen.getByText(/Partial degradation/i)).toBeInTheDocument();
    });
  });

  it("renders degraded state if latency is high", async () => {
    vi.mocked(apiGet).mockImplementation(async (url: string) => {
      if (url === "/health") return { data: { status: "ok", serving_tier: "tier2" } };
      if (url === "/v1/platform/slo") return { data: { error_rate: 0.01, p95_latency_ms: 30000 } };
      return { data: {} };
    });

    await act(async () => {
      render(<StatusPage />);
    });
    await waitFor(() => {
      expect(screen.getByText(/Partial degradation/i)).toBeInTheDocument();
    });
  });

  it("renders operational state if everything is fine", async () => {
    vi.mocked(apiGet).mockImplementation(async (url: string) => {
      if (url === "/health") return { data: { status: "ok", serving_tier: "tier3" } };
      if (url === "/v1/platform/slo") return { data: { error_rate: 0.01, p95_latency_ms: 1000 } };
      return { data: {} };
    });

    await act(async () => {
      render(<StatusPage />);
    });
    await waitFor(() => {
      expect(screen.getByText(/All systems operational/i)).toBeInTheDocument();
    });
  });
});
