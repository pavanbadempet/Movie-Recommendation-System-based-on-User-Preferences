/**
 * Accessibility audit tests using @axe-core/react + jest-axe.
 *
 * Each page component is rendered in isolation with mocked hooks/API calls
 * and checked for WCAG 2.0 A/AA violations.
 *
 * Validates: Requirements 12.1, 12.2, 12.3, 12.4
 */

/// <reference types="vitest/globals" />
import React from "react";
import { render } from "@testing-library/react";
import { axe, toHaveNoViolations } from "jest-axe";
import "@testing-library/jest-dom";

expect.extend(toHaveNoViolations);

// ─── Mock hooks ───────────────────────────────────────────────────────────────

vi.mock("../hooks/useHealth", () => ({
  useHealth: () => ({
    data: {
      status: "online",
      serving_tier: "tier2",
      hardware_profile: { gpu_available: false, ram_gb: 16.0, cpu_cores: 4 },
      tier_selection_reason: "hardware_auto_detection",
    },
    loading: false,
    error: null,
  }),
}));

vi.mock("../hooks/useSlo", () => ({
  useSlo: () => ({
    data: { p95_latency_ms: 120, p99_latency_ms: 200, error_rate: 0.002, request_rate: 5.4, uptime_seconds: 3600 },
    loading: false,
    error: null,
    degraded: false,
  }),
}));

vi.mock("../hooks/useKnowledgeGraph", () => ({
  useKnowledgeGraph: () => ({
    graphData: null,
    loading: false,
    error: null,
  }),
}));

vi.mock("../api", () => ({
  apiGet: () => Promise.resolve({ data: {}, baseUrl: "http://localhost:8000" }),
  apiPost: () => Promise.resolve({ data: {}, baseUrl: "http://localhost:8000" }),
  semanticBenchmark: () => Promise.resolve({ data: { status: "ok", metrics: {} }, baseUrl: "http://localhost:8000" }),
  getUserRecommendations: () => Promise.resolve({ data: [], baseUrl: "http://localhost:8000" }),
  getKGRecommendations: () => Promise.resolve({ data: { query_movie: { id: 1, title: "Test" }, recommendations: [] }, baseUrl: "http://localhost:8000" }),
  backendLabel: (url: string) => url,
  currentBackend: () => "http://localhost:8000",
  API_BASES: ["http://localhost:8000"],
}));

// ─── Import pages after mocks ─────────────────────────────────────────────────

import { Dashboard } from "../pages/Dashboard";
import { KnowledgeGraphPage } from "../pages/KnowledgeGraph";
import { EvaluationPage } from "../pages/Evaluation";
import { UserProfilePage } from "../pages/UserProfile";
import { AdminPanel } from "../pages/AdminPanel";

const AXE_OPTIONS = {
  runOnly: { type: "tag" as const, values: ["wcag2a", "wcag2aa"] },
};

// ─── Dashboard ────────────────────────────────────────────────────────────────

describe("Dashboard accessibility", () => {
  it("has no WCAG 2.0 A/AA violations", async () => {
    const { container } = render(<Dashboard />);
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });
});

// ─── Knowledge Graph ──────────────────────────────────────────────────────────

describe("KnowledgeGraphPage accessibility", () => {
  it("has no WCAG 2.0 A/AA violations in empty state", async () => {
    const { container } = render(<KnowledgeGraphPage titles={[]} />);
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });

  it("has no WCAG 2.0 A/AA violations with title list", async () => {
    const titles = [
      { id: 1, title: "Avatar" },
      { id: 2, title: "Inception" },
    ];
    const { container } = render(<KnowledgeGraphPage titles={titles} />);
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });
});

// ─── Evaluation ───────────────────────────────────────────────────────────────

describe("EvaluationPage accessibility", () => {
  it("has no WCAG 2.0 A/AA violations", async () => {
    const { container } = render(<EvaluationPage />);
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });
});

// ─── User Profile ─────────────────────────────────────────────────────────────

describe("UserProfilePage accessibility", () => {
  it("has no WCAG 2.0 A/AA violations when not authenticated", async () => {
    const { container } = render(
      <UserProfilePage
        token={null}
        username={null}
        onRequestLogin={vi.fn()}
        onSelectMovie={vi.fn()}
      />,
    );
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });

  it("has no WCAG 2.0 A/AA violations when authenticated", async () => {
    const { container } = render(
      <UserProfilePage
        token="test-token"
        username="alice"
        onRequestLogin={vi.fn()}
        onSelectMovie={vi.fn()}
      />,
    );
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });
});

// ─── Admin Panel ──────────────────────────────────────────────────────────────

describe("AdminPanel accessibility", () => {
  it("has no WCAG 2.0 A/AA violations when not authenticated", async () => {
    const { container } = render(<AdminPanel token={null} />);
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });

  it("has no WCAG 2.0 A/AA violations when authenticated", async () => {
    const { container } = render(<AdminPanel token="admin-token" />);
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });
});

// ─── Landing page ─────────────────────────────────────────────────────────────

import { LandingPage } from "../pages/Landing";

describe("LandingPage accessibility", () => {
  it("has no WCAG 2.0 A/AA violations", async () => {
    const { container } = render(<LandingPage onNavigate={vi.fn()} />);
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });
});

// ─── Pricing page ─────────────────────────────────────────────────────────────

import { PricingPage } from "../pages/Pricing";

describe("PricingPage accessibility", () => {
  it("has no WCAG 2.0 A/AA violations", async () => {
    const { container } = render(<PricingPage onNavigate={vi.fn()} />);
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });
});

// ─── Signup page ──────────────────────────────────────────────────────────────

import { SignupPage } from "../pages/Signup";

describe("SignupPage accessibility", () => {
  it("has no WCAG 2.0 A/AA violations on the registration form", async () => {
    const { container } = render(
      <SignupPage onNavigate={vi.fn()} onLoginSuccess={vi.fn()} />,
    );
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });
});

// ─── Getting Started page ─────────────────────────────────────────────────────

import { GettingStartedPage } from "../pages/GettingStarted";

describe("GettingStartedPage accessibility", () => {
  it("has no WCAG 2.0 A/AA violations on step 1", async () => {
    const { container } = render(<GettingStartedPage onNavigate={vi.fn()} />);
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });
});

// ─── Status page ──────────────────────────────────────────────────────────────

import { StatusPage } from "../pages/Status";

describe("StatusPage accessibility", () => {
  it("has no WCAG 2.0 A/AA violations in loading state", async () => {
    const { container } = render(<StatusPage />);
    const results = await axe(container, AXE_OPTIONS);
    expect(results).toHaveNoViolations();
  });
});
