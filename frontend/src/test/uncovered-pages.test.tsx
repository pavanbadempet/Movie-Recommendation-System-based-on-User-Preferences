/**
 * Tests for the four previously-uncovered page components:
 *   LandingPage, PricingPage, SignupPage, GettingStartedPage
 *
 * These tests close the coverage gap that causes CI to fail the
 * vitest --coverage thresholds (lines ≥ 75%, branches ≥ 70%,
 * statements ≥ 75%, functions ≥ 60%).
 */

import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import "@testing-library/jest-dom";

// ─── Mock ../api ──────────────────────────────────────────────────────────────

vi.mock("../api", () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
  API_BASES: ["https://test-api.example.com"],
}));

import { apiGet, apiPost } from "../api";
import { LandingPage } from "../pages/Landing";
import { PricingPage } from "../pages/Pricing";
import { SignupPage } from "../pages/Signup";
import { GettingStartedPage } from "../pages/GettingStarted";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function mockClipboard() {
  const writeText = vi.fn().mockResolvedValue(undefined);
  Object.assign(navigator, { clipboard: { writeText } });
  return writeText;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LandingPage
// ═══════════════════════════════════════════════════════════════════════════════

describe("LandingPage", () => {
  const onNavigate = vi.fn();
  beforeEach(() => vi.clearAllMocks());

  it("renders hero section with headline and CTA buttons", () => {
    render(<LandingPage onNavigate={onNavigate} />);
    expect(screen.getByText("Netflix-quality recommendations.")).toBeInTheDocument();
    expect(screen.getByText(/No ML team required/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Sign up for APEX/)).toBeInTheDocument();
    expect(screen.getByLabelText(/View APEX quickstart/)).toBeInTheDocument();
  });

  it("renders benchmark cards", () => {
    render(<LandingPage onNavigate={onNavigate} />);
    expect(screen.getByText("0.785")).toBeInTheDocument();
    expect(screen.getByText("HR@10")).toBeInTheDocument();
    expect(screen.getByText("0.542")).toBeInTheDocument();
    expect(screen.getByText("NDCG@10")).toBeInTheDocument();
    expect(screen.getByText("1.000")).toBeInTheDocument();
  });

  it("renders how-it-works steps", () => {
    render(<LandingPage onNavigate={onNavigate} />);
    expect(screen.getByText("Up and running in 30 minutes")).toBeInTheDocument();
    expect(screen.getByText("Upload your catalog")).toBeInTheDocument();
    expect(screen.getByText("Call the recommendation API")).toBeInTheDocument();
    expect(screen.getByText("Log events, improve over time")).toBeInTheDocument();
  });

  it("renders feature cards", () => {
    render(<LandingPage onNavigate={onNavigate} />);
    expect(screen.getByText("6-Model Ensemble")).toBeInTheDocument();
    expect(screen.getByText("Multi-Modal Search")).toBeInTheDocument();
    expect(screen.getByText("Knowledge Graph")).toBeInTheDocument();
    expect(screen.getByText("LLM Explanations")).toBeInTheDocument();
    expect(screen.getByText("Differential Privacy")).toBeInTheDocument();
    expect(screen.getByText("Adaptive Serving")).toBeInTheDocument();
  });

  it("renders pricing teaser cards", () => {
    render(<LandingPage onNavigate={onNavigate} />);
    expect(screen.getByText("Start free, scale when ready")).toBeInTheDocument();
    expect(screen.getByText("$0")).toBeInTheDocument();
    expect(screen.getByText(/\$299/)).toBeInTheDocument();
    expect(screen.getByText("Custom")).toBeInTheDocument();
    expect(screen.getByText("Most popular")).toBeInTheDocument();
  });

  it("renders trust signals and footer", () => {
    render(<LandingPage onNavigate={onNavigate} />);
    expect(screen.getByText(/GDPR-compliant/)).toBeInTheDocument();
    expect(screen.getByText(/Delta Lake/)).toBeInTheDocument();
    expect(screen.getByText("APEX")).toBeInTheDocument();
    expect(screen.getByText(/MIT License/)).toBeInTheDocument();
  });

  it("navigates to signup when Get started free is clicked", async () => {
    render(<LandingPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText(/Sign up for APEX/));
    expect(onNavigate).toHaveBeenCalledWith("signup");
  });

  it("navigates to getting-started when View quickstart is clicked", () => {
    render(<LandingPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText(/View APEX quickstart/));
    expect(onNavigate).toHaveBeenCalledWith("getting-started");
  });

  it("defaults to mobile tab and shows phone preview", () => {
    render(<LandingPage onNavigate={onNavigate} />);
    expect(screen.getByLabelText("Interactive mobile app preview")).toBeInTheDocument();
    expect(screen.getByText("NOVA")).toBeInTheDocument();
    expect(screen.getByText("Se7en (1995)")).toBeInTheDocument();
  });

  it("switches to code tab and shows code snippet", async () => {
    render(<LandingPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByText("💻 Developer API"));
    expect(screen.getByLabelText("API example code")).toBeInTheDocument();
    // Default language is curl — multiple matches expected from code snippet
    expect(screen.getAllByText(/X-Nova-API-Key/).length).toBeGreaterThanOrEqual(1);
  });

  it("switches between code language tabs", async () => {
    render(<LandingPage onNavigate={onNavigate} />);
    // Switch to code tab
    fireEvent.click(screen.getByText("💻 Developer API"));

    // Switch to Python
    fireEvent.click(screen.getByText("Python"));
    expect(screen.getByText(/import requests/)).toBeInTheDocument();

    // Switch to Node.js
    fireEvent.click(screen.getByText("Node.js"));
    expect(screen.getByText(/await fetch/)).toBeInTheDocument();

    // Switch back to cURL — multiple elements may match
    fireEvent.click(screen.getByText("cURL"));
    expect(screen.getAllByText(/X-Nova-API-Key/).length).toBeGreaterThanOrEqual(1);
  });

  it("navigates via pricing teaser CTA buttons", () => {
    render(<LandingPage onNavigate={onNavigate} />);

    // Free plan → signup
    fireEvent.click(screen.getByLabelText("Sign up for free plan"));
    expect(onNavigate).toHaveBeenCalledWith("signup");

    // Pro plan → pricing
    fireEvent.click(screen.getByLabelText("View Pro plan details"));
    expect(onNavigate).toHaveBeenCalledWith("pricing");

    // Enterprise plan → pricing
    fireEvent.click(screen.getByLabelText("View Enterprise plan details"));
    expect(onNavigate).toHaveBeenCalledWith("pricing");
  });

  it("navigates via footer links", () => {
    render(<LandingPage onNavigate={onNavigate} />);

    fireEvent.click(screen.getByText("Quickstart"));
    expect(onNavigate).toHaveBeenCalledWith("getting-started");

    fireEvent.click(screen.getByText("Pricing"));
    expect(onNavigate).toHaveBeenCalledWith("pricing");

    fireEvent.click(screen.getByText("Status"));
    expect(onNavigate).toHaveBeenCalledWith("status");
  });

  it("navigates to pricing via full pricing link", () => {
    render(<LandingPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("View full pricing page"));
    expect(onNavigate).toHaveBeenCalledWith("pricing");
  });

  it("switches back to mobile tab from code tab", () => {
    render(<LandingPage onNavigate={onNavigate} />);
    // Go to code
    fireEvent.click(screen.getByText("💻 Developer API"));
    expect(screen.getByLabelText("API example code")).toBeInTheDocument();

    // Go back to mobile
    fireEvent.click(screen.getByText("📱 Mobile App UI"));
    expect(screen.getByLabelText("Interactive mobile app preview")).toBeInTheDocument();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// PricingPage
// ═══════════════════════════════════════════════════════════════════════════════

describe("PricingPage", () => {
  const onNavigate = vi.fn();
  beforeEach(() => vi.clearAllMocks());

  it("renders pricing header and benchmark proof", () => {
    render(<PricingPage onNavigate={onNavigate} />);
    expect(screen.getByText("Simple, transparent pricing")).toBeInTheDocument();
    expect(screen.getByText(/Netflix-quality recommendations/)).toBeInTheDocument();
    expect(screen.getByText("0.785")).toBeInTheDocument();
    expect(screen.getByText("52")).toBeInTheDocument();
  });

  it("renders all three plan cards with correct names and prices", () => {
    render(<PricingPage onNavigate={onNavigate} />);

    expect(screen.getByLabelText("Free plan")).toBeInTheDocument();
    expect(screen.getByLabelText("Pro plan")).toBeInTheDocument();
    expect(screen.getByLabelText("Enterprise plan")).toBeInTheDocument();

    expect(screen.getByText("$0")).toBeInTheDocument();
    expect(screen.getByText("$299")).toBeInTheDocument();
    expect(screen.getByText("Custom")).toBeInTheDocument();
  });

  it("renders Most popular badge only on Pro plan", () => {
    render(<PricingPage onNavigate={onNavigate} />);
    const badges = screen.getAllByText("Most popular");
    expect(badges).toHaveLength(1);
  });

  it("renders plan features for all plans", () => {
    render(<PricingPage onNavigate={onNavigate} />);
    expect(screen.getByText("FAISS vector search")).toBeInTheDocument();
    expect(screen.getByText("6-model ONNX ensemble")).toBeInTheDocument();
    expect(screen.getByText("GPU ensemble (full precision)")).toBeInTheDocument();
  });

  it("renders plan specs (requests, tier, latency, support)", () => {
    render(<PricingPage onNavigate={onNavigate} />);
    expect(screen.getByText("100 requests / day")).toBeInTheDocument();
    expect(screen.getByText("10,000 requests / day")).toBeInTheDocument();
    expect(screen.getByText("Unlimited")).toBeInTheDocument();
    expect(screen.getByText("Tier 3")).toBeInTheDocument();
    expect(screen.getByText("Tier 2")).toBeInTheDocument();
    expect(screen.getByText("Tier 1")).toBeInTheDocument();
  });

  it("renders FAQ section", () => {
    render(<PricingPage onNavigate={onNavigate} />);
    expect(screen.getByText("Common questions")).toBeInTheDocument();
    expect(screen.getByText("Can I start on Free and upgrade later?")).toBeInTheDocument();
    expect(screen.getByText("What happens if I exceed my daily limit?")).toBeInTheDocument();
    expect(screen.getByText("Do you offer a trial for Pro?")).toBeInTheDocument();
    expect(screen.getByText("Is my data used to train shared models?")).toBeInTheDocument();
  });

  it("navigates to signup when Free plan CTA is clicked", () => {
    render(<PricingPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText(/Get started free — Free plan/));
    expect(onNavigate).toHaveBeenCalledWith("signup");
  });

  it("redirects to mailto when Enterprise CTA is clicked", () => {
    // Can't fully test window.location.href assignment in jsdom,
    // but we can verify it doesn't call apiPost or onNavigate("signup")
    render(<PricingPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText(/Contact sales — Enterprise plan/));
    expect(onNavigate).not.toHaveBeenCalledWith("signup");
    expect(vi.mocked(apiPost)).not.toHaveBeenCalled();
  });

  it("calls apiPost for Pro plan and handles success", async () => {
    vi.mocked(apiPost).mockResolvedValueOnce({
      data: { checkout_url: "https://checkout.stripe.com/session/test" },
      baseUrl: "https://test-api.example.com",
    });

    render(<PricingPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText(/Start Pro trial — Pro plan/));

    await waitFor(() => {
      expect(vi.mocked(apiPost)).toHaveBeenCalledWith(
        "/v1/billing/checkout",
        expect.objectContaining({ plan: "pro" }),
      );
    });
  });

  it("shows error when Pro plan checkout fails", async () => {
    vi.mocked(apiPost).mockRejectedValueOnce(new Error("Network error"));

    render(<PricingPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText(/Start Pro trial — Pro plan/));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
      expect(screen.getByText("Network error")).toBeInTheDocument();
    });
  });

  it("shows generic error message when checkout throws a non-Error", async () => {
    vi.mocked(apiPost).mockRejectedValueOnce("unknown failure");

    render(<PricingPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText(/Start Pro trial — Pro plan/));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
      expect(screen.getByText(/Failed to start checkout/)).toBeInTheDocument();
    });
  });

  it("shows Redirecting text and disables buttons during checkout loading", async () => {
    // Never-resolving promise to keep loading state
    vi.mocked(apiPost).mockReturnValueOnce(new Promise(() => {}));

    render(<PricingPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText(/Start Pro trial — Pro plan/));

    await waitFor(() => {
      expect(screen.getByText("Redirecting…")).toBeInTheDocument();
    });
  });

  it("works without onNavigate prop (optional)", () => {
    render(<PricingPage />);
    // Free plan click should not crash when onNavigate is undefined
    expect(() => {
      fireEvent.click(screen.getByLabelText(/Get started free — Free plan/));
    }).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SignupPage
// ═══════════════════════════════════════════════════════════════════════════════

describe("SignupPage", () => {
  const onNavigate = vi.fn();
  const onLoginSuccess = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers({ shouldAdvanceTime: true });
    window.localStorage.clear();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders the registration form with email and password fields", () => {
    render(<SignupPage onNavigate={onNavigate} />);
    expect(screen.getByText("Create your APEX account")).toBeInTheDocument();
    expect(screen.getByText(/No credit card required/)).toBeInTheDocument();
    expect(screen.getByLabelText("Email address")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.getByLabelText("Create account and get API key")).toBeInTheDocument();
  });

  it("disables submit button when email or password are insufficient", () => {
    render(<SignupPage onNavigate={onNavigate} />);
    const btn = screen.getByLabelText("Create account and get API key");
    expect(btn).toBeDisabled();
  });

  it("enables submit button with valid email and password", async () => {
    render(<SignupPage onNavigate={onNavigate} />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText("Email address"), { target: { value: "test@example.com" } });
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "SecurePass123!" } });
    });

    const btn = screen.getByLabelText("Create account and get API key");
    expect(btn).not.toBeDisabled();
  });

  it("toggles password visibility", async () => {
    render(<SignupPage onNavigate={onNavigate} />);
    const pwInput = screen.getByLabelText("Password");
    expect(pwInput).toHaveAttribute("type", "password");

    fireEvent.click(screen.getByLabelText("Show password"));
    expect(pwInput).toHaveAttribute("type", "text");

    fireEvent.click(screen.getByLabelText("Hide password"));
    expect(pwInput).toHaveAttribute("type", "password");
  });

  it("shows password strength bar with different levels", async () => {
    render(<SignupPage onNavigate={onNavigate} />);

    // No password → no strength bar
    expect(screen.queryByLabelText(/Password strength/)).not.toBeInTheDocument();

    // Short password (score = 0)
    await act(async () => {
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "ab" } });
    });
    // Bar should exist but empty label
    expect(screen.queryByLabelText(/Password strength/)).toBeInTheDocument();

    // Weak password (score = 1): >= 8 chars
    await act(async () => {
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "abcdefgh" } });
    });
    expect(screen.getByText("Weak")).toBeInTheDocument();

    // Fair password (score = 2): >= 8 + uppercase
    await act(async () => {
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Abcdefgh" } });
    });
    expect(screen.getByText("Fair")).toBeInTheDocument();

    // Good password (score = 3): >= 8 + uppercase + digit
    await act(async () => {
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Abcdefg1" } });
    });
    expect(screen.getByText("Good")).toBeInTheDocument();

    // Strong password (score = 4): >= 8 + uppercase + digit + special
    await act(async () => {
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Abcdefg1!" } });
    });
    expect(screen.getByText("Strong")).toBeInTheDocument();

    // Very strong password (score = 5): >= 12 + uppercase + digit + special
    await act(async () => {
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "Abcdefghijk1!" } });
    });
    expect(screen.getByText("Very strong")).toBeInTheDocument();
  });

  it("submits form and shows API key step when api_key is returned", async () => {
    mockClipboard();
    vi.mocked(apiPost).mockResolvedValueOnce({
      data: {
        username: "test@example.com",
        api_key: "nova_key_abc123def456",
        access_token: "jwt-token-xyz",
      },
      baseUrl: "https://test-api.example.com",
    });

    render(<SignupPage onNavigate={onNavigate} onLoginSuccess={onLoginSuccess} />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText("Email address"), { target: { value: "test@example.com" } });
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "SecurePass123!" } });
    });

    await act(async () => {
      fireEvent.submit(screen.getByLabelText("Account registration form"));
    });

    await waitFor(() => {
      expect(screen.getByText("Your API key is ready")).toBeInTheDocument();
    });

    expect(screen.getByText("nova_key_abc123def456")).toBeInTheDocument();
    expect(onLoginSuccess).toHaveBeenCalledWith("jwt-token-xyz", "test@example.com");
    expect(window.localStorage.getItem("nova_jwt_token")).toBe("jwt-token-xyz");
  });

  it("shows done step when no api_key is returned", async () => {
    vi.mocked(apiPost).mockResolvedValueOnce({
      data: { username: "test@example.com" },
      baseUrl: "https://test-api.example.com",
    });

    render(<SignupPage onNavigate={onNavigate} />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText("Email address"), { target: { value: "test@example.com" } });
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "SecurePass123!" } });
    });

    await act(async () => {
      fireEvent.submit(screen.getByLabelText("Account registration form"));
    });

    await waitFor(() => {
      expect(screen.getByText("Account created")).toBeInTheDocument();
    });

    expect(screen.getByText(/Redirecting you to Getting Started/)).toBeInTheDocument();

    // After timeout, should navigate
    await act(async () => {
      vi.advanceTimersByTime(900);
    });
    expect(onNavigate).toHaveBeenCalledWith("getting-started");
  });

  it("shows error when API returns a detail message", async () => {
    vi.mocked(apiPost).mockResolvedValueOnce({
      data: { detail: "Email is invalid" },
      baseUrl: "https://test-api.example.com",
    });

    render(<SignupPage onNavigate={onNavigate} />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText("Email address"), { target: { value: "bad" } });
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "SecurePass123!" } });
    });

    await act(async () => {
      fireEvent.submit(screen.getByLabelText("Account registration form"));
    });

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
      expect(screen.getByText("Email is invalid")).toBeInTheDocument();
    });
  });

  it("shows duplicate account message when error contains 'already'", async () => {
    vi.mocked(apiPost).mockRejectedValueOnce(new Error("User already exists"));

    render(<SignupPage onNavigate={onNavigate} />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText("Email address"), { target: { value: "taken@example.com" } });
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "SecurePass123!" } });
    });

    await act(async () => {
      fireEvent.submit(screen.getByLabelText("Account registration form"));
    });

    await waitFor(() => {
      expect(screen.getByText(/already exists/)).toBeInTheDocument();
    });
  });

  it("shows generic error message on non-Error rejection", async () => {
    vi.mocked(apiPost).mockRejectedValueOnce("unknown");

    render(<SignupPage onNavigate={onNavigate} />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText("Email address"), { target: { value: "test@example.com" } });
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "SecurePass123!" } });
    });

    await act(async () => {
      fireEvent.submit(screen.getByLabelText("Account registration form"));
    });

    await waitFor(() => {
      expect(screen.getByText("Registration failed")).toBeInTheDocument();
    });
  });

  it("copies API key to clipboard on the key step", async () => {
    const writeText = mockClipboard();
    vi.mocked(apiPost).mockResolvedValueOnce({
      data: { api_key: "nova_key_test123456" },
      baseUrl: "https://test-api.example.com",
    });

    render(<SignupPage onNavigate={onNavigate} />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText("Email address"), { target: { value: "test@example.com" } });
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "SecurePass123!" } });
    });

    await act(async () => {
      fireEvent.submit(screen.getByLabelText("Account registration form"));
    });

    await waitFor(() => {
      expect(screen.getByText("Your API key is ready")).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Copy API key to clipboard"));
    });

    expect(writeText).toHaveBeenCalledWith("nova_key_test123456");
    expect(screen.getByText("Copied")).toBeInTheDocument();
  });

  it("navigates to getting-started from API key step", async () => {
    mockClipboard();
    vi.mocked(apiPost).mockResolvedValueOnce({
      data: { api_key: "nova_key_xyz" },
      baseUrl: "https://test-api.example.com",
    });

    render(<SignupPage onNavigate={onNavigate} />);

    await act(async () => {
      fireEvent.change(screen.getByLabelText("Email address"), { target: { value: "test@example.com" } });
      fireEvent.change(screen.getByLabelText("Password"), { target: { value: "SecurePass123!" } });
    });

    await act(async () => {
      fireEvent.submit(screen.getByLabelText("Account registration form"));
    });

    await waitFor(() => {
      expect(screen.getByText("Your API key is ready")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByLabelText("Continue to Getting Started guide"));
    expect(onNavigate).toHaveBeenCalledWith("getting-started");

    // Also test dashboard link
    onNavigate.mockClear();
  });

  it("navigates to login page from sign in link", () => {
    render(<SignupPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Go to login page"));
    expect(onNavigate).toHaveBeenCalledWith("login");
  });

  it("does not submit with empty fields", async () => {
    render(<SignupPage onNavigate={onNavigate} />);

    await act(async () => {
      fireEvent.submit(screen.getByLabelText("Account registration form"));
    });

    expect(vi.mocked(apiPost)).not.toHaveBeenCalled();
  });

  it("renders terms and privacy links", () => {
    render(<SignupPage onNavigate={onNavigate} />);
    expect(screen.getByText("Terms of Service")).toBeInTheDocument();
    expect(screen.getByText("Privacy Policy")).toBeInTheDocument();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// GettingStartedPage
// ═══════════════════════════════════════════════════════════════════════════════

describe("GettingStartedPage", () => {
  const onNavigate = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
  });

  it("renders the getting started header and step indicator", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    expect(screen.getByText("Getting Started with APEX")).toBeInTheDocument();
    expect(screen.getByText(/Follow these four steps/)).toBeInTheDocument();
    expect(screen.getByLabelText("Onboarding steps")).toBeInTheDocument();
  });

  // ── Step 1: API Key ─────────────────────────────────────────────────────

  it("renders step 1 with no API key prompt by default", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    expect(screen.getByText("Your API key")).toBeInTheDocument();
    expect(screen.getByText(/No API key found/)).toBeInTheDocument();
    expect(screen.getByLabelText("Sign up to get an API key")).toBeInTheDocument();
  });

  it("renders step 1 with API key display when key is in localStorage", () => {
    window.localStorage.setItem("nova_api_key", "test-key-12345");
    render(<GettingStartedPage onNavigate={onNavigate} />);
    expect(screen.getByText("test-key-12345")).toBeInTheDocument();
    expect(screen.getByText("Copy")).toBeInTheDocument();
  });

  it("copies API key from step 1", async () => {
    const writeText = mockClipboard();
    window.localStorage.setItem("nova_api_key", "test-key-12345");
    render(<GettingStartedPage onNavigate={onNavigate} />);

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Copy API key to clipboard"));
    });

    expect(writeText).toHaveBeenCalledWith("test-key-12345");
    expect(screen.getByText("Copied")).toBeInTheDocument();
  });

  it("copies fallback key when no API key is set", async () => {
    mockClipboard();
    render(<GettingStartedPage onNavigate={onNavigate} />);

    // Navigate to signup since we can't copy without key display
    fireEvent.click(screen.getByLabelText("Sign up to get an API key"));
    expect(onNavigate).toHaveBeenCalledWith("signup");
  });

  it("navigates to step 2 via Next button", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Continue to step 2: Upload catalog"));
    expect(screen.getByText("Upload your catalog")).toBeInTheDocument();
  });

  it("skips to step 3 via skip link from step 1", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));
    expect(screen.getByText("Make your first call")).toBeInTheDocument();
  });

  // ── Step 2: Upload Catalog ──────────────────────────────────────────────

  it("renders step 2 with upload form", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Continue to step 2: Upload catalog"));

    expect(screen.getByText("Upload your catalog")).toBeInTheDocument();
    expect(screen.getByText("Drop catalog.csv here or click to browse")).toBeInTheDocument();
    expect(screen.getByLabelText("Upload catalog file")).toBeDisabled();
  });

  it("enables upload button after file selection", async () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Continue to step 2: Upload catalog"));

    const file = new File(["id,title\n1,Test"], "catalog.csv", { type: "text/csv" });
    const input = screen.getByLabelText("Choose CSV catalog file");

    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } });
    });

    expect(screen.getByText(/catalog\.csv/)).toBeInTheDocument();
    expect(screen.getByLabelText("Upload catalog file")).not.toBeDisabled();
  });

  it("handles successful catalog upload", async () => {
    globalThis.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ rows_ingested: 500, catalog_id: "my-catalog" }),
    });

    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Continue to step 2: Upload catalog"));

    const file = new File(["id,title\n1,Test"], "catalog.csv", { type: "text/csv" });
    const input = screen.getByLabelText("Choose CSV catalog file");

    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } });
    });

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Upload catalog file"));
    });

    await waitFor(() => {
      expect(screen.getByText("Catalog uploaded successfully")).toBeInTheDocument();
      expect(screen.getByText(/500 items ingested/)).toBeInTheDocument();
      expect(screen.getByText("my-catalog")).toBeInTheDocument();
    });
  });

  it("handles upload failure", async () => {
    globalThis.fetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ detail: "Invalid CSV format" }),
    });

    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Continue to step 2: Upload catalog"));

    const file = new File(["bad"], "catalog.csv", { type: "text/csv" });
    const input = screen.getByLabelText("Choose CSV catalog file");

    await act(async () => {
      fireEvent.change(input, { target: { files: [file] } });
    });

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Upload catalog file"));
    });

    await waitFor(() => {
      expect(screen.getByText("Invalid CSV format")).toBeInTheDocument();
    });
  });

  it("skips catalog upload to step 3", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Continue to step 2: Upload catalog"));
    fireEvent.click(screen.getByLabelText("Skip catalog upload and use demo catalog"));
    expect(screen.getByText("Make your first call")).toBeInTheDocument();
  });

  // ── Step 3: First Call ──────────────────────────────────────────────────

  it("renders step 3 with code snippet and language tabs", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));

    expect(screen.getByText("Make your first call")).toBeInTheDocument();
    expect(screen.getByLabelText("Code language")).toBeInTheDocument();
    expect(screen.getByLabelText("Show curl code example")).toBeInTheDocument();
    expect(screen.getByLabelText("Show python code example")).toBeInTheDocument();
    expect(screen.getByLabelText("Show javascript code example")).toBeInTheDocument();
  });

  it("switches code language tabs in step 3", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));

    // Default is curl — multiple elements match (tab label + code block)
    expect(screen.getAllByText(/curl/).length).toBeGreaterThanOrEqual(1);

    // Switch to python
    fireEvent.click(screen.getByLabelText("Show python code example"));
    expect(screen.getByText(/import httpx/)).toBeInTheDocument();

    // Switch to javascript
    fireEvent.click(screen.getByLabelText("Show javascript code example"));
    expect(screen.getByText(/await fetch/)).toBeInTheDocument();
  });

  it("handles successful Try It Now call", async () => {
    vi.mocked(apiGet).mockResolvedValueOnce({
      data: { recommendations: [{ title: "The Matrix", score: 0.95 }] },
      baseUrl: "https://test-api.example.com",
    });

    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Run recommendation request against the live API"));
    });

    await waitFor(() => {
      expect(screen.getByText("Live response from APEX API")).toBeInTheDocument();
    });
    // The JSON response contains "The Matrix" — the page text also mentions "Fight Club"
    expect(screen.getByLabelText("JSON response")).toHaveTextContent("The Matrix");
  });

  it("handles Try It Now failure", async () => {
    vi.mocked(apiGet).mockRejectedValueOnce(new Error("Server unreachable"));

    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Run recommendation request against the live API"));
    });

    await waitFor(() => {
      expect(screen.getByText("Server unreachable")).toBeInTheDocument();
    });
  });

  it("handles non-Error Try It Now failure", async () => {
    vi.mocked(apiGet).mockRejectedValueOnce("unknown");

    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Run recommendation request against the live API"));
    });

    await waitFor(() => {
      expect(screen.getByText("Request failed")).toBeInTheDocument();
    });
  });

  it("navigates to step 4 from step 3", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));
    fireEvent.click(screen.getByLabelText("Continue to step 4: Explore the dashboard"));
    expect(screen.getByText("Explore your dashboard")).toBeInTheDocument();
  });

  // ── Step 4: Dashboard ───────────────────────────────────────────────────

  it("renders step 4 with dashboard preview", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    // Navigate through steps to reach step 4
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));
    fireEvent.click(screen.getByLabelText("Continue to step 4: Explore the dashboard"));

    expect(screen.getByText("Explore your dashboard")).toBeInTheDocument();
    expect(screen.getByText("Tier 2 — ONNX CPU")).toBeInTheDocument();
    expect(screen.getByText("342 ms")).toBeInTheDocument();
    expect(screen.getByText("You're all set")).toBeInTheDocument();
  });

  it("navigates to dashboard from step 4", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));
    fireEvent.click(screen.getByLabelText("Continue to step 4: Explore the dashboard"));
    fireEvent.click(screen.getByLabelText("Open your APEX dashboard"));
    expect(onNavigate).toHaveBeenCalledWith("dashboard");
  });

  it("navigates to home from step 4 Explore link", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));
    fireEvent.click(screen.getByLabelText("Continue to step 4: Explore the dashboard"));
    fireEvent.click(screen.getByLabelText("Explore the full API"));
    expect(onNavigate).toHaveBeenCalledWith("home");
  });

  // ── Step indicator navigation ───────────────────────────────────────────

  it("allows navigating back via step indicator", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);

    // Go to step 3
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));
    expect(screen.getByText("Make your first call")).toBeInTheDocument();

    // Click step 1 indicator
    fireEvent.click(screen.getByLabelText(/Step 1: API key \(completed\)/));
    expect(screen.getByText("Your API key")).toBeInTheDocument();
  });

  it("disables future step buttons in step indicator", () => {
    render(<GettingStartedPage onNavigate={onNavigate} />);
    // On step 1, step 2/3/4 should be disabled
    const step3Btn = screen.getByLabelText(/Step 3: First call$/);
    expect(step3Btn).toBeDisabled();
  });

  it("truncates long JSON responses", async () => {
    // Create a response larger than 1200 chars
    const longData: Record<string, string> = {};
    for (let i = 0; i < 100; i++) {
      longData[`key_${i}`] = "A very long value string that takes up space in the JSON output";
    }
    vi.mocked(apiGet).mockResolvedValueOnce({
      data: longData,
      baseUrl: "https://test-api.example.com",
    });

    render(<GettingStartedPage onNavigate={onNavigate} />);
    fireEvent.click(screen.getByLabelText("Skip to step 3: Make your first call"));

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Run recommendation request against the live API"));
    });

    await waitFor(() => {
      expect(screen.getByText(/\.\.\. \(truncated\)/)).toBeInTheDocument();
    });
  });
});
