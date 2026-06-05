/**
 * Extended tests for UserProfilePage covering branches not hit in pages.test.tsx:
 *  - Watch history section with localStorage data
 *  - Features error state
 *  - Recommendations error state
 *  - Negative / invalid behavior feature values
 */

import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

vi.mock("../api", () => ({
  apiGet: vi.fn(),
  getUserRecommendations: vi.fn(),
}));

import { apiGet, getUserRecommendations } from "../api";
import { UserProfilePage } from "../pages/UserProfile";
import type { Movie } from "../types";

const RECENT_KEY = "nova_recent_movies_v2";

const sampleMovie: Movie = {
  id: 1,
  title: "Avatar",
  poster_path: "/avatar.jpg",
  release_date: "2009-12-18",
  genres: "Action, Adventure",
};

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
});

// ─── Watch history ────────────────────────────────────────────────────────────

describe("UserProfilePage — watch history", () => {
  it("shows watch history when localStorage has recent movies", async () => {
    window.localStorage.setItem(RECENT_KEY, JSON.stringify([sampleMovie]));
    vi.mocked(apiGet).mockResolvedValue({ data: {}, baseUrl: "http://localhost:8000" });
    vi.mocked(getUserRecommendations).mockResolvedValue({ data: [], baseUrl: "http://localhost:8000" });

    render(
      <UserProfilePage
        token="tok"
        username="alice"
        onRequestLogin={vi.fn()}
        onSelectMovie={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Avatar")).toBeInTheDocument();
    });
  });

  it("shows empty history message when localStorage is empty", async () => {
    vi.mocked(apiGet).mockResolvedValue({ data: {}, baseUrl: "http://localhost:8000" });
    vi.mocked(getUserRecommendations).mockResolvedValue({ data: [], baseUrl: "http://localhost:8000" });

    render(
      <UserProfilePage
        token="tok"
        username="alice"
        onRequestLogin={vi.fn()}
        onSelectMovie={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText(/no watch history yet/i)).toBeInTheDocument();
    });
  });

  it("calls onSelectMovie when a history movie card is clicked", async () => {
    window.localStorage.setItem(RECENT_KEY, JSON.stringify([sampleMovie]));
    vi.mocked(apiGet).mockResolvedValue({ data: {}, baseUrl: "http://localhost:8000" });
    vi.mocked(getUserRecommendations).mockResolvedValue({ data: [], baseUrl: "http://localhost:8000" });

    const onSelectMovie = vi.fn();
    render(
      <UserProfilePage
        token="tok"
        username="alice"
        onRequestLogin={vi.fn()}
        onSelectMovie={onSelectMovie}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Avatar")).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /view details for avatar/i }));
    expect(onSelectMovie).toHaveBeenCalledWith(sampleMovie);
  });
});

// ─── Error states ─────────────────────────────────────────────────────────────

describe("UserProfilePage — error states", () => {
  it("shows features error when apiGet fails", async () => {
    vi.mocked(apiGet).mockRejectedValue(new Error("Features unavailable"));
    vi.mocked(getUserRecommendations).mockResolvedValue({ data: [], baseUrl: "http://localhost:8000" });

    render(
      <UserProfilePage
        token="tok"
        username="alice"
        onRequestLogin={vi.fn()}
        onSelectMovie={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText(/features unavailable/i)).toBeInTheDocument();
    });
  });

  it("shows recommendations error when getUserRecommendations fails", async () => {
    vi.mocked(apiGet).mockResolvedValue({ data: {}, baseUrl: "http://localhost:8000" });
    vi.mocked(getUserRecommendations).mockRejectedValue(new Error("Recs unavailable"));

    render(
      <UserProfilePage
        token="tok"
        username="alice"
        onRequestLogin={vi.fn()}
        onSelectMovie={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText(/recs unavailable/i)).toBeInTheDocument();
    });
  });
});

// ─── BehaviorCard edge cases ──────────────────────────────────────────────────

describe("UserProfilePage — behavior card edge cases", () => {
  it("shows dashes for negative feature values", async () => {
    vi.mocked(apiGet).mockResolvedValue({
      data: { total_ratings: -1, avg_rating: -5, click_count: -3, view_count: -2 },
      baseUrl: "http://localhost:8000",
    });
    vi.mocked(getUserRecommendations).mockResolvedValue({ data: [], baseUrl: "http://localhost:8000" });

    render(
      <UserProfilePage
        token="tok"
        username="alice"
        onRequestLogin={vi.fn()}
        onSelectMovie={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText(/behavior statistics/i)).toBeInTheDocument();
    });
    // All negative values should render as "—"
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(4);
  });

  it("shows formatted values for valid feature data", async () => {
    vi.mocked(apiGet).mockResolvedValue({
      data: { total_ratings: 42, avg_rating: 4.2, click_count: 100, view_count: 200 },
      baseUrl: "http://localhost:8000",
    });
    vi.mocked(getUserRecommendations).mockResolvedValue({ data: [], baseUrl: "http://localhost:8000" });

    render(
      <UserProfilePage
        token="tok"
        username="alice"
        onRequestLogin={vi.fn()}
        onSelectMovie={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("42")).toBeInTheDocument();
      expect(screen.getByText("4.2")).toBeInTheDocument();
    });
  });
});

// ─── Recommendations grid ─────────────────────────────────────────────────────

describe("UserProfilePage — recommendations grid", () => {
  it("renders recommendation cards when recs are returned", async () => {
    vi.mocked(apiGet).mockResolvedValue({ data: {}, baseUrl: "http://localhost:8000" });
    vi.mocked(getUserRecommendations).mockResolvedValue({
      data: [
        { id: 1, title: "Avatar", genres: "Action" },
        { id: 2, title: "Inception", genres: "Sci-Fi" },
      ],
      baseUrl: "http://localhost:8000",
    });

    render(
      <UserProfilePage
        token="tok"
        username="alice"
        onRequestLogin={vi.fn()}
        onSelectMovie={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Avatar")).toBeInTheDocument();
      expect(screen.getByText("Inception")).toBeInTheDocument();
    });
  });
});
