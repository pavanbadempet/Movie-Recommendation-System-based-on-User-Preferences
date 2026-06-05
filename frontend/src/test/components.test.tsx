/**
 * Component tests for the Nova Recommendation UI.
 *
 * Covers:
 *  - AuthPage: renders login/register forms, handles input, shows errors,
 *    shows loading state, and toggles between modes.
 *  - MovieCard (inline): renders movie title, poster, rating, and feedback.
 *  - ErrorBanner / LoadingSpinner: simple UI state helpers.
 *
 * All API calls are mocked so no real network requests are made.
 */

import React from "react";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

// ─── Mock the API module ──────────────────────────────────────────────────────

vi.mock("../api", () => ({
  loginUser: vi.fn(),
  registerUser: vi.fn(),
  backendLabel: vi.fn((url: string) => url),
  currentBackend: vi.fn(() => "http://localhost:8000"),
  recordEvent: vi.fn(),
}));

import { loginUser, registerUser } from "../api";
import { AuthPage } from "../AuthPage";
import type { Movie } from "../types";

// ─── AuthPage ─────────────────────────────────────────────────────────────────

describe("AuthPage", () => {
  const mockOnLogin = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    // Stub only localStorage, not the whole window object
    vi.spyOn(window.localStorage.__proto__, "getItem").mockReturnValue(null);
    vi.spyOn(window.localStorage.__proto__, "setItem").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ── Rendering ──────────────────────────────────────────────────────────────

  it("renders the login form by default", () => {
    render(<AuthPage onLogin={mockOnLogin} />);
    expect(screen.getByPlaceholderText(/username/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/password/i)).toBeInTheDocument();
    // The submit button text contains "Sign In"
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("shows 'Welcome Back' heading in login mode", () => {
    render(<AuthPage onLogin={mockOnLogin} />);
    expect(screen.getByText(/welcome back/i)).toBeInTheDocument();
  });

  it("shows 'Join Nova' heading after switching to register mode", async () => {
    render(<AuthPage onLogin={mockOnLogin} />);
    await userEvent.click(
      screen.getByRole("button", { name: /don't have an account/i }),
    );
    expect(screen.getByText(/join nova/i)).toBeInTheDocument();
  });

  it("renders 'Create Account' submit button in register mode", async () => {
    render(<AuthPage onLogin={mockOnLogin} />);
    await userEvent.click(
      screen.getByRole("button", { name: /don't have an account/i }),
    );
    expect(
      screen.getByRole("button", { name: /create account/i }),
    ).toBeInTheDocument();
  });

  // ── User input ─────────────────────────────────────────────────────────────

  it("accepts text in the username field", async () => {
    render(<AuthPage onLogin={mockOnLogin} />);
    const usernameInput = screen.getByPlaceholderText(/username/i);
    await userEvent.type(usernameInput, "alice");
    expect(usernameInput).toHaveValue("alice");
  });

  it("accepts text in the password field", async () => {
    render(<AuthPage onLogin={mockOnLogin} />);
    const passwordInput = screen.getByPlaceholderText(/password/i);
    await userEvent.type(passwordInput, "s3cr3t");
    expect(passwordInput).toHaveValue("s3cr3t");
  });

  // ── Validation ─────────────────────────────────────────────────────────────

  it("shows an error when submitting with empty fields", async () => {
    render(<AuthPage onLogin={mockOnLogin} />);
    // Submit the form directly without filling in any fields
    const form = screen.getByPlaceholderText(/username/i).closest("form")!;
    await act(async () => {
      fireEvent.submit(form);
    });
    expect(
      screen.getByText(/username and password are required/i),
    ).toBeInTheDocument();
  });

  it("does not call loginUser when fields are empty", async () => {
    render(<AuthPage onLogin={mockOnLogin} />);
    const form = screen.getByPlaceholderText(/username/i).closest("form")!;
    await act(async () => {
      fireEvent.submit(form);
    });
    expect(loginUser).not.toHaveBeenCalled();
  });

  // ── Loading state ──────────────────────────────────────────────────────────

  it("disables the submit button while loading", async () => {
    // loginUser never resolves → component stays in loading state
    vi.mocked(loginUser).mockReturnValue(new Promise(() => {}));

    render(<AuthPage onLogin={mockOnLogin} />);
    await userEvent.type(screen.getByPlaceholderText(/username/i), "alice");
    await userEvent.type(screen.getByPlaceholderText(/password/i), "secret");

    const form = screen.getByPlaceholderText(/username/i).closest("form")!;
    await act(async () => {
      fireEvent.submit(form);
    });

    // After submit the button is disabled (loading spinner replaces text)
    const submitBtn = form.querySelector("button[type='submit']") as HTMLButtonElement;
    expect(submitBtn).toBeDisabled();
  });

  it("disables the username input while loading", async () => {
    vi.mocked(loginUser).mockReturnValue(new Promise(() => {}));

    render(<AuthPage onLogin={mockOnLogin} />);
    await userEvent.type(screen.getByPlaceholderText(/username/i), "alice");
    await userEvent.type(screen.getByPlaceholderText(/password/i), "secret");

    const form = screen.getByPlaceholderText(/username/i).closest("form")!;
    await act(async () => {
      fireEvent.submit(form);
    });

    expect(screen.getByPlaceholderText(/username/i)).toBeDisabled();
  });

  // ── Successful login ───────────────────────────────────────────────────────

  it("calls onLogin with the token on successful sign-in", async () => {
    vi.mocked(loginUser).mockResolvedValue({
      data: { access_token: "tok-abc", token_type: "bearer" },
      baseUrl: "http://localhost:8000",
    });

    render(<AuthPage onLogin={mockOnLogin} />);
    await userEvent.type(screen.getByPlaceholderText(/username/i), "alice");
    await userEvent.type(screen.getByPlaceholderText(/password/i), "secret");

    const form = screen.getByPlaceholderText(/username/i).closest("form")!;
    await act(async () => {
      fireEvent.submit(form);
    });

    await waitFor(() => {
      expect(mockOnLogin).toHaveBeenCalledWith("tok-abc", "alice");
    });
  });

  // ── Error state ────────────────────────────────────────────────────────────

  it("shows an error message when loginUser throws", async () => {
    vi.mocked(loginUser).mockRejectedValue(new Error("Invalid credentials"));

    render(<AuthPage onLogin={mockOnLogin} />);
    await userEvent.type(screen.getByPlaceholderText(/username/i), "alice");
    await userEvent.type(screen.getByPlaceholderText(/password/i), "wrong");

    const form = screen.getByPlaceholderText(/username/i).closest("form")!;
    await act(async () => {
      fireEvent.submit(form);
    });

    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
  });

  it("clears the error when toggling between login and register", async () => {
    vi.mocked(loginUser).mockRejectedValue(new Error("Bad credentials"));

    render(<AuthPage onLogin={mockOnLogin} />);
    await userEvent.type(screen.getByPlaceholderText(/username/i), "alice");
    await userEvent.type(screen.getByPlaceholderText(/password/i), "wrong");

    const form = screen.getByPlaceholderText(/username/i).closest("form")!;
    await act(async () => {
      fireEvent.submit(form);
    });

    await waitFor(() => {
      expect(screen.getByText(/bad credentials/i)).toBeInTheDocument();
    });

    // Toggle to register — error should disappear
    await userEvent.click(
      screen.getByRole("button", { name: /don't have an account/i }),
    );
    expect(screen.queryByText(/bad credentials/i)).not.toBeInTheDocument();
  });

  // ── Registration ───────────────────────────────────────────────────────────

  it("calls registerUser and shows success message on registration", async () => {
    vi.mocked(registerUser).mockResolvedValue({
      data: { username: "bob" },
      baseUrl: "http://localhost:8000",
    });

    render(<AuthPage onLogin={mockOnLogin} />);

    // Switch to register mode
    await userEvent.click(
      screen.getByRole("button", { name: /don't have an account/i }),
    );

    await userEvent.type(screen.getByPlaceholderText(/username/i), "bob");
    await userEvent.type(
      screen.getByPlaceholderText(/create password/i),
      "pass123",
    );

    const form = screen.getByPlaceholderText(/username/i).closest("form")!;
    await act(async () => {
      fireEvent.submit(form);
    });

    await waitFor(() => {
      expect(registerUser).toHaveBeenCalledWith("bob", "pass123");
      expect(screen.getByText(/registration successful/i)).toBeInTheDocument();
    });
  });
});

// ─── MovieCard (inline component) ────────────────────────────────────────────

interface MovieCardProps {
  movie: Movie;
  onSelect: (movie: Movie) => void;
}

function MovieCard({ movie, onSelect }: MovieCardProps) {
  const posterSrc = movie.poster_path
    ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
    : "https://placehold.co/500x750/141418/f8fafc?text=Movie";

  return (
    <article data-testid="movie-card">
      <button type="button" onClick={() => onSelect(movie)}>
        <img src={posterSrc} alt={movie.title} />
      </button>
      <h3>{movie.title}</h3>
      {movie.vote_average != null && (
        <span data-testid="rating">{movie.vote_average.toFixed(1)}</span>
      )}
      {movie.genres && <span data-testid="genres">{movie.genres}</span>}
      {movie.release_date && (
        <span data-testid="year">{movie.release_date.slice(0, 4)}</span>
      )}
    </article>
  );
}

describe("MovieCard", () => {
  const sampleMovie: Movie = {
    id: 19995,
    title: "Avatar",
    poster_path: "/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg",
    vote_average: 7.6,
    genres: "Action, Adventure, Fantasy",
    release_date: "2009-12-18",
    overview: "A paraplegic Marine dispatched to the moon Pandora.",
  };

  it("renders the movie title", () => {
    render(<MovieCard movie={sampleMovie} onSelect={vi.fn()} />);
    expect(screen.getByText("Avatar")).toBeInTheDocument();
  });

  it("renders the poster image with correct alt text", () => {
    render(<MovieCard movie={sampleMovie} onSelect={vi.fn()} />);
    const img = screen.getByRole("img", { name: "Avatar" });
    expect(img).toBeInTheDocument();
    expect(img).toHaveAttribute(
      "src",
      "https://image.tmdb.org/t/p/w500/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg",
    );
  });

  it("uses a placeholder image when poster_path is null", () => {
    const movieNoPoster: Movie = { ...sampleMovie, poster_path: null };
    render(<MovieCard movie={movieNoPoster} onSelect={vi.fn()} />);
    const img = screen.getByRole("img");
    expect(img.getAttribute("src")).toContain("placehold.co");
  });

  it("renders the rating", () => {
    render(<MovieCard movie={sampleMovie} onSelect={vi.fn()} />);
    expect(screen.getByTestId("rating")).toHaveTextContent("7.6");
  });

  it("renders the genres", () => {
    render(<MovieCard movie={sampleMovie} onSelect={vi.fn()} />);
    expect(screen.getByTestId("genres")).toHaveTextContent(
      "Action, Adventure, Fantasy",
    );
  });

  it("renders the release year", () => {
    render(<MovieCard movie={sampleMovie} onSelect={vi.fn()} />);
    expect(screen.getByTestId("year")).toHaveTextContent("2009");
  });

  it("calls onSelect with the movie when the poster button is clicked", async () => {
    const onSelect = vi.fn();
    render(<MovieCard movie={sampleMovie} onSelect={onSelect} />);
    await userEvent.click(screen.getByRole("button"));
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith(sampleMovie);
  });

  it("renders without crashing when optional fields are absent", () => {
    const minimalMovie: Movie = { id: 1, title: "Unknown Film" };
    render(<MovieCard movie={minimalMovie} onSelect={vi.fn()} />);
    expect(screen.getByText("Unknown Film")).toBeInTheDocument();
    expect(screen.queryByTestId("rating")).not.toBeInTheDocument();
    expect(screen.queryByTestId("genres")).not.toBeInTheDocument();
  });
});

// ─── Error / Loading state helpers ───────────────────────────────────────────

function ErrorBanner({ message }: { message: string }) {
  if (!message) return null;
  return (
    <div role="alert" className="error-banner">
      {message}
    </div>
  );
}

function LoadingSpinner({ loading }: { loading: boolean }) {
  if (!loading) return null;
  return (
    <div role="status" aria-label="Loading">
      Loading…
    </div>
  );
}

describe("ErrorBanner", () => {
  it("renders the error message", () => {
    render(<ErrorBanner message="Something went wrong" />);
    expect(screen.getByRole("alert")).toHaveTextContent("Something went wrong");
  });

  it("renders nothing when message is empty", () => {
    const { container } = render(<ErrorBanner message="" />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("LoadingSpinner", () => {
  it("renders when loading is true", () => {
    render(<LoadingSpinner loading={true} />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("renders nothing when loading is false", () => {
    const { container } = render(<LoadingSpinner loading={false} />);
    expect(container).toBeEmptyDOMElement();
  });
});
