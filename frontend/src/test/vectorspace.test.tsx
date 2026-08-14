import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { VectorSpace } from "../VectorSpace";

describe("VectorSpace Component", () => {
  beforeAll(() => {
    vi.stubGlobal("requestAnimationFrame", vi.fn());
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
  });

  afterAll(() => {
    vi.unstubAllGlobals();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test("renders 3D Neural Vector Galaxy title and controls", () => {
    vi.useFakeTimers();
    render(<VectorSpace />);
    expect(screen.getByText("3D Neural Vector Galaxy")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Search movie...")).toBeInTheDocument();
  });

  test("can interact with zoom controls and search", () => {
    render(<VectorSpace />);
    const searchInput = screen.getByPlaceholderText("Search movie...");
    expect(searchInput).toBeInTheDocument();
    
    fireEvent.change(searchInput, { target: { value: "Inception" } });
    expect(searchInput).toHaveValue("Inception");

    const zoomInBtn = screen.getByTitle("Zoom In");
    expect(zoomInBtn).toBeInTheDocument();
    fireEvent.click(zoomInBtn);

    const zoomOutBtn = screen.getByTitle("Zoom Out");
    expect(zoomOutBtn).toBeInTheDocument();
    fireEvent.click(zoomOutBtn);
  });
});
