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

  test("renders 3D embedding title and controls", () => {
    vi.useFakeTimers();
    render(<VectorSpace />);
    expect(screen.getByText("APEX 3D Movie Embedding Space")).toBeInTheDocument();
    expect(screen.getByText("Pause Auto-Rotate")).toBeInTheDocument();
  });

  test("can toggle auto-rotate state", () => {
    render(<VectorSpace />);
    const button = screen.getByText("Pause Auto-Rotate");
    expect(button).toBeInTheDocument();
    
    fireEvent.click(button);
    expect(screen.getByText("Auto-Rotate")).toBeInTheDocument();
  });
});
