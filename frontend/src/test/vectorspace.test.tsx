import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { VectorSpace } from "../VectorSpace";

describe("VectorSpace Component", () => {
  test("renders 3D embedding title and controls", () => {
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
