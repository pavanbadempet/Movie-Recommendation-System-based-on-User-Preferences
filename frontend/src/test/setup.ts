import "@testing-library/jest-dom";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeAll, afterAll, vi } from "vitest";

const originalLocation = window.location;

beforeAll(() => {
  Object.defineProperty(window, "location", {
    writable: true,
    value: { ...originalLocation, href: "", assign: vi.fn(), replace: vi.fn() },
  });
});

afterAll(() => {
  window.location = originalLocation;
});

afterEach(() => {
  cleanup();
});
