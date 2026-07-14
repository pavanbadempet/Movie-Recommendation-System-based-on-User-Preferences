/// <reference types="vitest/globals" />
import "@testing-library/jest-dom";
import { cleanup } from "@testing-library/react";

const originalLocation = window.location;

beforeAll(() => {
  Object.defineProperty(window, "location", {
    writable: true,
    value: { ...originalLocation, href: "", assign: vi.fn(), replace: vi.fn() },
  });
});

afterAll(() => {
  window.location = originalLocation as any;
});

afterEach(() => {
  cleanup();
});
