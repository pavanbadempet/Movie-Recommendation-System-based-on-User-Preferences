/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: process.env.VITE_BASE_PATH || "/",
  plugins: [react()],
  server: {
    port: 5173,
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    exclude: ["e2e/**", "node_modules/**", "dist/**"],
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      // Exclude files that cannot be meaningfully unit-tested in jsdom:
      //  - main.tsx: monolithic app shell (1,955 lines), integration-tested via E2E
      //  - types.ts / vite-env.d.ts / jest-axe.d.ts: pure type declarations
      //  - KnowledgeGraph.tsx: contains a D3 ForceGraph component that uses canvas/SVG
      //    layout APIs unavailable in jsdom — the page shell and search UI are tested
      //    via accessibility.test.tsx and pages.test.tsx; the D3 rendering requires E2E
      //  - dist/: compiled build artifacts
      //  - vite.config.ts: build config, not application code
      exclude: [
        "src/main.tsx",
        "src/types.ts",
        "src/vite-env.d.ts",
        "src/jest-axe.d.ts",
        "src/pages/KnowledgeGraph.tsx",
        "src/test/**",
        "node_modules/**",
        "dist/**",
        "vite.config.ts",
        "eslint.config.js",
      ],
      thresholds: { lines: 75, branches: 70, functions: 60, statements: 75 },
    },
  },
});
