/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import viteCompression from "vite-plugin-compression";

export default defineConfig({
  base: process.env.VITE_BASE_PATH || "./",
  plugins: [
    react(),
    viteCompression({ algorithm: "brotliCompress", ext: ".br" }),
    viteCompression({ algorithm: "gzip", ext: ".gz" })
  ],
  server: {
    port: 5173,
  },
  build: {
    target: "esnext",
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/react/") || id.includes("node_modules/react-dom/")) {
            return "react";
          }
          if (id.includes("node_modules/d3")) {
            return "d3";
          }
          if (id.includes("node_modules/lucide-react")) {
            return "lucide";
          }
          if (id.includes("node_modules/onnxruntime-web")) {
            return "onnx";
          }
        }
      }
    }
  },
  test: {
    pool: "forks",
    fileParallelism: false,
    teardownTimeout: 1000,
    hookTimeout: 5000,
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    exclude: ["e2e/**", "node_modules/**", "dist/**"],
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov", "json"],
      clean: false,
      cleanOnRebuild: false,
      thresholds: {
        branches: 70,
      },
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
        "src/VectorSpace.tsx",
        "src/webgpuEngine.ts",
        "src/test/**",
        "node_modules/**",
        "dist/**",
        "vite.config.ts",
        "eslint.config.js",
      ],
    },
  },
});
