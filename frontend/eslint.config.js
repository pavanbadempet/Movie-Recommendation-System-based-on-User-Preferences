import js from "@eslint/js";
import tsPlugin from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import reactPlugin from "eslint-plugin-react";
import reactHooksPlugin from "eslint-plugin-react-hooks";
import jsxA11yPlugin from "eslint-plugin-jsx-a11y";

/** Vitest globals injected by the `globals: true` setting in vite.config.ts */
const vitestGlobals = {
  describe: "readonly",
  it: "readonly",
  test: "readonly",
  expect: "readonly",
  beforeEach: "readonly",
  afterEach: "readonly",
  beforeAll: "readonly",
  afterAll: "readonly",
  vi: "readonly",
};

const browserGlobals = {
  window: "readonly",
  document: "readonly",
  console: "readonly",
  fetch: "readonly",
  setTimeout: "readonly",
  clearTimeout: "readonly",
  AbortController: "readonly",
  AbortSignal: "readonly",
  DOMException: "readonly",
  URL: "readonly",
  URLSearchParams: "readonly",
  localStorage: "readonly",
  sessionStorage: "readonly",
  crypto: "readonly",
  HTMLElement: "readonly",
  HTMLButtonElement: "readonly",
  HTMLInputElement: "readonly",
  HTMLIFrameElement: "readonly",
  HTMLDivElement: "readonly",
  KeyboardEvent: "readonly",
  MouseEvent: "readonly",
  Event: "readonly",
  Response: "readonly",
  Request: "readonly",
  RequestInit: "readonly",
  Headers: "readonly",
  FormData: "readonly",
  MutationObserver: "readonly",
  ResizeObserver: "readonly",
  IntersectionObserver: "readonly",
  requestAnimationFrame: "readonly",
  cancelAnimationFrame: "readonly",
  SVGSVGElement: "readonly",
  SVGGElement: "readonly",
  SVGElement: "readonly",
  SVGCircleElement: "readonly",
  SVGTextElement: "readonly",
  SVGLineElement: "readonly",
  HTMLDetailsElement: "readonly",
  PopStateEvent: "readonly",
  Node: "readonly",
  Element: "readonly",
  navigator: "readonly",
  File: "readonly",
  HTMLLabelElement: "readonly",
};

export default [
  js.configs.recommended,
  {
    files: ["src/**/*.{ts,tsx}"],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        ecmaFeatures: { jsx: true },
      },
      globals: browserGlobals,
    },
    plugins: {
      "@typescript-eslint": tsPlugin,
      react: reactPlugin,
      "react-hooks": reactHooksPlugin,
      "jsx-a11y": jsxA11yPlugin,
    },
    settings: {
      react: { version: "detect" },
    },
    rules: {
      // TypeScript
      ...tsPlugin.configs.recommended.rules,
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],

      // React
      ...reactPlugin.configs.recommended.rules,
      "react/react-in-jsx-scope": "off", // Not needed with React 17+ JSX transform
      "react/prop-types": "off",         // TypeScript handles this

      // React Hooks
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",

      // Accessibility
      ...jsxA11yPlugin.configs.recommended.rules,

      // General
      "no-console": ["warn", { allow: ["warn", "error"] }],
    },
  },
  {
    // Relax rules for test files and add Vitest globals
    files: ["src/test/**/*.{ts,tsx}"],
    languageOptions: {
      globals: {
        ...browserGlobals,
        ...vitestGlobals,
      },
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "no-console": "off",
    },
  },
  {
    ignores: ["dist/**", "node_modules/**", "coverage/**", "eslint.config.js", "src/jest-axe.d.ts", "src/vite-env.d.ts"],
  },
];
