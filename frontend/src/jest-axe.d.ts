/**
 * Type shim for jest-axe@8 with Vitest.
 * jest-axe ships CommonJS without bundled .d.ts for ESM consumers.
 */
declare module "jest-axe" {
  export interface AxeResults {
    violations: AxeViolation[];
    passes: unknown[];
    incomplete: unknown[];
    inapplicable: unknown[];
  }

  export interface AxeViolation {
    id: string;
    impact: string | null;
    description: string;
    nodes: unknown[];
  }

  export interface RunOptions {
    runOnly?: {
      type: "tag" | "rule";
      values: string[];
    };
    rules?: Record<string, { enabled: boolean }>;
  }

  export function axe(
    html: Element | string,
    options?: RunOptions,
  ): Promise<AxeResults>;

  export function toHaveNoViolations(): {
    pass: boolean;
    message: () => string;
  };
}

// Extend Vitest's Assertion interface with jest-axe matchers
declare namespace Vi {
  interface Assertion {
    toHaveNoViolations(): void;
  }
}

declare module "vitest" {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  interface Assertion<T = any> {
    toHaveNoViolations(): void;
  }
  interface AsymmetricMatchersContaining {
    toHaveNoViolations(): void;
  }
}
