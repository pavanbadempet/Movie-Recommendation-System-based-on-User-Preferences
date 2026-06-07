import React from "react";

interface Props {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * React ErrorBoundary — catches render errors in the component tree and
 * displays a graceful fallback instead of a blank screen.
 */
export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Log to console in development; in production this would go to Sentry.
    console.error("[ErrorBoundary] Uncaught render error:", error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div
          role="alert"
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "200px",
            padding: "32px",
            color: "#f0f2f5",
            background: "rgba(124,58,237,0.08)",
            border: "1px solid rgba(124,58,237,0.3)",
            borderRadius: "12px",
            margin: "24px",
            gap: "12px",
          }}
        >
          <strong style={{ fontSize: "1.1rem" }}>Something went wrong</strong>
          <p style={{ color: "#9ca3b4", margin: 0, fontSize: "0.9rem" }}>
            {this.state.error?.message ?? "An unexpected error occurred."}
          </p>
          <button
            type="button"
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              marginTop: "8px",
              padding: "8px 20px",
              color: "#fff",
              border: "1px solid rgba(124,58,237,0.5)",
              borderRadius: "8px",
              background: "rgba(124,58,237,0.2)",
              cursor: "pointer",
              fontSize: "0.9rem",
            }}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
