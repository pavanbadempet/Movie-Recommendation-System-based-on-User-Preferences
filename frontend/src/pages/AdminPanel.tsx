import React, { useState } from "react";
import { AlertTriangle, CheckCircle2, Loader2, RefreshCw, Settings } from "lucide-react";
import { apiPost } from "../api";

// ─── Types ────────────────────────────────────────────────────────────────────

type WeightsResponse = {
  status: string;
  weights: Record<string, number>;
  source: "file" | "defaults";
};

// ─── Error Banner ─────────────────────────────────────────────────────────────

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="admin-error-banner" role="alert" aria-live="assertive">
      <AlertTriangle size={16} aria-hidden="true" />
      {message}
    </div>
  );
}

// ─── Weights Table ────────────────────────────────────────────────────────────

function WeightsTable({ weights, source }: { weights: Record<string, number>; source: string }) {
  const models = ["lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion"];
  const rows = models.map((model) => ({
    model,
    weight: weights[model] ?? 0,
  }));

  return (
    <div className="weights-table-wrapper" aria-label="Ensemble model weights">
      <p className="eval-meta">
        Source: <strong>{source}</strong>
      </p>
      <table className="eval-table" aria-label="Ensemble weights">
        <caption className="visually-hidden">Ensemble model weights</caption>
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Weight</th>
            <th scope="col">Share</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ model, weight }) => (
            <tr key={model}>
              <td style={{ textTransform: "capitalize" }}>{model}</td>
              <td>{weight.toFixed(4)}</td>
              <td>
                <div className="weight-bar-wrapper" aria-label={`${(weight * 100).toFixed(1)}%`}>
                  <div
                    className="weight-bar"
                    style={{ width: `${Math.max(2, weight * 100)}%` }}
                    role="presentation"
                  />
                  <span>{(weight * 100).toFixed(1)}%</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Admin Panel Page ─────────────────────────────────────────────────────────

export function AdminPanel({ token }: { token: string | null }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<WeightsResponse | null>(null);
  const [success, setSuccess] = useState(false);

  if (!token) {
    return (
      <section className="admin-shell" aria-labelledby="admin-heading">
        <h2 id="admin-heading">Admin Panel</h2>
        <p className="dashboard-error" role="alert">Admin access required.</p>
      </section>
    );
  }

  async function reloadWeights() {
    setLoading(true);
    setError(null);
    setResult(null);
    setSuccess(false);

    try {
      const res = await apiPost<WeightsResponse>(
        "/v1/admin/reload-ensemble-weights",
        {},
        15000,
      );
      setResult(res.data);
      setSuccess(true);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "An unexpected error occurred. Please try again.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="admin-shell" aria-labelledby="admin-heading">
      <div className="dashboard-header">
        <div>
          <h2 id="admin-heading">
            <Settings size={20} aria-hidden="true" />
            Admin Panel
          </h2>
          <p className="dashboard-subtitle">
            Manage ensemble weights and system configuration.
          </p>
        </div>
      </div>

      {/* Reload Ensemble Weights */}
      <div className="dashboard-card" aria-label="Ensemble weights management">
        <h3 className="dashboard-card-title">
          <RefreshCw size={16} aria-hidden="true" />
          Ensemble Weights
        </h3>
        <p className="eval-meta">
          Reload the ensemble model weights from <code>models/ensemble_weights.json</code> without
          restarting the server.
        </p>

        <button
          className="primary-action"
          type="button"
          onClick={reloadWeights}
          disabled={loading}
          aria-busy={loading}
          aria-label="Reload ensemble weights"
          style={{ marginTop: "12px" }}
        >
          {loading ? (
            <>
              <Loader2 size={16} className="spin" aria-hidden="true" />
              Reloading…
            </>
          ) : (
            <>
              <RefreshCw size={16} aria-hidden="true" />
              Reload Ensemble Weights
            </>
          )}
        </button>

        {error && <ErrorBanner message={error} />}

        {success && result && (
          <div className="admin-success" role="status" aria-live="polite">
            <CheckCircle2 size={16} aria-hidden="true" />
            Weights reloaded successfully.
          </div>
        )}

        {result?.weights && (
          <WeightsTable weights={result.weights} source={result.source} />
        )}
      </div>
    </section>
  );
}
