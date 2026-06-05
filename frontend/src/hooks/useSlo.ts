import { useEffect, useState } from "react";
import { apiGet } from "../api";

export type SloMetrics = {
  p95_latency_ms?: number | null;
  p99_latency_ms?: number | null;
  error_rate?: number | null;
  request_rate?: number | null;
  uptime_seconds?: number | null;
  status?: string | null;
  [key: string]: unknown;
};

export type UseSloResult = {
  data: SloMetrics | null;
  loading: boolean;
  error: string | null;
  degraded: boolean;
};

/**
 * Fetches live SLO metrics from /v1/platform/slo on mount.
 *
 * Returns { data, loading, error, degraded } where:
 * - degraded=true when the endpoint is unavailable (network error or 5xx)
 * - data is null and no exception is thrown on degraded state
 *
 * Validates: Requirements 6.1, 6.4
 */
export function useSlo(): UseSloResult {
  const [data, setData] = useState<SloMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [degraded, setDegraded] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;

    async function fetchSlo() {
      setLoading(true);
      setError(null);
      setDegraded(false);

      try {
        const result = await apiGet<SloMetrics>("/v1/platform/slo", {}, 10000);
        if (!cancelled) {
          setData(result.data);
          setDegraded(false);
        }
      } catch (err) {
        if (!cancelled) {
          // Network error or 5xx — enter degraded state without throwing
          const message = err instanceof Error ? err.message : "SLO endpoint unavailable";
          setError(message);
          setData(null);
          setDegraded(true);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchSlo();

    return () => {
      cancelled = true;
    };
  }, []);

  return { data, loading, error, degraded };
}
