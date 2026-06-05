import { useEffect, useState } from "react";
import { apiGet } from "../api";

export type HardwareProfile = {
  gpu_available: boolean;
  ram_gb: number;
  cpu_cores: number;
};

export type HealthData = {
  status: string;
  serving_tier: string | null;
  hardware_profile: HardwareProfile | null;
  tier_selection_reason: string | null;
};

export type UseHealthResult = {
  data: HealthData | null;
  loading: boolean;
  error: string | null;
};

/**
 * Fetches /health on mount and returns the serving tier, hardware profile,
 * and tier selection reason. Handles network errors gracefully.
 *
 * Validates: Requirements 6.2, 6.3
 */
export function useHealth(): UseHealthResult {
  const [data, setData] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchHealth() {
      setLoading(true);
      setError(null);

      try {
        const result = await apiGet<HealthData>("/health", {}, 10000);
        if (!cancelled) {
          setData(result.data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to fetch health data");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchHealth();

    return () => {
      cancelled = true;
    };
  }, []);

  return { data, loading, error };
}
