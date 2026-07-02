import { useEffect, useState } from "react";
import { getKGRecommendations, getRecommendations } from "../api";
import type { Movie } from "../types";

export type GraphNode = {
  id: string;
  label: string;
  type: "seed" | "recommendation";
  movie: Movie;
};

export type GraphEdge = {
  source: string;
  target: string;
  label: string;
};

export type GraphData = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type UseKnowledgeGraphResult = {
  graphData: GraphData | null;
  loading: boolean;
  error: string | null;
};

/**
 * Fetches knowledge-graph recommendations for a given movie ID and transforms
 * the response into a { nodes, edges } structure suitable for D3 rendering.
 *
 * Validates: Requirements 8.1
 */
export function useKnowledgeGraph(movieId: number | null): UseKnowledgeGraphResult {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (movieId === null) {
      setGraphData(null);
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    setGraphData(null);

    const loadGraphData = async () => {
      try {
        const result = await getKGRecommendations(movieId, 12);
        if (cancelled) return;
        const { query_movie, recommendations } = result.data;
        const seedId = `movie-${query_movie.id}`;

        const nodes: GraphNode[] = [
          { id: seedId, label: query_movie.title, type: "seed", movie: query_movie },
          ...recommendations.map((rec) => ({
            id: `movie-${rec.id}`,
            label: rec.title,
            type: "recommendation" as const,
            movie: rec,
          })),
        ];

        const edges: GraphEdge[] = recommendations.map((rec) => ({
          source: seedId,
          target: `movie-${rec.id}`,
          label: rec.retrieval_stage ?? "related",
        }));

        setGraphData({ nodes, edges });
      } catch (err) {
        if (cancelled) return;
        const errMsg = err instanceof Error ? err.message : "Failed to load knowledge graph";
        
        // Check if error is due to missing KG artifacts (503 / disabled in Tier3 environment)
        if (
          errMsg.includes("503") ||
          errMsg.includes("disabled") ||
          errMsg.includes("missing artifacts") ||
          errMsg.includes("Unavailable")
        ) {
          console.warn("[APEX] Knowledge Graph service disabled in this environment. Falling back to vector recommendations to populate movie connection graph.");
          try {
            const result = await getRecommendations(movieId, 12);
            if (cancelled) return;
            const { query_movie, recommendations } = result.data;
            const seedId = `movie-${query_movie.id}`;

            const nodes: GraphNode[] = [
              { id: seedId, label: query_movie.title, type: "seed", movie: query_movie },
              ...recommendations.map((rec) => ({
                id: `movie-${rec.id}`,
                label: rec.title,
                type: "recommendation" as const,
                movie: rec,
              })),
            ];

            const edges: GraphEdge[] = recommendations.map((rec) => ({
              source: seedId,
              target: `movie-${rec.id}`,
              label: rec.retrieval_stage ?? "similar",
            }));

            setGraphData({ nodes, edges });
          } catch (fallbackErr) {
            if (cancelled) return;
            setError(fallbackErr instanceof Error ? fallbackErr.message : "Failed to load movie network");
          }
        } else {
          setError(errMsg);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void loadGraphData();

    return () => {
      cancelled = true;
    };
  }, [movieId]);

  return { graphData, loading, error };
}
