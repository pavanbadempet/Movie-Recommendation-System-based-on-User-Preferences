import { useEffect, useState } from "react";
import { getKGRecommendations } from "../api";
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

    getKGRecommendations(movieId, 12)
      .then((result) => {
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
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load knowledge graph");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [movieId]);

  return { graphData, loading, error };
}
