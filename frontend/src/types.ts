export type Movie = {
  id: number;
  title: string;
  overview?: string | null;
  genres?: string | null;
  vote_average?: number | null;
  vote_count?: number | null;
  popularity?: number | null;
  release_date?: string | null;
  poster_path?: string | null;
  metadata_completeness?: number | null;
  content_quality_score?: number | null;
  quality_bucket?: string | null;
  recommendable?: boolean | null;
  similarity_score?: number | null;
  retrieval_stage?: string | null;
  retrieval_signals?: Record<string, unknown> | null;
  semantic_twin?: Record<string, unknown> | null;
  semantic_signals?: Record<string, unknown> | null;
  explanation_text?: string | null;
  explanation?: string[] | null;
  trailer_key?: string | null;
  runtime?: number | null;
  director?: string | null;
  cast?: string | null;
};

export type MovieTitle = {
  id: number;
  title: string;
};

export type RecommendationResponse = {
  request_id?: string | null;
  query_movie: Movie;
  recommendations: Movie[];
};

export type EventType =
  | "view"
  | "search"
  | "click"
  | "rating"
  | "recommendation_request"
  | "recommendation_impression";

export type EventPayload = {
  event_type: EventType;
  tenant_id?: string | null;
  catalog_id?: string | null;
  content_id?: string | null;
  source_content_id?: string | null;
  movie_id?: number | null;
  query_text?: string | null;
  user_id?: string | null;
  session_id?: string | null;
  rating?: number | null;
  request_id?: string | null;
  metadata?: Record<string, unknown> | null;
};

export type EventResponse = {
  status: string;
  event_id: string;
  event_path: string;
  event_store: string;
  durable: boolean;
};

export type BackendHealth = {
  status: string;
  movie_count: number;
};

export type ApiRoot = {
  status: string;
  message?: string;
  version?: string;
};

export type PlatformStatus = {
  status: string;
  tenant_id: string;
  catalog_id: string;
  movie_count: number;
  event_store?: {
    mode?: string | null;
    durable?: boolean | null;
    total_events?: number | null;
  };
  ranker?: {
    available?: boolean;
    training_mode?: string | null;
    promotion?: string | null;
  };
  capabilities?: string[];
};

export type ReadinessComponent = {
  name: string;
  status: "ok" | "degraded" | "warming" | "missing" | "failed" | "unavailable" | "not_ready" | string;
  required: boolean;
  summary: string;
  details?: Record<string, unknown>;
};

export type PlatformReadiness = {
  status: "ready" | "degraded" | "not_ready" | string;
  strict: boolean;
  tenant_id: string;
  catalog_id: string;
  generated_at?: string;
  k?: number;
  app?: {
    version?: string | null;
    commit?: string | null;
  };
  summary?: {
    component_count?: number;
    ok_count?: number;
    required_count?: number;
    failed_required_count?: number;
  };
  components?: ReadinessComponent[];
};

export type ArtifactHealth = {
  generated_at: string;
  status: "ready" | "degraded" | "unavailable" | string;
  run_id?: string | null;
  run_date?: string | null;
  model_name?: string | null;
  row_counts?: {
    movies?: number | null;
    movie_ids?: number | null;
    semantic_twins?: number | null;
  };
  checks?: Record<string, boolean | null | string | number>;
  semantic_summary?: {
    row_count?: number | null;
    avg_confidence?: number | null;
    coverage?: Record<string, number> | null;
  };
  recommendations?: string[];
  errors?: string[];
};

export type SemanticBenchmark = {
  generated_at?: string;
  status: "ok" | "needs_attention" | "unavailable" | string;
  case_count?: number;
  evaluated_case_count?: number;
  k?: number;
  reason?: string;
  metrics?: {
    good_recall_at_k?: number;
    precision_at_k?: number;
    hit_rate_at_k?: number;
    mrr_at_k?: number;
    ndcg_at_k?: number;
    bad_match_rate_at_k?: number;
    bad_case_rate_at_k?: number;
    explanation_coverage?: number;
    good_hit_count?: number;
    bad_hit_count?: number;
    stage_distribution?: Record<string, number>;
  };
};

export type BackendResult<T> = {
  data: T;
  baseUrl: string;
};
