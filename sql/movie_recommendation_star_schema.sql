-- Nova analytical data model
-- Purpose: interview-ready star schema for catalog, recommendation, and user-event analytics.

-- Dimension: movie catalog with SCD Type 2 history.
CREATE TABLE IF NOT EXISTS dim_movie_scd2 (
    movie_sk BIGINT GENERATED ALWAYS AS IDENTITY,
    movie_id BIGINT NOT NULL,
    title STRING NOT NULL,
    genres STRING,
    director STRING,
    cast STRING,
    original_language STRING,
    release_date DATE,
    vote_average DOUBLE,
    vote_count BIGINT,
    popularity DOUBLE,
    record_hash STRING NOT NULL,
    effective_start_at TIMESTAMP NOT NULL,
    effective_end_at TIMESTAMP NOT NULL,
    is_current BOOLEAN NOT NULL,
    CONSTRAINT dim_movie_scd2_pk PRIMARY KEY (movie_sk)
);

-- Fact: recommendation API impressions.
CREATE TABLE IF NOT EXISTS fact_recommendation_impression (
    impression_id STRING NOT NULL,
    request_id STRING NOT NULL,
    query_movie_id BIGINT,
    recommended_movie_id BIGINT NOT NULL,
    rank_position INT NOT NULL,
    similarity_score DOUBLE,
    rerank_score DOUBLE,
    algorithm_version STRING NOT NULL,
    served_at TIMESTAMP NOT NULL,
    CONSTRAINT fact_recommendation_impression_pk PRIMARY KEY (impression_id)
);

-- Fact: semantic search requests.
CREATE TABLE IF NOT EXISTS fact_search_request (
    request_id STRING NOT NULL,
    query_text STRING NOT NULL,
    result_count INT NOT NULL,
    latency_ms DOUBLE,
    api_status_code INT,
    searched_at TIMESTAMP NOT NULL,
    CONSTRAINT fact_search_request_pk PRIMARY KEY (request_id)
);

-- Fact: user events from Kafka or application logs.
CREATE TABLE IF NOT EXISTS fact_user_event (
    event_id STRING NOT NULL,
    user_id STRING,
    movie_id BIGINT,
    event_type STRING NOT NULL,
    rating INT,
    event_ts TIMESTAMP NOT NULL,
    ingestion_ts TIMESTAMP NOT NULL,
    source_system STRING NOT NULL,
    CONSTRAINT fact_user_event_pk PRIMARY KEY (event_id)
);

-- Query 1: top recommended movies by impressions and average rank.
SELECT
    m.title,
    COUNT(*) AS impressions,
    AVG(f.rank_position) AS avg_rank_position,
    AVG(f.similarity_score) AS avg_similarity_score
FROM fact_recommendation_impression f
JOIN dim_movie_scd2 m
    ON f.recommended_movie_id = m.movie_id
   AND m.is_current = TRUE
GROUP BY m.title
ORDER BY impressions DESC, avg_rank_position ASC
LIMIT 20;

-- Query 2: daily search latency and success rate.
SELECT
    CAST(searched_at AS DATE) AS search_date,
    COUNT(*) AS total_searches,
    AVG(latency_ms) AS avg_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
    SUM(CASE WHEN api_status_code BETWEEN 200 AND 299 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS success_rate
FROM fact_search_request
GROUP BY CAST(searched_at AS DATE)
ORDER BY search_date DESC;

-- Query 3: current genre engagement from user events.
SELECT
    m.genres,
    e.event_type,
    COUNT(*) AS event_count,
    AVG(e.rating) AS avg_rating
FROM fact_user_event e
JOIN dim_movie_scd2 m
    ON e.movie_id = m.movie_id
   AND e.event_ts >= m.effective_start_at
   AND e.event_ts < m.effective_end_at
GROUP BY m.genres, e.event_type
ORDER BY event_count DESC;

-- Query 4: movies whose catalog attributes changed most often.
SELECT
    movie_id,
    MAX(title) AS latest_known_title,
    COUNT(*) AS version_count,
    MIN(effective_start_at) AS first_seen_at,
    MAX(effective_start_at) AS last_changed_at
FROM dim_movie_scd2
GROUP BY movie_id
HAVING COUNT(*) > 1
ORDER BY version_count DESC, last_changed_at DESC;
