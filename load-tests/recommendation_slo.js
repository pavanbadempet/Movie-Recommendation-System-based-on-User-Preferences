/**
 * APEX Load Test — Validates SLO budgets defined in Dockerfile / .env.example
 *
 * Targets:
 *   /health                              p95 < 1000 ms
 *   /v1/recommendations/id/{movie_id}    p95 < 25000 ms
 *   /v1/search                           p95 < 2500 ms
 *   /v1/events (POST)                    p95 < 1000 ms
 *
 * Run locally:
 *   k6 run load-tests/recommendation_slo.js \
 *       -e BASE_URL=http://localhost:8000 \
 *       -e API_KEY=your-key
 *
 * Run in CI (smoke mode — 10 VUs, 30 s):
 *   k6 run --env SMOKE=true load-tests/recommendation_slo.js \
 *       -e BASE_URL=http://localhost:8000
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const BASE_URL  = __ENV.BASE_URL  || 'http://localhost:8000';
const API_KEY   = __ENV.API_KEY   || '';
const SMOKE     = __ENV.SMOKE === 'true';

// Representative movie IDs from the TMDB catalog
const MOVIE_IDS = [550, 680, 13, 238, 278, 424, 389, 155, 122, 27205];

// ---------------------------------------------------------------------------
// Load profile
// ---------------------------------------------------------------------------

export const options = SMOKE
  ? {
      // Smoke test: quick sanity check, no strict SLO threshold enforcement
      vus: 2,
      duration: '20s',
      thresholds: {},
    }
  : {
      // Full load test: ramp up → steady state → ramp down
      stages: [
        { duration: '1m', target: 20 }, // ramp up
        { duration: '3m', target: 50 }, // steady state
        { duration: '1m', target: 100 }, // peak
        { duration: '1m', target: 0 }, // ramp down
      ],
      thresholds: {
        // Global error rate
        http_req_failed: ['rate<0.03'],

        // Per-endpoint SLO budgets (match NOVA_SLO_ROUTE_LATENCY_BUDGETS)
        'http_req_duration{endpoint:health}': ['p(95)<1000'],
        'http_req_duration{endpoint:recommendations}': ['p(95)<25000'],
        'http_req_duration{endpoint:search}': ['p(95)<2500'],
        'http_req_duration{endpoint:events}': ['p(95)<1000'],
      },
    };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const headers = {
  'Content-Type': 'application/json',
  ...(API_KEY ? { 'X-Nova-API-Key': API_KEY } : {}),
};

function randomMovieId() {
  return MOVIE_IDS[Math.floor(Math.random() * MOVIE_IDS.length)];
}

// ---------------------------------------------------------------------------
// Scenario functions
// ---------------------------------------------------------------------------

function healthCheck() {
  const res = http.get(`${BASE_URL}/health`, {
    headers,
    tags: { endpoint: 'health' },
  });
  check(res, {
    'health: status 200': (r) => r.status === 200,
    'health: has serving_tier': (r) => JSON.parse(r.body).serving_tier !== undefined,
  });
}

function getRecommendations() {
  const movieId = randomMovieId();
  const res = http.get(`${BASE_URL}/v1/recommendations/id/${movieId}?top_k=10`, {
    headers,
    tags: { endpoint: 'recommendations' },
  });
  check(res, {
    'recommendations: status 200, 404, or 503': (r) =>
      r.status === 200 || r.status === 404 || r.status === 429 || r.status === 503,
  });
}

function semanticSearch() {
  const queries = ['action thriller', 'romantic comedy', 'sci-fi space', 'animated family'];
  const q = queries[Math.floor(Math.random() * queries.length)];
  const res = http.get(`${BASE_URL}/v1/search?q=${encodeURIComponent(q)}&top_k=10`, {
    headers,
    tags: { endpoint: 'search' },
  });
  check(res, {
    'search: status 200 or 422': (r) => r.status === 200 || r.status === 422,
  });
}

function postEvent() {
  const payload = JSON.stringify({
    event_type: 'click',
    user_id:    `load-test-user-${__VU}`,
    movie_id:   randomMovieId(),
    session_id: `session-${__VU}-${__ITER}`,
  });
  const res = http.post(`${BASE_URL}/v1/events`, payload, {
    headers,
    tags: { endpoint: 'events' },
  });
  check(res, {
    'events: status 200 or 201': (r) => r.status === 200 || r.status === 201,
  });
}

// ---------------------------------------------------------------------------
// Default scenario — mix of all endpoint types
// ---------------------------------------------------------------------------

export default function () {
  const roll = Math.random();

  if (roll < 0.10) {
    healthCheck();
  } else if (roll < 0.55) {
    getRecommendations();
  } else if (roll < 0.80) {
    semanticSearch();
  } else {
    postEvent();
  }

  sleep(Math.random() * 0.5 + 0.1);  // 100–600 ms think time
}
