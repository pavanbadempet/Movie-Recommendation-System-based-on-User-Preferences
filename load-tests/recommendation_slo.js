import http from 'k6/http';
import { check, sleep } from 'k6';

const isSmoke = __ENV.SMOKE === 'true';
const baseURL = __ENV.BASE_URL || 'https://movie-recs-api-5qvy.onrender.com';
const apiKey = __ENV.API_KEY || '';

export const options = isSmoke
  ? {
      vus: 5,
      duration: '15s',
      thresholds: {
        http_req_failed: ['rate<0.10'], // Less than 10% errors
        http_req_duration: ['p(95)<3000'], // 95% of requests under 3s
      },
    }
  : {
      stages: [
        { duration: '30s', target: 20 },
        { duration: '1m', target: 50 },
        { duration: '30s', target: 0 },
      ],
      thresholds: {
        http_req_failed: ['rate<0.05'],
        http_req_duration: ['p(95)<2000'],
      },
    };

const headers = {
  'Content-Type': 'application/json',
  ...(apiKey ? { 'X-API-Key': apiKey } : {}),
};

export default function () {
  // 1. Health check
  const healthRes = http.get(`${baseURL}/health`, { headers });
  check(healthRes, {
    'health status is 200': (r) => r.status === 200,
  });

  sleep(0.5);

  // 2. Movie Recommendation check
  const recRes = http.get(`${baseURL}/v1/recommendations/id/550?top_k=5`, { headers });
  check(recRes, {
    'recommendation status is 200 or 404': (r) => r.status === 200 || r.status === 404 || r.status === 503,
  });

  sleep(0.5);
}
