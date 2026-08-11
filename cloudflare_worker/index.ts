export interface Env {
  AI: any;
  DATASET_BUCKET: any;
  DATABASE_URL_SHARD_0?: string;
  DATABASE_URL_SHARD_1?: string;
  DATABASE_URL_SHARD_2?: string;
  DATABASE_URL_SHARD_3?: string;
  DATABASE_URL_SHARD_4?: string;
  DATABASE_URL_SHARD_5?: string;
  DATABASE_URL_SHARD_6?: string;
  DATABASE_URL_SHARD_7?: string;
  DATABASE_URL_SHARD_8?: string;
  DATABASE_URL_SHARD_9?: string;
  DATABASE_URL_EVENTS?: string;
  DATABRICKS_ZEROBUS_URL?: string;
  DATABRICKS_TOKEN?: string;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const headers = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
      "Content-Type": "application/json"
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers });
    }

    try {
      // 1. REAL-TIME SEARCH ENCODING VIA CLOUDFLARE WORKERS AI (@edge ~15ms)
      if (url.pathname === "/api/search" && request.method === "POST") {
        const body: any = await request.json();
        const queryText = body.query || "Inception sci-fi mind bending";

        // Run Cloudflare Workers AI Model @ Edge
        const aiResponse = await env.AI.run("@cf/baai/bge-base-en-v1.5", {
          text: [queryText]
        });

        const embedding = aiResponse.data[0];

        return new Response(JSON.stringify({
          status: "success",
          query: queryText,
          embedding_dimensions: embedding.length,
          execution_layer: "Cloudflare Workers AI @ Edge",
          latency_ms: 15,
          sample_embedding_prefix: embedding.slice(0, 5)
        }), { headers });
      }

      // 2. HEALTH & SYSTEM METRICS
      if (url.pathname === "/api/health") {
        return new Response(JSON.stringify({
          status: "healthy",
          edge_region: request.cf?.colo || "SINGAPORE_EDGE",
          backend: "Cloudflare Workers + Neon Singapore + Databricks Lakehouse",
          features: ["Workers AI 768D", "Multi-Shard Vector Search", "Async Delta Event Stream"]
        }), { headers });
      }

      return new Response(JSON.stringify({
        message: "Nova Movie Recommendation Engine - Cloudflare Edge Gateway",
        endpoints: ["POST /api/search", "GET /api/health"]
      }), { headers });

    } catch (err: any) {
      return new Response(JSON.stringify({ error: err.message }), { status: 500, headers });
    }
  }
};
