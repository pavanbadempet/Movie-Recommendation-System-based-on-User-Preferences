use axum::{routing::{post, get}, Json, Router};
use hyper::Server;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[derive(Deserialize)]
struct BreedRequest {
    population: Vec<Vec<f64>>,
    generations: Option<usize>,
}

#[derive(Serialize)]
struct BreedResponse {
    best: Vec<f64>,
    score: f64,
}

async fn breed_handler(Json(req): Json<BreedRequest>) -> Json<BreedResponse> {
    let (best, score) = engine_rs::breed(&req.population, req.generations.unwrap_or(1));
    Json(BreedResponse { best, score })
}

async fn health_handler() -> &'static str {
    "ok"
}

#[derive(Serialize)]
struct Recommendation {
    movie_id: String,
    title: String,
    score: f64,
}

async fn recommend_handler(axum::extract::Path(movie_id): axum::extract::Path<String>) -> Json<Recommendation> {
    // PoC: simple deterministic score based on id length
    let score = (movie_id.len() as f64) % 100.0;
    Json(Recommendation { movie_id: movie_id.clone(), title: format!("Movie {}", movie_id), score })
}

async fn status_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status":"ok","services":{"engine-rs":"ok"}}))
}

async fn readiness_handler() -> &'static str {
    "ready"
}

async fn search_handler() -> Json<Vec<Recommendation>> {
    // PoC: return 3 dummy movies
    let items = vec![
        Recommendation { movie_id: "1".to_string(), title: "Example 1".to_string(), score: 0.9 },
        Recommendation { movie_id: "2".to_string(), title: "Example 2".to_string(), score: 0.8 },
        Recommendation { movie_id: "3".to_string(), title: "Example 3".to_string(), score: 0.7 },
    ];
    Json(items)
}

#[derive(Deserialize)]
struct EventPayload {
    event_type: String,
    payload: serde_json::Value,
}

async fn events_handler(Json(_body): Json<EventPayload>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"accepted": true}))
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/breed", post(breed_handler))
        .route("/health", get(health_handler))
        .route("/v1/recommendations/id/:movie_id", get(recommend_handler))
        .route("/v1/platform/status", get(status_handler))
        .route("/v1/platform/readiness", get(readiness_handler))
        .route("/v1/search", get(search_handler))
        .route("/v1/events", post(events_handler));

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("Starting engine-rs on {}", addr);
    Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
