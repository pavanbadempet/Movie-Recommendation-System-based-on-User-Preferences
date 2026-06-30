# APEX Quickstart

> Your first recommendation in 5 minutes.

---

## 1. Get your API key

Sign up at `https://your-apex-domain.com/signup`.
Your API key is shown **once** at registration. Store it in a password manager or your
secrets vault — it cannot be recovered, only rotated.

---

## 2. Upload your catalog

Your catalog is a CSV file with at minimum these columns:

| Column | Required | Description |
|---|---|---|
| `item_id` | ✅ | Unique identifier for each item |
| `title` | ✅ | Display name |
| `description` | ✅ | Text used for semantic search and embeddings |
| `genres` | Optional | Comma-separated genre tags |
| `poster_url` | Optional | Image URL for visual similarity features |

```bash
curl -X POST "https://your-apex-api.onrender.com/v1/catalog/upload" \
  -H "X-Nova-API-Key: YOUR_KEY" \
  -F "file=@catalog.csv"
```

Response:
```json
{
  "status": "ok",
  "rows_ingested": 10432,
  "catalog_id": "my-catalog-v1"
}
```

---

## 3. Get recommendations

```bash
# Recommendations for item ID 550
curl "https://your-apex-api.onrender.com/v1/recommendations/id/550?n=10" \
  -H "X-Nova-API-Key: YOUR_KEY"
```

```json
{
  "movie_id": 550,
  "title": "Fight Club",
  "recommendations": [
    { "movie_id": 807, "title": "Se7en", "score": 0.94 },
    { "movie_id": 680, "title": "Pulp Fiction", "score": 0.91 },
    ...
  ]
}
```

---

## 4. Add LLM explanations

Append `?explain=true` to any recommendation endpoint to get a personalized
natural-language explanation for each result (powered by GPT-4o via OpenRouter):

```bash
curl "https://your-apex-api.onrender.com/v1/recommendations/id/550?explain=true" \
  -H "X-Nova-API-Key: YOUR_KEY"
```

```json
{
  "recommendations": [
    {
      "title": "Se7en",
      "score": 0.94,
      "explanation": "Because you watched Fight Club, you'll love Se7en — both are
        dark psychological thrillers with twist endings directed by David Fincher."
    }
  ]
}
```

---

## 5. Semantic search

```bash
# Handles typos, abstract concepts, and mood-based queries
curl "https://your-apex-api.onrender.com/v1/search/ai?q=mind+bending+heist+film&limit=5" \
  -H "X-Nova-API-Key: YOUR_KEY"
```

---

## 6. Log user events (for personalization)

Sending user events improves recommendation quality over time via the online
learning loop:

```bash
# Log a rating event
curl -X POST "https://your-apex-api.onrender.com/v1/events" \
  -H "X-Nova-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_42", "item_id": 550, "event_type": "rating", "rating": 4.5}'

# Log a click event
curl -X POST "https://your-apex-api.onrender.com/v1/events" \
  -H "X-Nova-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_42", "item_id": 807, "event_type": "click"}'
```

---

## SDK snippets

### Python

```python
import httpx

BASE = "https://your-apex-api.onrender.com"
HEADERS = {"X-Nova-API-Key": "YOUR_KEY"}

# Get recommendations
resp = httpx.get(f"{BASE}/v1/recommendations/id/550", headers=HEADERS, params={"n": 10})
recs = resp.json()["recommendations"]

# Semantic search
resp = httpx.get(f"{BASE}/v1/search/ai", headers=HEADERS, params={"q": "dark thriller"})
results = resp.json()
```

### JavaScript / TypeScript

```typescript
const BASE = "https://your-apex-api.onrender.com";
const HEADERS = { "X-Nova-API-Key": "YOUR_KEY" };

// Get recommendations
const resp = await fetch(`${BASE}/v1/recommendations/id/550?n=10`, { headers: HEADERS });
const { recommendations } = await resp.json();

// Log a click event
await fetch(`${BASE}/v1/events`, {
  method: "POST",
  headers: { ...HEADERS, "Content-Type": "application/json" },
  body: JSON.stringify({ user_id: "user_42", item_id: 550, event_type: "click" }),
});
```

---

## Serving tiers

APEX auto-detects your deployment hardware and selects the best serving mode:

| Tier | Hardware | Models | Latency |
|---|---|---|---|
| **Tier 1** | GPU + ≥16 GB RAM | Full 6-model ensemble + RL | 50–200 ms |
| **Tier 2** | CPU + ≥8 GB RAM | ONNX-quantized ensemble | 200–800 ms |
| **Tier 3** | < 8 GB RAM | FAISS + TF-IDF only | 800–2000 ms |

Check your current tier:

```bash
curl "https://your-apex-api.onrender.com/health" | python -m json.tool
```

---

## Next steps

- [Full API Reference](API_REFERENCE.md) — all 52 endpoints with request/response schemas
- [Architecture Overview](ARCHITECTURE.md) — how the 4-layer stack works
- [Model Cards](MODEL_CARDS.md) — training data, metrics, and limitations for each model
- [Enterprise Guide](ENTERPRISE_GUIDE.md) — Databricks, Snowflake, AWS MWAA deployment
- [Interactive Docs](https://your-apex-api.onrender.com/docs) — live Swagger UI
