# API Documentation

Base URL: `http://localhost:8000` (or your deployed host). Interactive OpenAPI docs
are auto-served at `/docs` (Swagger UI) and `/redoc`.

All endpoints except `/health`, `/auth/register`, and `/auth/token` require a bearer
token: `Authorization: Bearer <access_token>`.

## Auth

### `POST /auth/register`
Register a new user.
```json
{ "username": "demo", "password": "demo1234", "role": "viewer" }
```
-> `201 { "username": "demo", "role": "viewer" }`

### `POST /auth/token`
OAuth2 password flow (form-encoded, not JSON): `username`, `password`.
-> `200 { "access_token": "...", "token_type": "bearer" }`

## Search

### `POST /search/`
```json
{ "query": "Electronics", "category": null, "top_k": 10 }
```
-> `200 [ { "product_id": "P000012", "category": "Electronics", "brand": "Brand_A", "price": 45.2, "avg_rating": 4.3, "num_reviews": 87 }, ... ]`

## Ranking

### `POST /rank/`
```json
{ "user_id": "U000001", "candidate_product_ids": null, "top_k": 10 }
```
-> `200 { "user_id": "U000001", "ranked_products": [ { "product_id": "P000045", "rank_score": 0.912, "rank_position": 1 }, ... ] }`

## Explainability

### `POST /explain/`
```json
{ "user_id": "U000001", "product_id": "P000045", "compare_to_product_id": null }
```
-> `200`
```json
{
  "product_id": "P000045",
  "explanation_text": "This product was ranked highly mainly because of: price scaled, review avg rating. ...",
  "confidence_score": 0.74,
  "ecs_score": 0.68,
  "passed_verification": true,
  "top_shap_features": [ { "feature": "price_scaled", "shap_value": 0.42, "direction": "increases_rank" }, ... ],
  "graph_path": [ { "from": "User:U000001", "to": "Product:P000045", "relation": "PURCHASED" } ]
}
```
Pass `compare_to_product_id` to instead get a pairwise "why A over B" explanation
(RankSHAP-style comparison).

## Knowledge graph

### `GET /graph/path?user_id=U000001&product_id=P000045`
-> `200 [ { "from": "User:U000001", "to": "Product:P000045", "relation": "PURCHASED" } ]`

## Evaluation

### `GET /evaluation/`
Returns the most recently generated `results/evaluation_report.json`
(run `python -m evaluation.evaluate` to produce one).

## System

### `GET /health`
-> `200 { "status": "ok" }`
