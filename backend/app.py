"""
backend/app.py
================
Main FastAPI application entry point. Wires together auth, search,
ranking, explainability, evaluation, and knowledge-graph routers.

Run with:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import auth_router, evaluation_router, explain_router, kg_router, ranking_router, search_router
from backend.service import get_ranking_service
from common.config import get_config, get_logger
from database.db import init_db

logger = get_logger(__name__)

app = FastAPI(
    title="Explainable AI Product Ranking API",
    description="Hybrid ML + Knowledge Graph + LLM explainability backend for product ranking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(search_router.router)
app.include_router(ranking_router.router)
app.include_router(explain_router.router)
app.include_router(evaluation_router.router)
app.include_router(kg_router.router)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize the database schema on application startup."""
    init_db()
    # Load the trained ensemble and feature data before accepting requests.
    # This moves the one-time startup cost out of the user's first action.
    get_ranking_service().ensure_loaded()
    logger.info("API startup complete. Config project name: %s", get_config()["project"]["name"])


@app.get("/health", tags=["system"])
def health_check() -> dict:
    """Simple liveness/readiness probe."""
    return {"status": "ok"}
