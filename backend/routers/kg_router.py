"""
backend/routers/kg_router.py
==============================
Knowledge-graph retrieval endpoint: returns the explanatory path between
a user and a product for visualization on the dashboard.
"""
from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, Depends

from backend.auth import get_current_user
from backend.service import RankingService, get_ranking_service

router = APIRouter(prefix="/graph", tags=["knowledge_graph"])


@router.get("/path", response_model=List[Dict])
def get_graph_path(
    user_id: str,
    product_id: str,
    service: RankingService = Depends(get_ranking_service),
    current_user: dict = Depends(get_current_user),
) -> List[Dict]:
    """Return the shortest knowledge-graph path between a user and a product."""
    service.ensure_loaded()
    return service.kg_builder.explain_path(user_id, product_id)
