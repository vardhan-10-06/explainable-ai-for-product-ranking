"""
backend/routers/search_router.py
==================================
Product search endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from typing import List

from backend.auth import get_current_user
from backend.schemas import ProductResult, SearchRequest
from backend.service import RankingService, get_ranking_service

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=List[ProductResult])
def search_products(
    payload: SearchRequest,
    service: RankingService = Depends(get_ranking_service),
    current_user: dict = Depends(get_current_user),
) -> List[ProductResult]:
    """Search the product catalog by free-text query and optional category filter."""
    try:
        results = service.search_products(payload.query, payload.category, payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [
        ProductResult(
            product_id=r["product_id"], title=r.get("title"), category=r["category"], brand=r["brand"],
            price=float(r.get("price", 0.0)), avg_rating=float(r.get("avg_rating", 0.0)),
            num_reviews=int(r.get("num_reviews", 0)),
        )
        for r in results
    ]
