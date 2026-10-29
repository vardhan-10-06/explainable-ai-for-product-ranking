"""
backend/routers/explain_router.py
====================================
Explanation endpoint: returns a verified natural-language explanation
(with ECS score) for why a product was ranked for a user.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.auth import get_current_user
from backend.schemas import ExplanationRequest, ExplanationResponse
from backend.service import RankingService, get_ranking_service

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.post("/", response_model=ExplanationResponse)
def explain_ranking(
    payload: ExplanationRequest,
    service: RankingService = Depends(get_ranking_service),
    current_user: dict = Depends(get_current_user),
) -> ExplanationResponse:
    """Generate and verify an explanation for a single ranking decision (optionally a pairwise comparison)."""
    try:
        result = service.explain(payload.user_id, payload.product_id, payload.compare_to_product_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ExplanationResponse(**result)
