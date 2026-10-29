"""
backend/routers/ranking_router.py
====================================
Ranking endpoint: returns top-K ranked products for a given user.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.schemas import RankedProduct, RankingRequestSchema, RankingResponse
from backend.service import RankingService, get_ranking_service
from database.db import get_session
from database.models import RankingRequest as RankingRequestLog

router = APIRouter(prefix="/rank", tags=["ranking"])


def _db_session():
    with get_session() as session:
        yield session


@router.post("/", response_model=RankingResponse)
def rank_products(
    payload: RankingRequestSchema,
    service: RankingService = Depends(get_ranking_service),
    current_user: dict = Depends(get_current_user),
    session: Session = Depends(_db_session),
) -> RankingResponse:
    """Rank candidate products for a user using the trained hybrid ensemble."""
    try:
        ranked = service.rank_for_user(payload.user_id, payload.candidate_product_ids, payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    log_entry = RankingRequestLog(user_id=payload.user_id, query=str(payload.candidate_product_ids))
    session.add(log_entry)

    return RankingResponse(
        user_id=payload.user_id,
        ranked_products=[RankedProduct(**item) for item in ranked],
    )
