"""
backend/schemas.py
====================
Pydantic request/response models for the FastAPI backend.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6)
    role: str = "viewer"


class UserOut(BaseModel):
    username: str
    role: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Free-text product search query")
    category: Optional[str] = None
    top_k: int = 10


class ProductResult(BaseModel):
    product_id: str
    title: Optional[str] = None
    category: str
    brand: str
    price: float
    avg_rating: float
    num_reviews: int


class RankingRequestSchema(BaseModel):
    user_id: str
    candidate_product_ids: Optional[List[str]] = Field(
        default=None, description="If omitted, ranks over all products the user hasn't purchased."
    )
    top_k: int = 10


class RankedProduct(BaseModel):
    product_id: str
    rank_score: float
    rank_position: int
    title: Optional[str] = None
    price: Optional[float] = None
    brand: Optional[str] = None
    num_reviews: Optional[int] = None


class RankingResponse(BaseModel):
    user_id: str
    ranked_products: List[RankedProduct]


class ExplanationRequest(BaseModel):
    user_id: str
    product_id: str
    compare_to_product_id: Optional[str] = None


class ExplanationResponse(BaseModel):
    product_id: str
    product_title: Optional[str] = None
    price: Optional[float] = None
    compared_product_id: Optional[str] = None
    compared_product_title: Optional[str] = None
    compared_price: Optional[float] = None
    rank_score: float
    compared_rank_score: Optional[float] = None
    score_difference: Optional[float] = None
    explanation_text: str
    plain_language_summary: str
    confidence_score: float
    ecs_score: float
    passed_verification: bool
    top_shap_features: List[Dict]
    comparison_features: List[Dict] = Field(default_factory=list)
    graph_path: List[Dict]


class EvaluationSummary(BaseModel):
    metrics: Dict
