"""
database/models.py
====================
SQLAlchemy ORM models mirroring database/schema.sql. Works against either
SQLite (default, zero-setup) or PostgreSQL by changing `database.url` in
configs/config.yaml -- SQLAlchemy abstracts the dialect differences.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class AppUser(Base):
    """Application user account for authenticated dashboard/API access."""
    __tablename__ = "app_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), nullable=False, default="viewer")
    created_at = Column(DateTime, default=datetime.utcnow)


class ProductCache(Base):
    """Denormalized product metadata cache for fast search/lookup in the API layer."""
    __tablename__ = "products_cache"

    product_id = Column(String(32), primary_key=True)
    category = Column(String(64))
    brand = Column(String(64))
    price = Column(Float)
    avg_rating = Column(Float)
    num_reviews = Column(Integer)


class RankingRequest(Base):
    """Log of a ranking/search request made through the API."""
    __tablename__ = "ranking_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(32), nullable=False)
    query = Column(Text)
    requested_at = Column(DateTime, default=datetime.utcnow)

    explanations = relationship("ExplanationLog", back_populates="ranking_request")


class ExplanationLog(Base):
    """Audit log of every generated explanation, its confidence, and verification outcome."""
    __tablename__ = "explanation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ranking_request_id = Column(Integer, ForeignKey("ranking_requests.id"))
    product_id = Column(String(32), nullable=False)
    rank_score = Column(Float, nullable=False)
    explanation_text = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False)
    ecs_score = Column(Float)
    passed_verification = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)

    ranking_request = relationship("RankingRequest", back_populates="explanations")


class EvaluationRun(Base):
    """Stores a full evaluation report (as JSON text) for historical comparison across runs."""
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_name = Column(String(128), nullable=False)
    metrics_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
