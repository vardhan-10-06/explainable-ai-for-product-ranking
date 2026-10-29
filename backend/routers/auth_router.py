"""
backend/routers/auth_router.py
================================
Authentication endpoints: user registration and JWT token issuance.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.auth import authenticate_user, create_access_token, hash_password
from backend.schemas import Token, UserCreate, UserOut
from common.config import get_logger
from database.db import get_session
from database.models import AppUser

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


def _db_session():
    with get_session() as session:
        yield session


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, session: Session = Depends(_db_session)) -> UserOut:
    """Register a new application user."""
    existing = session.query(AppUser).filter(AppUser.username == user_in.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    user = AppUser(username=user_in.username, hashed_password=hash_password(user_in.password), role=user_in.role)
    session.add(user)
    session.flush()
    logger.info("Registered new user: %s (role=%s)", user.username, user.role)
    return UserOut(username=user.username, role=user.role)


@router.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(_db_session)) -> Token:
    """Authenticate and issue a JWT access token (OAuth2 password flow)."""
    user = authenticate_user(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    token = create_access_token(subject=user.username, role=user.role)
    return Token(access_token=token)
