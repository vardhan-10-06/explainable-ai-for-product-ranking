"""
backend/auth.py
=================
JWT-based authentication for the FastAPI backend. Passwords are hashed
with bcrypt (passlib); tokens are signed HS256 JWTs with a configurable
expiry.

Secret key: read from the environment variable named in
`backend.secret_key_env` (default "JWT_SECRET_KEY"). Never hardcode the
secret; if the env var is unset we generate an ephemeral one at startup
and log a loud warning, since that means tokens won't survive a restart
(fine for local dev, NOT fine for production).
"""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from common.config import get_config, get_logger
from database.db import get_session
from database.models import AppUser

logger = get_logger(__name__)

# PBKDF2 is provided by passlib itself and avoids a runtime incompatibility
# between recent bcrypt releases and passlib's legacy bcrypt adapter.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

_cfg = get_config()["backend"]
_SECRET_KEY = os.environ.get(_cfg["secret_key_env"])
if not _SECRET_KEY:
    _SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "No %s environment variable set; using an ephemeral secret key for this process only. "
        "Set %s for production deployments so tokens remain valid across restarts.",
        _cfg["secret_key_env"], _cfg["secret_key_env"],
    )
_ALGORITHM = "HS256"


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password for storage."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its stored bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str, role: str) -> str:
    """
    Create a signed JWT access token.

    Args:
        subject: The username to embed as the token subject.
        role: The user's role, embedded as a custom claim for authorization checks.

    Returns:
        The encoded JWT string.
    """
    expire = datetime.utcnow() + timedelta(minutes=_cfg["access_token_expire_minutes"])
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def authenticate_user(session: Session, username: str, password: str) -> Optional[AppUser]:
    """Look up a user by username and verify their password."""
    user = session.query(AppUser).filter(AppUser.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency: decode and validate the bearer token, returning its claims."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
        return {"username": username, "role": payload.get("role", "viewer")}
    except JWTError as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise credentials_exception from exc


def require_role(required_role: str):
    """FastAPI dependency factory: restrict an endpoint to a specific role (e.g. 'admin')."""
    def _checker(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] != required_role and current_user["role"] != "admin":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user
    return _checker
