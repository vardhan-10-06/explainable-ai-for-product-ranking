"""
database/db.py
================
SQLAlchemy engine/session factory. Reads `database.url` from config, so
switching between SQLite (default) and PostgreSQL is a one-line config
change:

    database:
      url: "sqlite:///./database/app.db"
      # or: "postgresql://user:password@host:5432/dbname"
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from common.config import get_config, get_logger
from database.models import Base

logger = get_logger(__name__)

_engine = None
_SessionLocal = None


def get_engine():
    """Lazily create (and cache) the SQLAlchemy engine from config."""
    global _engine
    if _engine is None:
        url = get_config()["database"]["url"]
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
        logger.info("Database engine created for %s", url.split("://")[0])
    return _engine


def init_db() -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=get_engine())
    logger.info("Database tables ensured (create_all).")


def get_session_factory() -> sessionmaker:
    """Return (and cache) the sessionmaker bound to the configured engine."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


@contextmanager
def get_session() -> Iterator[Session]:
    """Context-managed DB session, used both by FastAPI dependencies and scripts."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
