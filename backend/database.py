"""SQLite / SQLAlchemy session (FYP: durable history + auth)."""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from sqlalchemy.pool import StaticPool


def _database_url() -> str:
    return os.environ.get("CARDIOSENSE_DATABASE_URL", "sqlite:///./cardiosense.db")


def create_db_engine():
    url = _database_url()
    if url.startswith("sqlite"):
        opts: dict = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url:
            opts["poolclass"] = StaticPool
        return create_engine(url, **opts)
    return create_engine(url)


engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
