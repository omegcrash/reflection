# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Database engine management.

Supports:
- PostgreSQL via asyncpg (production)
- SQLite via aiosqlite (development / single-node)

The active backend is determined by DATABASE_URL in settings.
"""

import logging
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

# Module-level singletons
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_settings():
    """Lazy import to avoid circular deps with settings module."""
    from ..core.settings import get_settings

    return get_settings()


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


async def init_database() -> None:
    """Create the async engine and session factory.

    Called once during application startup (lifespan).
    For SQLite, also creates tables directly from metadata
    since Alembic migrations use PostgreSQL-specific SQL.
    """
    global _engine, _session_factory

    settings = _get_settings()
    url = settings.database.url

    engine_kwargs: dict = {}

    if _is_sqlite(url):
        # SQLite: no pool, check_same_thread off
        engine_kwargs.update(
            connect_args={"check_same_thread": False},
            pool_pre_ping=False,
        )
        logger.info("Initializing SQLite database: %s", url)
    else:
        # PostgreSQL: connection pooling
        engine_kwargs.update(
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_pre_ping=True,
        )
        logger.info("Initializing PostgreSQL database")

    _engine = create_async_engine(
        url,
        echo=settings.database.echo,
        **engine_kwargs,
    )

    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # For SQLite in dev mode, create tables from ORM metadata
    # (production PostgreSQL uses Alembic migrations instead)
    if _is_sqlite(url):
        from .models import Base

        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("SQLite tables created from ORM metadata")

    logger.info("Database initialized")


async def close_database() -> None:
    """Dispose the engine and release all connections.

    Called during application shutdown.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("Database connections closed")
    _engine = None
    _session_factory = None


def get_database() -> AsyncEngine:
    """Return the active engine instance.

    Raises RuntimeError if init_database() hasn't been called.
    """
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields an async session.

    Usage in route handlers::

        @router.get("/tenants")
        async def list_tenants(session: AsyncSession = Depends(get_db_session)):
            repo = TenantRepository(session)
            ...

    The session is committed on success and rolled back on exception.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
