"""SQLAlchemy async engine pool and session management.

Provides a factory for creating async engines backed by asyncpg,
an async context manager for scoped sessions, and lifecycle helpers
for schema creation and graceful shutdown.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from .models import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton (set via ``init_engine``)
# ---------------------------------------------------------------------------
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def create_engine(
    url: str,
    *,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    pool_recycle: int = 1800,
    echo: bool = False,
    use_null_pool: bool = False,
) -> AsyncEngine:
    """Create and return a new SQLAlchemy :class:`AsyncEngine`.

    Args:
        url: Database connection URL. Must use the ``postgresql+asyncpg://``
            scheme.
        pool_size: Number of persistent connections to keep in the pool.
        max_overflow: Maximum additional connections beyond *pool_size*.
        pool_timeout: Seconds to wait for a connection from the pool before
            raising a timeout error.
        pool_recycle: Seconds after which a connection is recycled to avoid
            stale TCP connections.
        echo: If ``True``, log all emitted SQL statements.
        use_null_pool: If ``True``, disable connection pooling entirely.
            Useful in short-lived processes (tests, one-off scripts).

    Returns:
        A configured :class:`AsyncEngine` instance.
    """
    pool_kwargs: dict = {}
    if use_null_pool:
        pool_kwargs["poolclass"] = NullPool
    else:
        pool_kwargs.update(
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
        )

    engine = create_async_engine(
        url,
        echo=echo,
        **pool_kwargs,
    )
    logger.info("Created async engine for %s (pool_size=%s)", url.split("@")[-1], pool_size)
    return engine


async def init_engine(
    url: str,
    *,
    pool_size: int = 5,
    max_overflow: int = 10,
    echo: bool = False,
    create_tables: bool = False,
) -> AsyncEngine:
    """Initialise the module-level engine and session factory.

    This is the primary entry-point at application startup. Subsequent calls
    to :func:`get_session` will use the engine created here.

    Args:
        url: Database connection URL.
        pool_size: Pool size forwarded to :func:`create_engine`.
        max_overflow: Overflow forwarded to :func:`create_engine`.
        echo: SQL echo flag.
        create_tables: If ``True``, run ``CREATE TABLE IF NOT EXISTS`` for
            all ORM models on startup (useful for dev/test).

    Returns:
        The initialised :class:`AsyncEngine`.
    """
    global _engine, _session_factory  # noqa: PLW0603

    _engine = create_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=echo,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    if create_tables:
        await create_all(_engine)

    return _engine


async def create_all(engine: AsyncEngine | None = None) -> None:
    """Create all tables defined in the ORM metadata.

    Args:
        engine: Engine to use. Falls back to the module-level engine.

    Raises:
        RuntimeError: If no engine is available.
    """
    eng = engine or _engine
    if eng is None:
        raise RuntimeError(
            "No engine available. Call init_engine() first or pass an engine."
        )

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created / verified.")


async def dispose() -> None:
    """Dispose of the module-level engine and release all pooled connections."""
    global _engine, _session_factory  # noqa: PLW0603

    if _engine is not None:
        await _engine.dispose()
        logger.info("Engine disposed.")
        _engine = None
        _session_factory = None


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async session scoped to the caller's block.

    Usage::

        async with get_session() as session:
            result = await session.execute(select(OrderRecord))
            ...

    The session is committed on successful exit and rolled back on
    exception. It is always closed afterwards.

    Raises:
        RuntimeError: If :func:`init_engine` has not been called.
    """
    if _session_factory is None:
        raise RuntimeError(
            "Session factory not initialised. Call init_engine() first."
        )

    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def get_engine() -> AsyncEngine:
    """Return the module-level engine.

    Raises:
        RuntimeError: If :func:`init_engine` has not been called.
    """
    if _engine is None:
        raise RuntimeError("Engine not initialised. Call init_engine() first.")
    return _engine
