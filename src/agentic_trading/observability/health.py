"""Health check endpoint.

Reports health status of all system components.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agentic_trading.core.events import SystemHealth

logger = logging.getLogger(__name__)


class HealthChecker:
    """Checks health of system components."""

    def __init__(self) -> None:
        self._checks: dict[str, Any] = {}

    def register_check(
        self, component: str, check_fn: Any
    ) -> None:
        """Register a health check function for a component.

        check_fn should be async and return (healthy: bool, message: str).
        """
        self._checks[component] = check_fn

    async def check_all(self) -> list[SystemHealth]:
        """Run all health checks and return results."""
        results = []

        for component, check_fn in self._checks.items():
            start = time.monotonic()
            try:
                healthy, message = await check_fn()
                latency = (time.monotonic() - start) * 1000
                results.append(
                    SystemHealth(
                        component=component,
                        healthy=healthy,
                        message=message,
                        latency_ms=latency,
                    )
                )
            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                results.append(
                    SystemHealth(
                        component=component,
                        healthy=False,
                        message=f"Check failed: {e}",
                        latency_ms=latency,
                    )
                )

        return results

    async def is_healthy(self) -> bool:
        """Quick check: are all components healthy?"""
        results = await self.check_all()
        return all(r.healthy for r in results)


async def check_redis(redis_url: str) -> tuple[bool, str]:
    """Health check for Redis connection."""
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(redis_url)
        await r.ping()
        await r.aclose()
        return True, "Redis connected"
    except Exception as e:
        return False, f"Redis error: {e}"


async def check_postgres(postgres_url: str) -> tuple[bool, str]:
    """Health check for Postgres connection."""
    try:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(postgres_url)
        async with engine.connect() as conn:
            await conn.execute(
                __import__("sqlalchemy").text("SELECT 1")
            )
        await engine.dispose()
        return True, "Postgres connected"
    except Exception as e:
        return False, f"Postgres error: {e}"
