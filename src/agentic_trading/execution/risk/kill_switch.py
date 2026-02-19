"""Global kill switch.

The kill switch is checked before every order submission.  When active
all new orders are rejected and (optionally) open orders are cancelled.

State can be persisted in Redis for multi-process coordination or kept
purely in-memory for backtest mode.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agentic_trading.core.events import KillSwitchEvent

logger = logging.getLogger(__name__)

# Redis key used for distributed state
_REDIS_KEY = "kill_switch:active"
_REDIS_REASON_KEY = "kill_switch:reason"
_REDIS_TRIGGERED_BY_KEY = "kill_switch:triggered_by"
_REDIS_ACTIVATED_AT_KEY = "kill_switch:activated_at"


@dataclass
class _InMemoryState:
    """In-memory kill switch state for backtest or single-process mode."""

    active: bool = False
    reason: str = ""
    triggered_by: str = ""
    activated_at: float = 0.0


class KillSwitch:
    """Global trading kill switch.

    Supports two backends:

    * **Redis** (default for paper/live): state is stored in Redis so
      that multiple processes can observe the same switch.
    * **In-memory** (backtest): no external dependencies.

    Usage::

        ks = KillSwitch()                         # in-memory
        ks = KillSwitch(redis_url="redis://...")   # Redis-backed

        if await ks.is_active():
            raise KillSwitchActive("kill switch is on")

        await ks.activate(reason="drawdown limit hit", triggered_by="risk_engine")
        await ks.deactivate()

    Args:
        redis_url: Optional Redis connection URL.  When ``None`` the
            switch operates purely in-memory.
        redis_client: Optional pre-built ``redis.asyncio.Redis`` client.
            Takes precedence over *redis_url*.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self._redis: Any | None = None
        self._redis_url = redis_url
        self._mem = _InMemoryState()

        if redis_client is not None:
            self._redis = redis_client
        elif redis_url:
            # Lazily imported so the module works without redis installed
            # (e.g. in backtest mode).
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(
                    redis_url, decode_responses=True
                )
            except ImportError:
                logger.warning(
                    "redis.asyncio not installed; kill switch falling back "
                    "to in-memory mode"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def is_active(self) -> bool:
        """Return ``True`` if the kill switch is currently active.

        This method is designed to be called on the hot path (before
        every order submission) and should be as fast as possible.
        """
        if self._redis is not None:
            try:
                val = await self._redis.get(_REDIS_KEY)
                return val == "1"
            except Exception:
                logger.exception(
                    "Redis read failed for kill switch; "
                    "falling back to in-memory state"
                )
        return self._mem.active

    async def activate(
        self,
        reason: str,
        triggered_by: str = "",
    ) -> KillSwitchEvent:
        """Activate the kill switch.

        Args:
            reason: Human-readable reason for the activation.
            triggered_by: Identifier of the component that triggered it
                (e.g. ``"risk_engine"``, ``"cli"``, ``"config"``).

        Returns:
            A :class:`KillSwitchEvent` to be published on the event bus.
        """
        now = time.time()

        # Update in-memory state (always, as a fallback)
        self._mem.active = True
        self._mem.reason = reason
        self._mem.triggered_by = triggered_by
        self._mem.activated_at = now

        # Persist to Redis
        if self._redis is not None:
            try:
                async with self._redis.pipeline(transaction=True) as pipe:
                    pipe.set(_REDIS_KEY, "1")
                    pipe.set(_REDIS_REASON_KEY, reason)
                    pipe.set(_REDIS_TRIGGERED_BY_KEY, triggered_by)
                    pipe.set(_REDIS_ACTIVATED_AT_KEY, str(now))
                    await pipe.execute()
            except Exception:
                logger.exception("Redis write failed during kill switch activation")

        event = KillSwitchEvent(
            activated=True,
            reason=reason,
            triggered_by=triggered_by,
        )

        logger.critical(
            "KILL SWITCH ACTIVATED: reason=%s triggered_by=%s",
            reason,
            triggered_by,
        )
        return event

    async def deactivate(self) -> KillSwitchEvent:
        """Deactivate the kill switch.

        Returns:
            A :class:`KillSwitchEvent` indicating deactivation.
        """
        previous_reason = self._mem.reason

        # Clear in-memory state
        self._mem.active = False
        self._mem.reason = ""
        self._mem.triggered_by = ""
        self._mem.activated_at = 0.0

        # Clear Redis state
        if self._redis is not None:
            try:
                await self._redis.delete(
                    _REDIS_KEY,
                    _REDIS_REASON_KEY,
                    _REDIS_TRIGGERED_BY_KEY,
                    _REDIS_ACTIVATED_AT_KEY,
                )
            except Exception:
                logger.exception("Redis write failed during kill switch deactivation")

        event = KillSwitchEvent(
            activated=False,
            reason=f"deactivated (was: {previous_reason})" if previous_reason else "deactivated",
            triggered_by="manual",
        )

        logger.warning("KILL SWITCH DEACTIVATED")
        return event

    async def get_status(self) -> dict[str, Any]:
        """Return the full kill switch status as a dict.

        Useful for health-check endpoints and CLI inspection.
        """
        if self._redis is not None:
            try:
                active = await self._redis.get(_REDIS_KEY)
                reason = await self._redis.get(_REDIS_REASON_KEY) or ""
                triggered_by = await self._redis.get(_REDIS_TRIGGERED_BY_KEY) or ""
                activated_at = await self._redis.get(_REDIS_ACTIVATED_AT_KEY) or "0"
                return {
                    "active": active == "1",
                    "reason": reason,
                    "triggered_by": triggered_by,
                    "activated_at": float(activated_at),
                    "backend": "redis",
                }
            except Exception:
                logger.exception("Redis read failed for kill switch status")

        return {
            "active": self._mem.active,
            "reason": self._mem.reason,
            "triggered_by": self._mem.triggered_by,
            "activated_at": self._mem.activated_at,
            "backend": "memory",
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the Redis connection if one is held."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                logger.debug("Error closing Redis connection", exc_info=True)
            self._redis = None
