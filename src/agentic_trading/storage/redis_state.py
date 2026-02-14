"""Redis-backed state management for the trading platform.

Provides fast, in-memory read/write access to:
  - Current positions (per symbol)
  - Current balances (per currency)
  - Open orders (per symbol)
  - Kill switch flag
  - Order deduplication keys (idempotency via SET with TTL)

All keys are namespaced under a configurable prefix (default ``trading:``)
so multiple environments can share a single Redis instance.

Uses ``redis.asyncio`` for non-blocking I/O.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any

import redis.asyncio as aioredis

from agentic_trading.core.enums import Exchange, MarginMode, PositionSide
from agentic_trading.core.models import Balance, Order, Position

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom JSON serialisation for Decimal / Enum types
# ---------------------------------------------------------------------------


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that converts :class:`Decimal` to string."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


def _serialize(obj: dict[str, Any]) -> str:
    """Serialize a dict to a JSON string using :class:`_DecimalEncoder`."""
    return json.dumps(obj, cls=_DecimalEncoder)


def _deserialize(raw: str | bytes | None) -> dict[str, Any] | None:
    """Deserialize a JSON string back to a dict."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------

def _pos_key(prefix: str, symbol: str) -> str:
    return f"{prefix}position:{symbol}"


def _bal_key(prefix: str, currency: str) -> str:
    return f"{prefix}balance:{currency}"


def _open_orders_key(prefix: str, symbol: str) -> str:
    return f"{prefix}open_orders:{symbol}"


def _kill_switch_key(prefix: str) -> str:
    return f"{prefix}kill_switch"


def _dedupe_key(prefix: str, client_order_id: str) -> str:
    return f"{prefix}dedupe:{client_order_id}"


# ---------------------------------------------------------------------------
# RedisStateStore
# ---------------------------------------------------------------------------


class RedisStateStore:
    """Async Redis state store for trading platform runtime state.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
        prefix: Key namespace prefix. Defaults to ``"trading:"``.
        dedupe_ttl_seconds: TTL for order deduplication keys.
            Defaults to 3600 (1 hour).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "trading:",
        dedupe_ttl_seconds: int = 3600,
    ) -> None:
        self._url = redis_url
        self._prefix = prefix
        self._dedupe_ttl = dedupe_ttl_seconds
        self._redis: aioredis.Redis | None = None

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        """Establish the Redis connection pool."""
        if self._redis is not None:
            return
        self._redis = aioredis.from_url(
            self._url,
            decode_responses=False,  # We handle decoding ourselves
            max_connections=20,
        )
        # Verify connectivity
        await self._redis.ping()
        logger.info("Redis connected: %s", self._url.split("@")[-1])

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis connection closed.")

    @property
    def redis(self) -> aioredis.Redis:
        """Return the underlying Redis client, raising if not connected."""
        if self._redis is None:
            raise RuntimeError(
                "RedisStateStore not connected. Call connect() first."
            )
        return self._redis

    # -- positions -----------------------------------------------------------

    async def set_position(self, position: Position) -> None:
        """Store a position snapshot for a symbol.

        Args:
            position: The current position state to store.
        """
        key = _pos_key(self._prefix, position.symbol)
        data = position.model_dump(mode="json")
        await self.redis.set(key, _serialize(data))
        logger.debug("Set position for %s", position.symbol)

    async def get_position(self, symbol: str) -> Position | None:
        """Retrieve the current position for a symbol.

        Args:
            symbol: Unified symbol string.

        Returns:
            :class:`Position` if set, otherwise ``None``.
        """
        key = _pos_key(self._prefix, symbol)
        raw = await self.redis.get(key)
        data = _deserialize(raw)
        if data is None:
            return None
        return Position(**data)

    async def delete_position(self, symbol: str) -> None:
        """Remove a position from the store.

        Args:
            symbol: Unified symbol string.
        """
        key = _pos_key(self._prefix, symbol)
        await self.redis.delete(key)
        logger.debug("Deleted position for %s", symbol)

    async def get_all_positions(self) -> list[Position]:
        """Retrieve all currently stored positions.

        Returns:
            List of :class:`Position` objects.
        """
        pattern = _pos_key(self._prefix, "*")
        positions: list[Position] = []
        async for key in self.redis.scan_iter(match=pattern, count=100):
            raw = await self.redis.get(key)
            data = _deserialize(raw)
            if data is not None:
                positions.append(Position(**data))
        return positions

    # -- balances ------------------------------------------------------------

    async def set_balance(self, balance: Balance) -> None:
        """Store a balance for a currency.

        Args:
            balance: The current balance state.
        """
        key = _bal_key(self._prefix, balance.currency)
        data = balance.model_dump(mode="json")
        await self.redis.set(key, _serialize(data))
        logger.debug("Set balance for %s", balance.currency)

    async def get_balance(self, currency: str) -> Balance | None:
        """Retrieve the current balance for a currency.

        Args:
            currency: Currency code (e.g. ``"USDT"``).

        Returns:
            :class:`Balance` if set, otherwise ``None``.
        """
        key = _bal_key(self._prefix, currency)
        raw = await self.redis.get(key)
        data = _deserialize(raw)
        if data is None:
            return None
        return Balance(**data)

    async def get_all_balances(self) -> list[Balance]:
        """Retrieve all currently stored balances.

        Returns:
            List of :class:`Balance` objects.
        """
        pattern = _bal_key(self._prefix, "*")
        balances: list[Balance] = []
        async for key in self.redis.scan_iter(match=pattern, count=100):
            raw = await self.redis.get(key)
            data = _deserialize(raw)
            if data is not None:
                balances.append(Balance(**data))
        return balances

    # -- open orders ---------------------------------------------------------

    async def set_open_orders(self, symbol: str, orders: list[Order]) -> None:
        """Replace the current open orders for a symbol.

        The entire list is stored atomically. To update a single order,
        read the list, modify it, and write it back.

        Args:
            symbol: Unified symbol string.
            orders: List of open (non-terminal) orders.
        """
        key = _open_orders_key(self._prefix, symbol)
        data = [o.model_dump(mode="json") for o in orders]
        await self.redis.set(key, _serialize(data))
        logger.debug("Set %d open orders for %s", len(orders), symbol)

    async def get_open_orders(self, symbol: str) -> list[Order]:
        """Retrieve the current open orders for a symbol.

        Args:
            symbol: Unified symbol string.

        Returns:
            List of :class:`Order` objects.
        """
        key = _open_orders_key(self._prefix, symbol)
        raw = await self.redis.get(key)
        if raw is None:
            return []
        data = _deserialize(raw)
        if data is None or not isinstance(data, list):
            return []
        return [Order(**item) for item in data]

    async def clear_open_orders(self, symbol: str) -> None:
        """Remove all open orders for a symbol.

        Args:
            symbol: Unified symbol string.
        """
        key = _open_orders_key(self._prefix, symbol)
        await self.redis.delete(key)
        logger.debug("Cleared open orders for %s", symbol)

    # -- kill switch ---------------------------------------------------------

    async def set_kill_switch(self, active: bool, reason: str = "") -> None:
        """Set or clear the kill switch flag.

        When active, all order submission should be halted.

        Args:
            active: ``True`` to engage the kill switch, ``False`` to clear it.
            reason: Human-readable reason for the state change.
        """
        key = _kill_switch_key(self._prefix)
        data = {"active": active, "reason": reason}
        await self.redis.set(key, _serialize(data))
        level = logging.WARNING if active else logging.INFO
        logger.log(level, "Kill switch %s: %s", "ACTIVATED" if active else "cleared", reason)

    async def get_kill_switch(self) -> tuple[bool, str]:
        """Check the kill switch state.

        Returns:
            Tuple of ``(is_active, reason)``. Defaults to ``(False, "")``
            if the key does not exist.
        """
        key = _kill_switch_key(self._prefix)
        raw = await self.redis.get(key)
        data = _deserialize(raw)
        if data is None:
            return False, ""
        return bool(data.get("active", False)), str(data.get("reason", ""))

    # -- order deduplication -------------------------------------------------

    async def check_dedupe(self, client_order_id: str) -> bool:
        """Check whether a deduplication key already exists.

        Args:
            client_order_id: The client-side order identifier used as
                a dedupe key.

        Returns:
            ``True`` if the key exists (meaning this order has already
            been submitted), ``False`` otherwise.
        """
        key = _dedupe_key(self._prefix, client_order_id)
        return bool(await self.redis.exists(key))

    async def add_dedupe(self, client_order_id: str) -> bool:
        """Add a deduplication key with TTL.

        Uses ``SET NX`` (set-if-not-exists) for atomicity.

        Args:
            client_order_id: The client-side order identifier.

        Returns:
            ``True`` if the key was newly set (order is fresh),
            ``False`` if it already existed (duplicate).
        """
        key = _dedupe_key(self._prefix, client_order_id)
        result = await self.redis.set(key, "1", nx=True, ex=self._dedupe_ttl)
        is_new = result is not None
        if not is_new:
            logger.warning("Duplicate order detected: %s", client_order_id)
        return is_new

    async def remove_dedupe(self, client_order_id: str) -> None:
        """Remove a deduplication key.

        Useful when an order is cancelled and the slot should be freed.

        Args:
            client_order_id: The client-side order identifier.
        """
        key = _dedupe_key(self._prefix, client_order_id)
        await self.redis.delete(key)

    # -- utilities -----------------------------------------------------------

    async def flush_all_state(self) -> None:
        """Delete ALL keys under the configured prefix.

        .. warning::
            This is destructive and intended only for tests or full state
            resets. Use with extreme caution.
        """
        pattern = f"{self._prefix}*"
        deleted = 0
        async for key in self.redis.scan_iter(match=pattern, count=200):
            await self.redis.delete(key)
            deleted += 1
        logger.warning("Flushed %d keys matching %s", deleted, pattern)

    async def health_check(self) -> bool:
        """Ping Redis to verify connectivity.

        Returns:
            ``True`` if Redis is reachable, ``False`` otherwise.
        """
        try:
            return bool(await self.redis.ping())
        except Exception:
            logger.exception("Redis health check failed.")
            return False
