"""Unified BusManager facade for the bus layer.

Composes the legacy topic-routed event bus (MemoryEventBus /
RedisStreamsBus), the domain-event bus (InMemoryEventBus), the topic
schema registry, and bus observability into a single entry point.

Usage::

    from agentic_trading.bus.manager import BusManager

    mgr = BusManager.from_config(mode=Mode.PAPER, redis_url="redis://...")
    await mgr.start()

    # Topic-routed publish/subscribe (legacy)
    await mgr.publish("strategy.signal", signal_event)
    await mgr.subscribe("feature.vector", "runner", handler)

    # Domain-event publish/subscribe (canonical)
    await mgr.publish_domain(signal_created_event)
    mgr.subscribe_domain(SignalCreated, handler)

    # Observability
    metrics = mgr.get_metrics()
    errors = mgr.get_error_counts()
    dead = mgr.get_dead_letters()

    await mgr.stop()
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class BusManager:
    """Unified facade for the bus layer.

    Owns both the legacy topic-routed event bus and the new domain-event
    bus, provides lifecycle management, unified observability, and
    convenience helpers for publish/subscribe on both buses.

    Parameters
    ----------
    legacy_bus:
        The topic-routed event bus (MemoryEventBus or RedisStreamsBus).
    domain_bus:
        The type-routed domain event bus (``None`` to disable).
    """

    def __init__(
        self,
        legacy_bus: Any,
        domain_bus: Any | None = None,
    ) -> None:
        self._legacy_bus = legacy_bus
        self._domain_bus = domain_bus

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        mode: Any | None = None,
        redis_url: str = "redis://localhost:6379/0",
        on_handler_error: Callable[
            [str, str, str, Exception], None
        ] | None = None,
        enable_domain_bus: bool = False,
        enforce_ownership: bool = True,
        event_store: Any | None = None,
    ) -> BusManager:
        """Build a fully wired BusManager from configuration.

        Parameters
        ----------
        mode:
            Trading mode (``Mode.BACKTEST``, ``Mode.PAPER``, or
            ``Mode.LIVE``).  Determines the legacy bus implementation.
            When ``None``, defaults to ``Mode.BACKTEST``.
        redis_url:
            Redis connection URL for paper/live mode.
        on_handler_error:
            Optional callback ``(topic, group, msg_id, exc)`` invoked
            when a handler raises.  Applied to the legacy bus.
        enable_domain_bus:
            Whether to create the domain-event bus alongside the
            legacy bus (default ``False``).
        enforce_ownership:
            Whether the domain bus enforces write ownership
            (default ``True``).  Ignored if ``enable_domain_bus``
            is ``False``.
        event_store:
            Optional ``IEventStore`` for domain-event persistence.
            Ignored if ``enable_domain_bus`` is ``False``.

        Returns
        -------
        BusManager
        """
        from agentic_trading.core.enums import Mode

        resolved_mode = mode if mode is not None else Mode.BACKTEST

        # --- Legacy bus ---
        from agentic_trading.bus.bus import create_event_bus

        legacy_bus = create_event_bus(
            resolved_mode,
            redis_url=redis_url,
            on_handler_error=on_handler_error,
        )

        # --- Domain bus (opt-in) ---
        domain_bus = None
        if enable_domain_bus:
            from agentic_trading.infrastructure.event_bus import (
                InMemoryEventBus as DomainEventBus,
            )

            domain_bus = DomainEventBus(
                enforce_ownership=enforce_ownership,
                event_store=event_store,
            )

        return cls(
            legacy_bus=legacy_bus,
            domain_bus=domain_bus,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all buses."""
        await self._legacy_bus.start()
        if self._domain_bus is not None:
            await self._domain_bus.start()
        logger.info("BusManager started")

    async def stop(self) -> None:
        """Stop all buses."""
        if self._domain_bus is not None:
            await self._domain_bus.stop()
        await self._legacy_bus.stop()
        logger.info("BusManager stopped")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def legacy_bus(self):
        """The topic-routed event bus (MemoryEventBus or RedisStreamsBus)."""
        return self._legacy_bus

    @property
    def domain_bus(self):
        """The type-routed domain event bus (``None`` if disabled)."""
        return self._domain_bus

    @property
    def is_running(self) -> bool:
        """Whether the legacy bus is running."""
        return getattr(self._legacy_bus, "_running", False)

    # ------------------------------------------------------------------
    # Legacy bus — publish / subscribe
    # ------------------------------------------------------------------

    async def publish(self, topic: str, event) -> None:
        """Publish an event to a topic on the legacy bus.

        Delegates to ``legacy_bus.publish()``.
        """
        await self._legacy_bus.publish(topic, event)

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: Callable[..., Coroutine],
    ) -> None:
        """Subscribe to a topic on the legacy bus.

        Delegates to ``legacy_bus.subscribe()``.
        """
        await self._legacy_bus.subscribe(topic, group, handler)

    # ------------------------------------------------------------------
    # Domain bus — publish / subscribe
    # ------------------------------------------------------------------

    async def publish_domain(self, event) -> None:
        """Publish a domain event to the domain bus.

        Raises ``RuntimeError`` if the domain bus is not enabled.
        """
        if self._domain_bus is None:
            raise RuntimeError("Domain bus is not enabled")
        await self._domain_bus.publish(event)

    def subscribe_domain(self, event_type, handler) -> None:
        """Subscribe to a domain event type on the domain bus.

        Raises ``RuntimeError`` if the domain bus is not enabled.
        """
        if self._domain_bus is None:
            raise RuntimeError("Domain bus is not enabled")
        self._domain_bus.subscribe(event_type, handler)

    # ------------------------------------------------------------------
    # Observability — legacy bus
    # ------------------------------------------------------------------

    def get_error_counts(self) -> dict[str, int]:
        """Error counts from the legacy bus keyed by ``topic/group``."""
        return self._legacy_bus.get_error_counts()

    def get_dead_letters(self) -> list:
        """Dead-letter entries from the legacy bus."""
        return list(self._legacy_bus.dead_letters)

    def clear_dead_letters(self) -> list:
        """Drain and return all dead letters from the legacy bus."""
        return self._legacy_bus.clear_dead_letters()

    @property
    def messages_processed(self) -> int:
        """Total messages processed by the legacy bus."""
        return self._legacy_bus.messages_processed

    # ------------------------------------------------------------------
    # Observability — domain bus
    # ------------------------------------------------------------------

    def get_domain_error_counts(self) -> dict[str, int]:
        """Error counts from the domain bus (empty if disabled)."""
        if self._domain_bus is None:
            return {}
        return self._domain_bus.get_error_counts()

    def get_domain_dead_letters(self) -> list:
        """Dead-letter entries from the domain bus (empty if disabled)."""
        if self._domain_bus is None:
            return []
        return list(self._domain_bus.dead_letters)

    @property
    def domain_messages_processed(self) -> int:
        """Total messages processed by the domain bus (0 if disabled)."""
        if self._domain_bus is None:
            return 0
        return self._domain_bus.messages_processed

    # ------------------------------------------------------------------
    # Unified metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Combined metrics from all buses.

        Returns a dict with legacy and domain bus stats.
        """
        metrics: dict[str, Any] = {
            "legacy_messages_processed": self.messages_processed,
            "legacy_error_counts": self.get_error_counts(),
            "legacy_dead_letter_count": len(self.get_dead_letters()),
        }
        if self._domain_bus is not None:
            metrics["domain_messages_processed"] = self.domain_messages_processed
            metrics["domain_error_counts"] = self.get_domain_error_counts()
            metrics["domain_dead_letter_count"] = len(
                self.get_domain_dead_letters()
            )
        return metrics

    # ------------------------------------------------------------------
    # Schema registry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_topic_for_event(event) -> str | None:
        """Look up the topic for a legacy event.

        Delegates to ``schemas.get_topic_for_event()``.
        """
        from agentic_trading.bus.schemas import get_topic_for_event

        return get_topic_for_event(event)

    @staticmethod
    def get_event_class(event_type_name: str):
        """Look up a legacy event class by name.

        Delegates to ``schemas.get_event_class()``.
        """
        from agentic_trading.bus.schemas import get_event_class

        return get_event_class(event_type_name)

    @staticmethod
    def list_topics() -> list[str]:
        """Return all registered topic names."""
        from agentic_trading.bus.schemas import TOPIC_SCHEMAS

        return sorted(TOPIC_SCHEMAS.keys())
