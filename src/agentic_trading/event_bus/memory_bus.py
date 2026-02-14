"""In-memory event bus for testing and backtesting.

No external dependencies. Handlers are called synchronously in publish order.
Supports consumer groups for compatibility with Redis Streams interface.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

from agentic_trading.core.events import BaseEvent

logger = logging.getLogger(__name__)


class MemoryEventBus:
    """In-memory event bus. Thread-safe within a single asyncio event loop."""

    def __init__(self) -> None:
        # topic → list of (group, handler)
        self._handlers: dict[
            str, list[tuple[str, Callable[[BaseEvent], Coroutine[Any, Any, None]]]]
        ] = defaultdict(list)
        self._history: list[tuple[str, BaseEvent]] = []
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def publish(self, topic: str, event: BaseEvent) -> None:
        """Publish event to all handlers subscribed to the topic."""
        self._history.append((topic, event))

        handlers = self._handlers.get(topic, [])
        for _group, handler in handlers:
            try:
                await handler(event)
            except Exception:
                logger.exception(
                    "Handler error on topic=%s event=%s",
                    topic,
                    type(event).__name__,
                )

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: Callable[[BaseEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe a handler to a topic with a consumer group name."""
        self._handlers[topic].append((group, handler))

    def get_history(self, topic: str | None = None) -> list[tuple[str, BaseEvent]]:
        """Get event history, optionally filtered by topic. For testing."""
        if topic is None:
            return list(self._history)
        return [(t, e) for t, e in self._history if t == topic]

    def clear_history(self) -> None:
        """Clear event history. For testing."""
        self._history.clear()
