"""In-memory event bus for testing and backtesting.

No external dependencies. Handlers are called synchronously in publish order.
Supports consumer groups for compatibility with Redis Streams interface.

Improvements:
- Optional error callback for handler failures
- Per-topic/group error counters
- Dead-letter tracking for failed messages
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from agentic_trading.core.events import BaseEvent

logger = logging.getLogger(__name__)


@dataclass
class MemoryDeadLetter:
    """Record of a handler failure in the memory bus."""

    topic: str
    group: str
    event_type: str
    error: str
    timestamp: float = field(default_factory=time.monotonic)


class MemoryEventBus:
    """In-memory event bus. Thread-safe within a single asyncio event loop.

    Improvements over the original:
    - Optional ``on_handler_error`` callback for external metrics/alerting.
    - Per-topic/group error counters accessible via ``get_error_counts()``.
    - Dead-letter tracking for handler failures.
    """

    def __init__(
        self,
        on_handler_error: Callable[
            [str, str, str, Exception], None
        ] | None = None,
    ) -> None:
        # topic → list of (group, handler)
        self._handlers: dict[
            str, list[tuple[str, Callable[[BaseEvent], Coroutine[Any, Any, None]]]]
        ] = defaultdict(list)
        self._history: list[tuple[str, BaseEvent]] = []
        self._running = False
        self._on_handler_error = on_handler_error

        # Observability
        self._error_counts: dict[str, int] = defaultdict(int)
        self._dead_letters: list[MemoryDeadLetter] = []
        self._messages_processed: int = 0

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def publish(self, topic: str, event: BaseEvent) -> None:
        """Publish event to all handlers subscribed to the topic."""
        self._history.append((topic, event))

        handlers = self._handlers.get(topic, [])
        for group, handler in handlers:
            try:
                await handler(event)
                self._messages_processed += 1
            except Exception as exc:
                error_key = f"{topic}/{group}"
                self._error_counts[error_key] += 1
                self._dead_letters.append(
                    MemoryDeadLetter(
                        topic=topic,
                        group=group,
                        event_type=type(event).__name__,
                        error=str(exc),
                    )
                )
                logger.exception(
                    "Handler error on topic=%s group=%s event=%s",
                    topic,
                    group,
                    type(event).__name__,
                )

                # Fire external error callback
                if self._on_handler_error is not None:
                    try:
                        self._on_handler_error(
                            topic, group, event.event_id, exc,
                        )
                    except Exception:
                        logger.warning(
                            "on_handler_error callback failed",
                            exc_info=True,
                        )

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: Callable[[BaseEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe a handler to a topic with a consumer group name."""
        self._handlers[topic].append((group, handler))

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_error_counts(self) -> dict[str, int]:
        """Return per-topic/group error counts."""
        return dict(self._error_counts)

    @property
    def dead_letters(self) -> list[MemoryDeadLetter]:
        """Access the dead-letter list (read-only snapshot)."""
        return list(self._dead_letters)

    @property
    def messages_processed(self) -> int:
        """Total messages successfully processed."""
        return self._messages_processed

    def clear_dead_letters(self) -> list[MemoryDeadLetter]:
        """Drain the dead-letter list and return all entries."""
        drained = self._dead_letters[:]
        self._dead_letters.clear()
        return drained

    # ------------------------------------------------------------------
    # Testing helpers
    # ------------------------------------------------------------------

    def get_history(self, topic: str | None = None) -> list[tuple[str, BaseEvent]]:
        """Get event history, optionally filtered by topic. For testing."""
        if topic is None:
            return list(self._history)
        return [(t, e) for t, e in self._history if t == topic]

    def clear_history(self) -> None:
        """Clear event history. For testing."""
        self._history.clear()
