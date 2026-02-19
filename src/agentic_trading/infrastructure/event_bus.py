"""Event bus abstraction and in-memory implementation.

Design goals
------------
1.  **Type-routed dispatching** — subscribers register for a concrete
    ``DomainEvent`` subclass.  When an event is published the bus routes
    it to every handler whose registered type matches ``type(event)``.
2.  **Write-ownership enforcement** — if ``enforce_ownership=True``,
    ``publish()`` verifies that ``event.source`` matches the value in
    ``WRITE_OWNERSHIP`` for that event type.  Violations raise
    ``WriteOwnershipError``.
3.  **Event store integration** — if an ``IEventStore`` is provided,
    every successfully published event is auto-appended (idempotently)
    to the store, giving a single canonical event stream for replay.
4.  **Coexistence** — the existing ``IEventBus`` / ``MemoryEventBus``
    (topic-string based, Pydantic ``BaseEvent``) is untouched.  This new
    bus operates alongside it and will gradually take over.

This module provides:

*  ``INewEventBus``  — the protocol (interface).
*  ``InMemoryEventBus`` — deterministic implementation for backtest and
   tests.
*  ``WriteOwnershipError`` — raised on ownership violations.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from agentic_trading.domain.events import DomainEvent, WRITE_OWNERSHIP
from agentic_trading.infrastructure.event_store import IEventStore

logger = logging.getLogger(__name__)

# Type alias for async event handlers.
EventHandler = Callable[[DomainEvent], Awaitable[None]]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class WriteOwnershipError(Exception):
    """Raised when an agent publishes an event it does not own."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class INewEventBus(Protocol):
    """Publish/subscribe bus for canonical ``DomainEvent`` types.

    Unlike the legacy ``IEventBus`` (topic-string based), this bus routes
    by **event type**.  Subscribers register for a concrete class and the
    bus dispatches accordingly.
    """

    async def publish(self, event: DomainEvent) -> None:
        """Publish *event* to all matching subscribers.

        If write-ownership is enforced, raises ``WriteOwnershipError``
        when ``event.source`` doesn't match the registered owner.
        """
        ...

    def subscribe(
        self,
        event_type: type[DomainEvent],
        handler: EventHandler,
    ) -> None:
        """Register *handler* for events of exactly *event_type*."""
        ...

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    # -- Observability -----------------------------------------------------

    def get_history(
        self,
        event_type: type[DomainEvent] | None = None,
    ) -> list[DomainEvent]:
        """Return published events, optionally filtered by type."""
        ...

    def get_error_counts(self) -> dict[str, int]:
        """Return ``{event_type_name: error_count}``."""
        ...

    @property
    def dead_letters(self) -> list[tuple[DomainEvent, str]]:
        """Events whose handlers failed, with the error message."""
        ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------

class InMemoryEventBus:
    """Deterministic, in-process event bus.

    Parameters
    ----------
    enforce_ownership
        When ``True`` (default), ``publish()`` rejects events whose
        ``source`` doesn't match ``WRITE_OWNERSHIP[type(event)]``.
    event_store
        Optional ``IEventStore``.  When provided, every successfully
        published event is auto-appended to the store (idempotently).
    """

    def __init__(
        self,
        *,
        enforce_ownership: bool = True,
        event_store: IEventStore | None = None,
    ) -> None:
        self._handlers: dict[
            type[DomainEvent], list[EventHandler]
        ] = defaultdict(list)
        self._history: list[DomainEvent] = []
        self._error_counts: dict[str, int] = defaultdict(int)
        self._dead_letters: list[tuple[DomainEvent, str]] = []
        self._messages_processed: int = 0
        self._running = False
        self._enforce_ownership = enforce_ownership
        self._event_store = event_store

    # -- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    # -- Core API ----------------------------------------------------------

    async def publish(self, event: DomainEvent) -> None:
        """Publish *event* to all subscribed handlers.

        Raises
        ------
        WriteOwnershipError
            If ownership enforcement is on and ``event.source`` is wrong.
        """
        event_cls = type(event)

        # --- ownership gate ------------------------------------------------
        if self._enforce_ownership and event_cls in WRITE_OWNERSHIP:
            expected = WRITE_OWNERSHIP[event_cls]
            if event.source != expected:
                raise WriteOwnershipError(
                    f"{event_cls.__name__} must be published by "
                    f"source={expected!r}, got source={event.source!r}"
                )

        self._history.append(event)

        # --- event store persistence (best-effort) -------------------------
        if self._event_store is not None:
            try:
                await self._event_store.append(event)
            except Exception:
                logger.exception(
                    "Event store append failed for %s",
                    event_cls.__name__,
                )

        for handler in self._handlers.get(event_cls, []):
            try:
                await handler(event)
                self._messages_processed += 1
            except Exception as exc:
                key = event_cls.__name__
                self._error_counts[key] += 1
                self._dead_letters.append((event, str(exc)))
                logger.exception(
                    "Handler error on %s: %s", key, exc,
                )

    def subscribe(
        self,
        event_type: type[DomainEvent],
        handler: EventHandler,
    ) -> None:
        """Register *handler* for *event_type*."""
        self._handlers[event_type].append(handler)

    # -- Observability -----------------------------------------------------

    def get_history(
        self,
        event_type: type[DomainEvent] | None = None,
    ) -> list[DomainEvent]:
        """Return published events, optionally filtered."""
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if type(e) is event_type]

    def clear_history(self) -> None:
        """Clear the event history (testing helper)."""
        self._history.clear()

    def get_error_counts(self) -> dict[str, int]:
        return dict(self._error_counts)

    @property
    def dead_letters(self) -> list[tuple[DomainEvent, str]]:
        return list(self._dead_letters)

    def clear_dead_letters(self) -> list[tuple[DomainEvent, str]]:
        """Drain and return dead letters."""
        drained = self._dead_letters[:]
        self._dead_letters.clear()
        return drained

    @property
    def messages_processed(self) -> int:
        return self._messages_processed

    @property
    def event_store(self) -> IEventStore | None:
        """The attached event store, if any."""
        return self._event_store
