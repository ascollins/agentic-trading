"""Append-only event store for replay, audit, and state reconstruction.

Design invariants
-----------------
1.  ``append()`` is **idempotent** on ``event.event_id`` — appending
    the same event twice is a silent no-op.
2.  ``read()`` returns events in **append order** (monotonically
    increasing sequence number).
3.  ``replay()`` yields events lazily for memory-efficient reprocessing.
4.  The store is **append-only** — events can never be deleted or
    modified.  ``clear()`` exists only for testing.

This module provides:

*  ``IEventStore`` — the protocol.
*  ``InMemoryEventStore`` — simple list-backed implementation for
   testing, backtesting, and local development.
*  ``JsonFileEventStore`` — append-to-JSONL-file implementation for
   durable local persistence (CI / paper trading).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from agentic_trading.domain.events import DomainEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON helpers (Decimal / datetime safe)
# ---------------------------------------------------------------------------

class _EventEncoder(json.JSONEncoder):
    """Handles Decimal and datetime serialization."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def _event_to_dict(event: DomainEvent) -> dict[str, Any]:
    """Serialize a frozen dataclass to a JSON-safe dict."""
    from dataclasses import asdict
    d = asdict(event)
    d["__event_type__"] = type(event).__qualname__
    return d


def _event_from_dict(
    d: dict[str, Any],
    registry: dict[str, type[DomainEvent]],
) -> DomainEvent | None:
    """Deserialize a dict back into a DomainEvent subclass.

    Returns ``None`` if the event type is unrecognized (forward compat).
    """
    type_name = d.pop("__event_type__", None)
    if type_name is None or type_name not in registry:
        return None
    cls = registry[type_name]

    # Restore Decimal fields
    import dataclasses
    field_types: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        field_types[f.name] = f.type

    restored: dict[str, Any] = {}
    for k, v in d.items():
        if k in field_types:
            ft = field_types[k]
            # Handle Decimal and Decimal | None
            if ft is Decimal or (
                isinstance(ft, str) and "Decimal" in ft
            ):
                if v is not None:
                    restored[k] = Decimal(str(v))
                else:
                    restored[k] = None
            elif ft is datetime or (
                isinstance(ft, str) and "datetime" in ft
            ):
                if isinstance(v, str):
                    restored[k] = datetime.fromisoformat(v)
                else:
                    restored[k] = v
            else:
                # tuples come back as lists from JSON
                if isinstance(v, list):
                    restored[k] = tuple(
                        tuple(item) if isinstance(item, list) else item
                        for item in v
                    )
                else:
                    restored[k] = v
        else:
            restored[k] = v
    return cls(**restored)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class IEventStore(Protocol):
    """Append-only event log for replay and audit."""

    async def append(self, event: DomainEvent) -> None:
        """Persist an event.  Idempotent on ``event.event_id``."""
        ...

    async def read(
        self,
        event_type: type[DomainEvent] | None = None,
        correlation_id: str | None = None,
        after_sequence: int | None = None,
        limit: int = 10_000,
    ) -> list[DomainEvent]:
        """Read events in append order, with optional filters."""
        ...

    async def replay(
        self,
        event_type: type[DomainEvent] | None = None,
        from_timestamp: datetime | None = None,
    ) -> AsyncIterator[DomainEvent]:
        """Yield events lazily for state reconstruction."""
        ...

    async def get_by_correlation(self, correlation_id: str) -> list[DomainEvent]:
        """Return all events sharing a ``correlation_id``."""
        ...


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------

class InMemoryEventStore:
    """List-backed event store.  No persistence across restarts.

    Good for: unit tests, backtest mode, local development.
    """

    def __init__(self) -> None:
        self._events: list[DomainEvent] = []
        self._seen_ids: set[str] = set()

    async def append(self, event: DomainEvent) -> None:
        """Append *event*.  No-op if ``event_id`` already stored."""
        if event.event_id in self._seen_ids:
            return
        self._seen_ids.add(event.event_id)
        self._events.append(event)

    async def read(
        self,
        event_type: type[DomainEvent] | None = None,
        correlation_id: str | None = None,
        after_sequence: int | None = None,
        limit: int = 10_000,
    ) -> list[DomainEvent]:
        """Read events in append order with optional filters."""
        start = after_sequence if after_sequence is not None else 0
        out: list[DomainEvent] = []
        for event in self._events[start:]:
            if event_type is not None and type(event) is not event_type:
                continue
            if correlation_id is not None and event.correlation_id != correlation_id:
                continue
            out.append(event)
            if len(out) >= limit:
                break
        return out

    async def replay(
        self,
        event_type: type[DomainEvent] | None = None,
        from_timestamp: datetime | None = None,
    ) -> AsyncIterator[DomainEvent]:
        """Yield stored events lazily."""
        for event in self._events:
            if event_type is not None and type(event) is not event_type:
                continue
            if from_timestamp is not None and event.timestamp < from_timestamp:
                continue
            yield event

    async def get_by_correlation(self, correlation_id: str) -> list[DomainEvent]:
        return [
            e for e in self._events
            if e.correlation_id == correlation_id
        ]

    # -- Testing helpers ---------------------------------------------------

    def clear(self) -> None:
        """Remove all events.  Testing only."""
        self._events.clear()
        self._seen_ids.clear()

    def __len__(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# JSON-Lines file implementation
# ---------------------------------------------------------------------------

class JsonFileEventStore:
    """Append-only JSONL file store.  Durable across restarts.

    Good for: paper trading, CI golden tests, local persistence.

    Each line is a JSON object with an ``__event_type__`` discriminator.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._seen_ids: set[str] = set()
        self._registry: dict[str, type[DomainEvent]] = self._build_registry()

        # Load existing event IDs for idempotency
        if self._path.exists():
            self._load_seen_ids()

    @staticmethod
    def _build_registry() -> dict[str, type[DomainEvent]]:
        """Build name → class lookup from all known event types."""
        from agentic_trading.domain.events import ALL_DOMAIN_EVENTS
        return {cls.__qualname__: cls for cls in ALL_DOMAIN_EVENTS}

    def _load_seen_ids(self) -> None:
        """Scan existing file to populate the dedup set."""
        try:
            with self._path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        eid = d.get("event_id")
                        if eid:
                            self._seen_ids.add(eid)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    async def append(self, event: DomainEvent) -> None:
        if event.event_id in self._seen_ids:
            return
        self._seen_ids.add(event.event_id)
        d = _event_to_dict(event)
        line = json.dumps(d, cls=_EventEncoder)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(line + "\n")

    async def read(
        self,
        event_type: type[DomainEvent] | None = None,
        correlation_id: str | None = None,
        after_sequence: int | None = None,
        limit: int = 10_000,
    ) -> list[DomainEvent]:
        out: list[DomainEvent] = []
        seq = 0
        if not self._path.exists():
            return out
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if after_sequence is not None and seq <= after_sequence:
                    seq += 1
                    continue
                seq += 1
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = _event_from_dict(d, self._registry)
                if event is None:
                    continue
                if event_type is not None and type(event) is not event_type:
                    continue
                if correlation_id is not None and event.correlation_id != correlation_id:
                    continue
                out.append(event)
                if len(out) >= limit:
                    break
        return out

    async def replay(
        self,
        event_type: type[DomainEvent] | None = None,
        from_timestamp: datetime | None = None,
    ) -> AsyncIterator[DomainEvent]:
        if not self._path.exists():
            return
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = _event_from_dict(d, self._registry)
                if event is None:
                    continue
                if event_type is not None and type(event) is not event_type:
                    continue
                if from_timestamp is not None and event.timestamp < from_timestamp:
                    continue
                yield event

    async def get_by_correlation(self, correlation_id: str) -> list[DomainEvent]:
        return await self.read(correlation_id=correlation_id)

    def __len__(self) -> int:
        return len(self._seen_ids)
