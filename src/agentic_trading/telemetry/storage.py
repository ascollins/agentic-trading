"""Spine event storage backends.

``ISpineStorage`` is the protocol.  Two implementations ship:

* ``MemorySpineStorage`` -- for backtest mode and unit tests.
* ``PostgresSpineStorage`` -- for paper / live modes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from agentic_trading.telemetry.models import SpineEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ISpineStorage(Protocol):
    """Append-only storage for spine events."""

    async def write_batch(self, events: list[SpineEvent]) -> None:
        """Write a batch of spine events to storage."""
        ...

    async def query_by_trace(
        self,
        trace_id: str,
        *,
        tenant_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Return all events for a given trace_id."""
        ...

    async def query_by_span(
        self,
        span_id: str,
        *,
        tenant_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Return all events for a given span_id."""
        ...

    async def query_by_time_range(
        self,
        start: datetime,
        end: datetime,
        *,
        tenant_id: str = "default",
        event_type: str | None = None,
        component: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return events within a time range, optionally filtered."""
        ...


# ---------------------------------------------------------------------------
# MemorySpineStorage  (backtest + tests)
# ---------------------------------------------------------------------------


class MemorySpineStorage:
    """In-memory implementation -- no persistence, no dependencies."""

    def __init__(self) -> None:
        self._events: list[SpineEvent] = []

    # -- write --------------------------------------------------------------

    async def write_batch(self, events: list[SpineEvent]) -> None:
        self._events.extend(events)

    # -- queries ------------------------------------------------------------

    async def query_by_trace(
        self,
        trace_id: str,
        *,
        tenant_id: str = "default",
    ) -> list[dict[str, Any]]:
        return [
            e.model_dump(mode="json")
            for e in self._events
            if e.trace_id == trace_id and e.tenant_id == tenant_id
        ]

    async def query_by_span(
        self,
        span_id: str,
        *,
        tenant_id: str = "default",
    ) -> list[dict[str, Any]]:
        return [
            e.model_dump(mode="json")
            for e in self._events
            if e.span_id == span_id and e.tenant_id == tenant_id
        ]

    async def query_by_time_range(
        self,
        start: datetime,
        end: datetime,
        *,
        tenant_id: str = "default",
        event_type: str | None = None,
        component: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for e in self._events:
            if e.tenant_id != tenant_id:
                continue
            if e.timestamp < start or e.timestamp > end:
                continue
            if event_type is not None and e.event_type.value != event_type:
                continue
            if component is not None and e.component != component:
                continue
            results.append(e.model_dump(mode="json"))
            if len(results) >= limit:
                break
        return results

    # -- helpers for tests --------------------------------------------------

    @property
    def events(self) -> list[SpineEvent]:
        """Direct access to stored events (for assertions)."""
        return self._events

    def clear(self) -> None:
        """Remove all stored events."""
        self._events.clear()


# ---------------------------------------------------------------------------
# PostgresSpineStorage  (paper + live)
# ---------------------------------------------------------------------------


class PostgresSpineStorage:
    """PostgreSQL implementation using parameterized batch inserts.

    Requires an async SQLAlchemy ``AsyncSession`` factory.
    """

    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    async def write_batch(self, events: list[SpineEvent]) -> None:
        if not events:
            return

        from sqlalchemy import text

        sql = text(
            "INSERT INTO spine_events "
            "(event_id, trace_id, span_id, causation_id, tenant_id, "
            " event_type, component, actor, timestamp, schema_version, "
            " input_hash, output_hash, latency_ms, error, payload) "
            "VALUES "
            "(:event_id, :trace_id, :span_id, :causation_id, :tenant_id, "
            " :event_type, :component, :actor, :timestamp, :schema_version, "
            " :input_hash, :output_hash, :latency_ms, :error, :payload) "
            "ON CONFLICT (event_id) DO NOTHING"
        )

        rows = []
        for e in events:
            import json as _json

            rows.append(
                {
                    "event_id": e.event_id,
                    "trace_id": e.trace_id,
                    "span_id": e.span_id,
                    "causation_id": e.causation_id,
                    "tenant_id": e.tenant_id,
                    "event_type": e.event_type.value,
                    "component": e.component,
                    "actor": e.actor,
                    "timestamp": e.timestamp,
                    "schema_version": e.schema_version,
                    "input_hash": e.input_hash,
                    "output_hash": e.output_hash,
                    "latency_ms": e.latency_ms,
                    "error": e.error or "",
                    "payload": _json.dumps(
                        e.payload, default=str, sort_keys=True,
                    ),
                }
            )

        async with self._session_factory() as session:
            await session.execute(sql, rows)
            await session.commit()

    async def query_by_trace(
        self,
        trace_id: str,
        *,
        tenant_id: str = "default",
    ) -> list[dict[str, Any]]:
        from sqlalchemy import text

        sql = text(
            "SELECT * FROM spine_events "
            "WHERE tenant_id = :tenant_id AND trace_id = :trace_id "
            "ORDER BY timestamp"
        )
        async with self._session_factory() as session:
            result = await session.execute(
                sql, {"tenant_id": tenant_id, "trace_id": trace_id},
            )
            return [dict(row._mapping) for row in result]

    async def query_by_span(
        self,
        span_id: str,
        *,
        tenant_id: str = "default",
    ) -> list[dict[str, Any]]:
        from sqlalchemy import text

        sql = text(
            "SELECT * FROM spine_events "
            "WHERE tenant_id = :tenant_id AND span_id = :span_id "
            "ORDER BY timestamp"
        )
        async with self._session_factory() as session:
            result = await session.execute(
                sql, {"tenant_id": tenant_id, "span_id": span_id},
            )
            return [dict(row._mapping) for row in result]

    async def query_by_time_range(
        self,
        start: datetime,
        end: datetime,
        *,
        tenant_id: str = "default",
        event_type: str | None = None,
        component: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        from sqlalchemy import text

        conditions = [
            "tenant_id = :tenant_id",
            "timestamp >= :start",
            "timestamp <= :end",
        ]
        params: dict[str, Any] = {
            "tenant_id": tenant_id,
            "start": start,
            "end": end,
            "limit": limit,
        }

        if event_type is not None:
            conditions.append("event_type = :event_type")
            params["event_type"] = event_type
        if component is not None:
            conditions.append("component = :component")
            params["component"] = component

        where = " AND ".join(conditions)
        sql = text(
            f"SELECT * FROM spine_events "
            f"WHERE {where} "
            f"ORDER BY timestamp LIMIT :limit"
        )
        async with self._session_factory() as session:
            result = await session.execute(sql, params)
            return [dict(row._mapping) for row in result]
