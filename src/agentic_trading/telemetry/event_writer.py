"""Async buffered spine event writer.

Buffers events in memory and flushes to storage in batches.  Errors
are logged, never raised -- telemetry must never crash the trading
pipeline.

Follows the standard ``start() / stop()`` lifecycle from CLAUDE.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from agentic_trading.telemetry.models import SpineEvent
from agentic_trading.telemetry.storage import ISpineStorage

logger = logging.getLogger(__name__)


class EventWriter:
    """Async buffered writer for spine events.

    Parameters
    ----------
    storage:
        Backend implementing :class:`ISpineStorage`.
    batch_size:
        Flush automatically after this many buffered events.
    flush_interval:
        Seconds between background flush sweeps.
    enabled:
        When ``False``, :meth:`write` is a no-op.
    """

    def __init__(
        self,
        storage: ISpineStorage,
        *,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        enabled: bool = True,
        max_buffer_size: int = 10_000,
    ) -> None:
        self._storage = storage
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._enabled = enabled
        self._max_buffer_size = max_buffer_size

        self._buffer: list[SpineEvent] = []
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._running = False

        # Counters
        self._events_written: int = 0
        self._flush_count: int = 0
        self._error_count: int = 0
        self._events_dropped: int = 0

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the background flush loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._flush_loop(), name="spine-event-writer",
        )
        logger.info(
            "EventWriter started (batch_size=%d, interval=%.1fs)",
            self._batch_size,
            self._flush_interval,
        )

    async def stop(self) -> None:
        """Stop the background flush loop and flush remaining events."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        # Final flush
        await self.flush()
        logger.info(
            "EventWriter stopped (events_written=%d, flushes=%d, errors=%d)",
            self._events_written,
            self._flush_count,
            self._error_count,
        )

    # -- public API ---------------------------------------------------------

    async def write(self, event: SpineEvent) -> None:
        """Buffer a spine event for writing.

        If the buffer reaches ``batch_size``, an automatic flush is triggered.
        """
        if not self._enabled:
            return

        async with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._batch_size:
                await self._do_flush()

    async def flush(self) -> None:
        """Manually flush all buffered events to storage."""
        async with self._lock:
            await self._do_flush()

    # -- internals ----------------------------------------------------------

    async def _flush_loop(self) -> None:
        """Background loop that flushes at ``flush_interval``."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("EventWriter flush loop error")
                self._error_count += 1

    async def _do_flush(self) -> None:
        """Flush the buffer to storage.  Must be called with ``_lock`` held.

        On success the buffer is cleared.  On failure the events stay in
        the buffer so the next flush retries them.  A ``max_buffer_size``
        cap prevents unbounded memory growth during prolonged outages —
        oldest events are dropped when the cap is exceeded.
        """
        if not self._buffer:
            return

        batch = self._buffer.copy()

        try:
            await self._storage.write_batch(batch)
            # Clear only AFTER successful write
            self._buffer.clear()
            self._events_written += len(batch)
            self._flush_count += 1
        except Exception:
            self._error_count += 1
            logger.exception(
                "EventWriter storage error (buffered %d events for retry)",
                len(batch),
            )
            # Events stay in buffer for retry on next flush.
            # Enforce max buffer size to prevent OOM during prolonged outage.
            if len(self._buffer) > self._max_buffer_size:
                overflow = len(self._buffer) - self._max_buffer_size
                del self._buffer[:overflow]
                self._events_dropped += overflow
                logger.error(
                    "EventWriter overflow: dropped %d oldest events "
                    "(buffer capped at %d)",
                    overflow,
                    self._max_buffer_size,
                )

    # -- observability ------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def events_written(self) -> int:
        return self._events_written

    @property
    def flush_count(self) -> int:
        return self._flush_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def events_dropped(self) -> int:
        return self._events_dropped
