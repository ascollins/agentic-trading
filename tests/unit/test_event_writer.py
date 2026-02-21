"""Unit tests for EventWriter (async buffered spine event writer)."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from agentic_trading.telemetry.event_writer import EventWriter
from agentic_trading.telemetry.models import SpineEvent
from agentic_trading.telemetry.storage import MemorySpineStorage


def _make_event(**kwargs) -> SpineEvent:
    """Factory for SpineEvent test instances."""
    return SpineEvent(component="test", actor="test_agent", **kwargs)


class TestEventWriterBuffering:
    """Tests for write buffering and flush behavior."""

    @pytest.mark.asyncio
    async def test_write_buffers_events(self):
        storage = MemorySpineStorage()
        writer = EventWriter(storage, batch_size=100)

        await writer.write(_make_event())
        await writer.write(_make_event())
        await writer.write(_make_event())

        # Storage should still be empty -- not flushed yet
        assert len(storage.events) == 0
        assert writer.buffer_size == 3

    @pytest.mark.asyncio
    async def test_flush_writes_buffered(self):
        storage = MemorySpineStorage()
        writer = EventWriter(storage, batch_size=100)

        await writer.write(_make_event())
        await writer.write(_make_event())
        await writer.write(_make_event())

        await writer.flush()

        assert len(storage.events) == 3
        assert writer.buffer_size == 0
        assert writer.events_written == 3
        assert writer.flush_count == 1

    @pytest.mark.asyncio
    async def test_auto_flush_on_batch_size(self):
        storage = MemorySpineStorage()
        writer = EventWriter(storage, batch_size=5)

        for _ in range(5):
            await writer.write(_make_event())

        # Should have auto-flushed on the 5th write
        assert len(storage.events) == 5
        assert writer.buffer_size == 0

    @pytest.mark.asyncio
    async def test_auto_flush_partial_batch_remains(self):
        storage = MemorySpineStorage()
        writer = EventWriter(storage, batch_size=5)

        for _ in range(7):
            await writer.write(_make_event())

        # 5 flushed + 2 remaining in buffer
        assert len(storage.events) == 5
        assert writer.buffer_size == 2

        await writer.flush()
        assert len(storage.events) == 7
        assert writer.buffer_size == 0


class TestEventWriterLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_background_flush_loop(self):
        storage = MemorySpineStorage()
        writer = EventWriter(storage, batch_size=100, flush_interval=0.1)

        await writer.start()
        try:
            await writer.write(_make_event())

            # Wait for background flush (interval=0.1s)
            await asyncio.sleep(0.3)

            assert len(storage.events) == 1
        finally:
            await writer.stop()

    @pytest.mark.asyncio
    async def test_stop_flushes_remaining(self):
        storage = MemorySpineStorage()
        writer = EventWriter(storage, batch_size=100, flush_interval=60.0)

        await writer.start()
        await writer.write(_make_event())
        await writer.write(_make_event())
        await writer.write(_make_event())

        # Stop should flush remaining events
        await writer.stop()

        assert len(storage.events) == 3
        assert writer.buffer_size == 0

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self):
        storage = MemorySpineStorage()
        writer = EventWriter(storage)

        await writer.start()
        await writer.start()  # Should not create duplicate tasks
        await writer.stop()


class TestEventWriterDisabled:
    """Tests for disabled writer."""

    @pytest.mark.asyncio
    async def test_disabled_writer_noop(self):
        storage = MemorySpineStorage()
        writer = EventWriter(storage, enabled=False)

        await writer.write(_make_event())
        await writer.write(_make_event())
        await writer.flush()

        assert len(storage.events) == 0
        assert writer.buffer_size == 0
        assert writer.events_written == 0
        assert writer.enabled is False


class TestEventWriterErrorHandling:
    """Tests for error handling -- telemetry must never crash."""

    @pytest.mark.asyncio
    async def test_storage_error_logged_not_raised(self, caplog):
        storage = AsyncMock()
        storage.write_batch = AsyncMock(side_effect=RuntimeError("DB down"))

        writer = EventWriter(storage, batch_size=100)

        await writer.write(_make_event())

        with caplog.at_level(logging.ERROR):
            await writer.flush()

        # Should NOT raise, should log
        assert writer.error_count == 1
        assert writer.events_written == 0

    @pytest.mark.asyncio
    async def test_multiple_errors_counted(self):
        storage = AsyncMock()
        storage.write_batch = AsyncMock(side_effect=RuntimeError("DB down"))

        writer = EventWriter(storage, batch_size=100)

        await writer.write(_make_event())
        await writer.flush()
        await writer.flush()  # retry same event

        assert writer.error_count == 2


class TestEventWriterRetry:
    """Tests for retry-on-failure behavior (P0-1 fix)."""

    @pytest.mark.asyncio
    async def test_buffer_retained_on_failure(self):
        """Events stay in buffer when storage fails — available for retry."""
        storage = AsyncMock()
        storage.write_batch = AsyncMock(side_effect=RuntimeError("DB down"))

        writer = EventWriter(storage, batch_size=100)

        await writer.write(_make_event())
        await writer.write(_make_event())

        await writer.flush()

        # Buffer should NOT be cleared on failure
        assert writer.buffer_size == 2
        assert writer.error_count == 1
        assert writer.events_written == 0

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failure(self):
        """After storage recovers, buffered events are written on next flush."""
        call_count = 0

        async def flaky_write(batch):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("DB down")
            # Second call succeeds

        storage = AsyncMock()
        storage.write_batch = AsyncMock(side_effect=flaky_write)

        writer = EventWriter(storage, batch_size=100)

        await writer.write(_make_event())
        await writer.write(_make_event())

        # First flush fails — events retained
        await writer.flush()
        assert writer.buffer_size == 2
        assert writer.error_count == 1

        # Second flush succeeds — events written
        await writer.flush()
        assert writer.buffer_size == 0
        assert writer.events_written == 2
        assert writer.error_count == 1

    @pytest.mark.asyncio
    async def test_new_events_accumulate_with_retained_on_failure(self):
        """New events added after failure accumulate with retained events."""
        storage = AsyncMock()
        storage.write_batch = AsyncMock(side_effect=RuntimeError("DB down"))

        writer = EventWriter(storage, batch_size=100)

        await writer.write(_make_event())
        await writer.flush()  # Fails — 1 event retained

        await writer.write(_make_event())
        await writer.write(_make_event())

        # 3 events total in buffer (1 retained + 2 new)
        assert writer.buffer_size == 3

    @pytest.mark.asyncio
    async def test_max_buffer_size_prevents_oom(self):
        """Buffer overflow drops oldest events to prevent unbounded growth."""
        storage = AsyncMock()
        storage.write_batch = AsyncMock(side_effect=RuntimeError("DB down"))

        writer = EventWriter(storage, batch_size=100, max_buffer_size=5)

        # Write 5 events
        for _ in range(5):
            await writer.write(_make_event())

        # Flush fails — 5 events retained in buffer
        await writer.flush()
        assert writer.buffer_size == 5

        # Write 3 more events (total would be 8, exceeds max_buffer_size=5)
        for _ in range(3):
            await writer.write(_make_event())

        # Flush fails again — oldest should be dropped
        await writer.flush()
        assert writer.buffer_size == 5  # Capped at max
        assert writer.events_dropped == 3

    @pytest.mark.asyncio
    async def test_buffer_cleared_only_on_success(self):
        """Buffer is cleared ONLY after successful write, never before."""
        storage = MemorySpineStorage()
        writer = EventWriter(storage, batch_size=100)

        await writer.write(_make_event())
        await writer.write(_make_event())

        await writer.flush()

        # Success — buffer cleared, events written
        assert writer.buffer_size == 0
        assert writer.events_written == 2
        assert len(storage.events) == 2
