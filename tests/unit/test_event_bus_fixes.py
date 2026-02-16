"""Tests for event bus fixes: error handling, dead-letter, observability.

Covers the MemoryEventBus improvements and verifies the
RedisStreamsBus contract changes (without requiring a live Redis).
"""

from __future__ import annotations

import pytest

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.events import CandleEvent
from agentic_trading.event_bus.memory_bus import MemoryEventBus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(**overrides) -> CandleEvent:
    defaults = dict(
        symbol="BTC/USDT",
        exchange=Exchange.BYBIT,
        timeframe=Timeframe.M1,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        is_closed=True,
        source_module="test",
    )
    defaults.update(overrides)
    return CandleEvent(**defaults)


# ===========================================================================
# MemoryEventBus: error handling
# ===========================================================================


class TestMemoryBusErrorHandling:
    """Handler errors are tracked, logged, and reported via dead-letter."""

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash_bus(self):
        """A failing handler should not prevent other handlers from running."""
        bus = MemoryEventBus()
        received = []

        async def bad_handler(event):
            raise ValueError("boom")

        async def good_handler(event):
            received.append(event)

        await bus.subscribe("market", "bad_group", bad_handler)
        await bus.subscribe("market", "good_group", good_handler)

        await bus.publish("market", _make_candle())

        # Good handler still received the event
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_error_count_incremented(self):
        """Error counts should be tracked per topic/group."""
        bus = MemoryEventBus()

        async def bad_handler(event):
            raise RuntimeError("fail")

        await bus.subscribe("market", "g1", bad_handler)
        await bus.publish("market", _make_candle())
        await bus.publish("market", _make_candle())

        counts = bus.get_error_counts()
        assert counts["market/g1"] == 2

    @pytest.mark.asyncio
    async def test_dead_letter_recorded(self):
        """Failed messages should appear in the dead-letter list."""
        bus = MemoryEventBus()

        async def bad_handler(event):
            raise ValueError("serialize error")

        await bus.subscribe("market", "g1", bad_handler)
        await bus.publish("market", _make_candle())

        dls = bus.dead_letters
        assert len(dls) == 1
        assert dls[0].topic == "market"
        assert dls[0].group == "g1"
        assert dls[0].event_type == "CandleEvent"
        assert "serialize error" in dls[0].error

    @pytest.mark.asyncio
    async def test_clear_dead_letters(self):
        """clear_dead_letters drains the list and returns entries."""
        bus = MemoryEventBus()

        async def bad_handler(event):
            raise ValueError("err")

        await bus.subscribe("t", "g", bad_handler)
        await bus.publish("t", _make_candle())

        drained = bus.clear_dead_letters()
        assert len(drained) == 1
        assert len(bus.dead_letters) == 0

    @pytest.mark.asyncio
    async def test_messages_processed_count(self):
        """Successful messages increment the processed counter."""
        bus = MemoryEventBus()

        async def ok_handler(event):
            pass

        await bus.subscribe("t", "g", ok_handler)
        await bus.publish("t", _make_candle())
        await bus.publish("t", _make_candle())

        assert bus.messages_processed == 2

    @pytest.mark.asyncio
    async def test_failed_messages_not_counted_as_processed(self):
        """Failed messages should NOT increment the processed counter."""
        bus = MemoryEventBus()

        async def bad_handler(event):
            raise RuntimeError("fail")

        await bus.subscribe("t", "g", bad_handler)
        await bus.publish("t", _make_candle())

        assert bus.messages_processed == 0
        assert bus.get_error_counts()["t/g"] == 1


# ===========================================================================
# MemoryEventBus: error callback
# ===========================================================================


class TestMemoryBusErrorCallback:
    """The on_handler_error callback fires for each handler failure."""

    @pytest.mark.asyncio
    async def test_callback_invoked_on_error(self):
        """Error callback receives topic, group, event_id, and exception."""
        callback_calls = []

        def on_error(topic, group, msg_id, exc):
            callback_calls.append((topic, group, msg_id, type(exc).__name__))

        bus = MemoryEventBus(on_handler_error=on_error)

        async def bad_handler(event):
            raise TypeError("bad type")

        await bus.subscribe("market", "g1", bad_handler)
        event = _make_candle()
        await bus.publish("market", event)

        assert len(callback_calls) == 1
        topic, group, msg_id, exc_name = callback_calls[0]
        assert topic == "market"
        assert group == "g1"
        assert msg_id == event.event_id
        assert exc_name == "TypeError"

    @pytest.mark.asyncio
    async def test_callback_failure_does_not_crash(self):
        """A failing error callback should not crash the bus."""
        def on_error(topic, group, msg_id, exc):
            raise RuntimeError("callback crashed")

        bus = MemoryEventBus(on_handler_error=on_error)

        async def bad_handler(event):
            raise ValueError("err")

        await bus.subscribe("t", "g", bad_handler)
        # Should not raise
        await bus.publish("t", _make_candle())

        # Error still tracked
        assert bus.get_error_counts()["t/g"] == 1

    @pytest.mark.asyncio
    async def test_no_callback_by_default(self):
        """Without a callback, errors are still tracked but no callback fires."""
        bus = MemoryEventBus()

        async def bad_handler(event):
            raise ValueError("err")

        await bus.subscribe("t", "g", bad_handler)
        await bus.publish("t", _make_candle())

        assert bus.get_error_counts()["t/g"] == 1


# ===========================================================================
# MemoryEventBus: backward compatibility
# ===========================================================================


class TestMemoryBusBackwardCompat:
    """Ensure existing API remains unchanged."""

    @pytest.mark.asyncio
    async def test_basic_roundtrip(self):
        """Basic publish/subscribe still works."""
        bus = MemoryEventBus()
        received = []

        async def handler(event):
            received.append(event)

        await bus.subscribe("market", "g1", handler)
        event = _make_candle()
        await bus.publish("market", event)

        assert len(received) == 1
        assert received[0].symbol == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_history_still_works(self):
        """Event history recording is preserved."""
        bus = MemoryEventBus()
        await bus.publish("market", _make_candle())

        history = bus.get_history("market")
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_clear_history_still_works(self):
        """clear_history is preserved."""
        bus = MemoryEventBus()
        await bus.publish("market", _make_candle())
        bus.clear_history()
        assert len(bus.get_history()) == 0

    @pytest.mark.asyncio
    async def test_topic_isolation(self):
        """Events on one topic don't reach subscribers of another."""
        bus = MemoryEventBus()
        received = []

        async def handler(event):
            received.append(event)

        await bus.subscribe("strategy", "g1", handler)
        await bus.publish("market", _make_candle())
        assert len(received) == 0


# ===========================================================================
# RedisStreamsBus: unit tests (no Redis required)
# ===========================================================================


class TestRedisStreamsBusConstruction:
    """Verify RedisStreamsBus construction and observable state."""

    def test_default_construction(self):
        from agentic_trading.event_bus.redis_streams import RedisStreamsBus

        bus = RedisStreamsBus()
        assert bus.messages_processed == 0
        assert bus.dead_letters == []
        assert bus.get_error_counts() == {}

    def test_custom_retries(self):
        from agentic_trading.event_bus.redis_streams import RedisStreamsBus

        bus = RedisStreamsBus(max_handler_retries=5)
        assert bus._max_retries == 5

    def test_error_callback_stored(self):
        from agentic_trading.event_bus.redis_streams import RedisStreamsBus

        def my_callback(t, g, m, e):
            pass

        bus = RedisStreamsBus(on_handler_error=my_callback)
        assert bus._on_handler_error is my_callback

    def test_dead_letter_dataclass(self):
        from agentic_trading.event_bus.redis_streams import DeadLetter

        dl = DeadLetter(
            topic="t",
            group="g",
            msg_id="123",
            event_type="CandleEvent",
            error="boom",
            attempts=3,
        )
        assert dl.topic == "t"
        assert dl.attempts == 3


# ===========================================================================
# Bus factory
# ===========================================================================


class TestBusFactory:
    """Verify the factory passes through error callbacks."""

    def test_backtest_mode_creates_memory_bus(self):
        from agentic_trading.core.enums import Mode
        from agentic_trading.event_bus.bus import create_event_bus

        bus = create_event_bus(Mode.BACKTEST)
        assert isinstance(bus, MemoryEventBus)

    def test_paper_mode_creates_redis_bus(self):
        from agentic_trading.core.enums import Mode
        from agentic_trading.event_bus.bus import create_event_bus
        from agentic_trading.event_bus.redis_streams import RedisStreamsBus

        bus = create_event_bus(Mode.PAPER)
        assert isinstance(bus, RedisStreamsBus)

    def test_error_callback_passed_to_memory_bus(self):
        from agentic_trading.core.enums import Mode
        from agentic_trading.event_bus.bus import create_event_bus

        def cb(t, g, m, e):
            pass

        bus = create_event_bus(Mode.BACKTEST, on_handler_error=cb)
        assert bus._on_handler_error is cb

    def test_error_callback_passed_to_redis_bus(self):
        from agentic_trading.core.enums import Mode
        from agentic_trading.event_bus.bus import create_event_bus

        def cb(t, g, m, e):
            pass

        bus = create_event_bus(Mode.LIVE, on_handler_error=cb)
        assert bus._on_handler_error is cb


# ===========================================================================
# Schema registry: new approval events
# ===========================================================================


class TestSchemaRegistryApprovalEvents:
    """Verify new approval events are in the schema registry."""

    def test_approval_requested_registered(self):
        from agentic_trading.event_bus.schemas import get_event_class
        cls = get_event_class("ApprovalRequested")
        assert cls is not None
        assert cls.__name__ == "ApprovalRequested"

    def test_approval_resolved_registered(self):
        from agentic_trading.event_bus.schemas import get_event_class
        cls = get_event_class("ApprovalResolved")
        assert cls is not None
        assert cls.__name__ == "ApprovalResolved"

    def test_governance_approval_topic_exists(self):
        from agentic_trading.event_bus.schemas import TOPIC_SCHEMAS
        assert "governance.approval" in TOPIC_SCHEMAS
        assert len(TOPIC_SCHEMAS["governance.approval"]) == 2
