"""Integration test: MemoryEventBus publish/subscribe roundtrip."""

import pytest

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.events import CandleEvent
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_event_bus_roundtrip():
    """Publish a CandleEvent and verify the subscriber receives it intact."""
    bus = MemoryEventBus()
    received = []

    async def handler(event):
        received.append(event)

    await bus.subscribe("market", "test_group", handler)

    event = CandleEvent(
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M1,
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        is_closed=True,
        source_module="test",
    )
    await bus.publish("market", event)

    assert len(received) == 1
    assert received[0].symbol == "BTC/USDT"


@pytest.mark.asyncio
async def test_event_bus_multiple_subscribers():
    """Multiple subscribers on the same topic each receive the event."""
    bus = MemoryEventBus()
    received_a = []
    received_b = []

    async def handler_a(event):
        received_a.append(event)

    async def handler_b(event):
        received_b.append(event)

    await bus.subscribe("market", "group_a", handler_a)
    await bus.subscribe("market", "group_b", handler_b)

    event = CandleEvent(
        symbol="ETH/USDT",
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M5,
        timestamp=datetime.now(timezone.utc),
        open=2000.0,
        high=2050.0,
        low=1980.0,
        close=2020.0,
        volume=500.0,
        is_closed=True,
        source_module="test",
    )
    await bus.publish("market", event)

    assert len(received_a) == 1
    assert len(received_b) == 1
    assert received_a[0].event_id == received_b[0].event_id


@pytest.mark.asyncio
async def test_event_bus_topic_isolation():
    """Events on one topic do not reach subscribers of a different topic."""
    bus = MemoryEventBus()
    received = []

    async def handler(event):
        received.append(event)

    await bus.subscribe("strategy", "test_group", handler)

    event = CandleEvent(
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M1,
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        is_closed=True,
        source_module="test",
    )
    await bus.publish("market", event)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_event_bus_history():
    """The bus records published events in its history."""
    bus = MemoryEventBus()

    event = CandleEvent(
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M1,
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000.0,
        is_closed=True,
        source_module="test",
    )
    await bus.publish("market", event)

    history = bus.get_history("market")
    assert len(history) == 1
    assert history[0][0] == "market"
    assert history[0][1].symbol == "BTC/USDT"
