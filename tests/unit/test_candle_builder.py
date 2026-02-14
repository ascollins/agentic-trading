"""Test CandleBuilder aggregates 1m candles into higher timeframes."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.models import Candle
from agentic_trading.data.candle_builder import CandleBuilder


def _make_1m_candle(
    minutes_offset: int,
    open_price: float = 67000.0,
    high_price: float = 67100.0,
    low_price: float = 66900.0,
    close_price: float = 67050.0,
    volume: float = 10.0,
) -> Candle:
    """Create a 1m candle at a given minute offset from epoch-aligned base."""
    base = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    return Candle(
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M1,
        timestamp=base + timedelta(minutes=minutes_offset),
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        quote_volume=close_price * volume,
        trades=100,
        is_closed=True,
    )


class TestCandleBuilder5m:
    async def test_five_1m_candles_produce_one_5m(self, memory_bus):
        builder = CandleBuilder(
            event_bus=memory_bus,
            target_timeframes=[Timeframe.M5],
        )

        # Feed 5 candles at minute offsets 0-4 (all in same 5m window)
        candles = [
            _make_1m_candle(0, open_price=100.0, high_price=105.0, low_price=98.0, close_price=103.0, volume=10.0),
            _make_1m_candle(1, open_price=103.0, high_price=107.0, low_price=101.0, close_price=106.0, volume=8.0),
            _make_1m_candle(2, open_price=106.0, high_price=108.0, low_price=104.0, close_price=105.0, volume=12.0),
            _make_1m_candle(3, open_price=105.0, high_price=110.0, low_price=103.0, close_price=109.0, volume=15.0),
            _make_1m_candle(4, open_price=109.0, high_price=111.0, low_price=107.0, close_price=110.0, volume=9.0),
        ]

        emitted = []
        for c in candles:
            result = await builder.process(c)
            emitted.extend(result)

        # No 5m candle emitted yet (window not closed until a candle from the NEXT window arrives)
        assert len(emitted) == 0

        # Feed the first candle of the next 5m window to close the previous one
        next_candle = _make_1m_candle(5, open_price=110.0, high_price=112.0, low_price=109.0, close_price=111.0)
        result = await builder.process(next_candle)
        emitted.extend(result)

        assert len(emitted) == 1
        event = emitted[0]
        assert event.timeframe == Timeframe.M5
        assert event.is_closed is True
        assert event.open == 100.0
        assert event.high == 111.0  # max of all highs
        assert event.low == 98.0   # min of all lows
        assert event.close == 110.0  # last close

    async def test_volume_is_summed(self, memory_bus):
        builder = CandleBuilder(
            event_bus=memory_bus,
            target_timeframes=[Timeframe.M5],
        )

        volumes = [10.0, 8.0, 12.0, 15.0, 9.0]
        for i, vol in enumerate(volumes):
            await builder.process(_make_1m_candle(i, volume=vol))

        # Trigger close
        result = await builder.process(_make_1m_candle(5))
        assert len(result) == 1
        assert result[0].volume == pytest.approx(sum(volumes))

    async def test_non_1m_candle_ignored(self, memory_bus):
        builder = CandleBuilder(
            event_bus=memory_bus,
            target_timeframes=[Timeframe.M5],
        )
        candle_5m = Candle(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            timeframe=Timeframe.M5,
            timestamp=datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
            open=100.0, high=105.0, low=98.0, close=103.0, volume=50.0,
        )
        result = await builder.process(candle_5m)
        assert result == []

    async def test_partial_candle_available(self, memory_bus):
        builder = CandleBuilder(
            event_bus=memory_bus,
            target_timeframes=[Timeframe.M5],
        )
        await builder.process(_make_1m_candle(0))
        await builder.process(_make_1m_candle(1))

        partial = builder.get_partial("BTC/USDT", Exchange.BINANCE, Timeframe.M5)
        assert partial is not None
        assert partial.is_closed is False
