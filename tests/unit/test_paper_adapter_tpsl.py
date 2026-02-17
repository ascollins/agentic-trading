"""Tests for PaperAdapter TP/SL simulation.

Covers:
- set_trading_stop stores TP/SL levels.
- Take profit triggers when price crosses TP (long and short).
- Stop loss triggers when price crosses SL (long and short).
- Trailing stop updates reference and triggers.
- Position is closed at trigger price.
- FillEvent is published on event bus when triggered.
- Stops are cleared after triggering.
- No trigger when price doesn't cross levels.
- Merge behavior: updating existing stops.
- Reset clears stops.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentic_trading.core.enums import Exchange, OrderStatus, OrderType, Side, Timeframe, TimeInForce
from agentic_trading.core.events import FillEvent, OrderIntent
from agentic_trading.execution.adapters.paper import PaperAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(
    event_bus: Any | None = None,
    balance: Decimal = Decimal("100000"),
) -> PaperAdapter:
    return PaperAdapter(
        exchange=Exchange.BYBIT,
        initial_balances={"USDT": balance},
        event_bus=event_bus,
    )


def _make_intent(
    symbol: str = "BTC/USDT",
    side: Side = Side.BUY,
    qty: Decimal = Decimal("0.01"),
) -> OrderIntent:
    return OrderIntent(
        dedupe_key="test-dedupe",
        strategy_id="trend_following",
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=side,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=qty,
        price=None,
        trace_id="test-trace",
    )


# ---------------------------------------------------------------------------
# TP/SL storage
# ---------------------------------------------------------------------------

class TestSetTradingStop:
    @pytest.mark.asyncio
    async def test_stores_tp_sl(self) -> None:
        adapter = _make_adapter()
        result = await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("52000"),
            stop_loss=Decimal("48000"),
        )
        assert result["result"] == "paper_stop_stored"
        assert "BTC/USDT" in adapter._trading_stops
        stop = adapter._trading_stops["BTC/USDT"]
        assert stop.take_profit == Decimal("52000")
        assert stop.stop_loss == Decimal("48000")

    @pytest.mark.asyncio
    async def test_stores_trailing_stop(self) -> None:
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        await adapter.set_trading_stop(
            "BTC/USDT",
            trailing_stop=Decimal("500"),
        )
        stop = adapter._trading_stops["BTC/USDT"]
        assert stop.trailing_stop == Decimal("500")
        assert stop.trailing_ref == Decimal("50000")

    @pytest.mark.asyncio
    async def test_merge_updates_existing(self) -> None:
        adapter = _make_adapter()
        await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("52000"),
        )
        await adapter.set_trading_stop(
            "BTC/USDT",
            stop_loss=Decimal("48000"),
        )
        stop = adapter._trading_stops["BTC/USDT"]
        assert stop.take_profit == Decimal("52000")
        assert stop.stop_loss == Decimal("48000")


# ---------------------------------------------------------------------------
# Take Profit triggers
# ---------------------------------------------------------------------------

class TestTakeProfitTrigger:
    @pytest.mark.asyncio
    async def test_long_tp_triggers_on_price_above(self) -> None:
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        await adapter.submit_order(_make_intent(side=Side.BUY, qty=Decimal("0.01")))

        await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("52000"),
        )

        # Price hasn't reached TP yet
        adapter.set_market_price("BTC/USDT", Decimal("51000"))
        assert "BTC/USDT" in adapter._trading_stops  # Not triggered

        # Price reaches TP
        adapter.set_market_price("BTC/USDT", Decimal("52000"))
        assert "BTC/USDT" not in adapter._trading_stops  # Triggered & cleared

        # Position should be closed
        pos = adapter._positions.get("BTC/USDT")
        assert pos is not None
        assert pos.qty == Decimal("0")

    @pytest.mark.asyncio
    async def test_short_tp_triggers_on_price_below(self) -> None:
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        await adapter.submit_order(_make_intent(side=Side.SELL, qty=Decimal("0.01")))

        await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("48000"),
        )

        # Price reaches TP (below for short)
        adapter.set_market_price("BTC/USDT", Decimal("48000"))
        assert "BTC/USDT" not in adapter._trading_stops
        pos = adapter._positions.get("BTC/USDT")
        assert pos.qty == Decimal("0")


# ---------------------------------------------------------------------------
# Stop Loss triggers
# ---------------------------------------------------------------------------

class TestStopLossTrigger:
    @pytest.mark.asyncio
    async def test_long_sl_triggers_on_price_below(self) -> None:
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        await adapter.submit_order(_make_intent(side=Side.BUY, qty=Decimal("0.01")))

        await adapter.set_trading_stop(
            "BTC/USDT",
            stop_loss=Decimal("48000"),
        )

        # Price drops to SL
        adapter.set_market_price("BTC/USDT", Decimal("48000"))
        assert "BTC/USDT" not in adapter._trading_stops
        pos = adapter._positions.get("BTC/USDT")
        assert pos.qty == Decimal("0")

    @pytest.mark.asyncio
    async def test_short_sl_triggers_on_price_above(self) -> None:
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        await adapter.submit_order(_make_intent(side=Side.SELL, qty=Decimal("0.01")))

        await adapter.set_trading_stop(
            "BTC/USDT",
            stop_loss=Decimal("52000"),
        )

        adapter.set_market_price("BTC/USDT", Decimal("52000"))
        assert "BTC/USDT" not in adapter._trading_stops
        pos = adapter._positions.get("BTC/USDT")
        assert pos.qty == Decimal("0")


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------

class TestTrailingStop:
    @pytest.mark.asyncio
    async def test_trailing_updates_ref_and_triggers(self) -> None:
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        await adapter.submit_order(_make_intent(side=Side.BUY, qty=Decimal("0.01")))

        await adapter.set_trading_stop(
            "BTC/USDT",
            trailing_stop=Decimal("1000"),
        )

        # Price goes up — trailing ref should follow
        adapter.set_market_price("BTC/USDT", Decimal("53000"))
        stop = adapter._trading_stops.get("BTC/USDT")
        assert stop is not None
        assert stop.trailing_ref == Decimal("53000")

        # Price drops but not enough to trigger (53000 - 1000 = 52000)
        adapter.set_market_price("BTC/USDT", Decimal("52500"))
        assert "BTC/USDT" in adapter._trading_stops

        # Price drops below trailing threshold
        adapter.set_market_price("BTC/USDT", Decimal("51999"))
        assert "BTC/USDT" not in adapter._trading_stops
        pos = adapter._positions.get("BTC/USDT")
        assert pos.qty == Decimal("0")


# ---------------------------------------------------------------------------
# Event bus publication
# ---------------------------------------------------------------------------

class TestEventBusPublication:
    @pytest.mark.asyncio
    async def test_fill_published_on_trigger(self) -> None:
        """When TP/SL triggers, a FillEvent should be published."""
        published: list[Any] = []

        class _MockBus:
            async def publish(self, topic: str, event: Any) -> None:
                published.append((topic, event))

        bus = _MockBus()
        adapter = _make_adapter(event_bus=bus)
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        await adapter.submit_order(_make_intent(side=Side.BUY, qty=Decimal("0.01")))

        await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("52000"),
        )

        # Trigger TP
        adapter.set_market_price("BTC/USDT", Decimal("52000"))

        # Let the fire-and-forget task run
        await asyncio.sleep(0.01)

        assert len(published) == 1
        topic, fill = published[0]
        assert topic == "execution.fill"
        assert isinstance(fill, FillEvent)
        assert fill.symbol == "BTC/USDT"
        assert fill.side == Side.SELL  # Close a long
        assert fill.price == Decimal("52000")
        assert fill.qty == Decimal("0.01")


# ---------------------------------------------------------------------------
# No trigger / edge cases
# ---------------------------------------------------------------------------

class TestNoTrigger:
    @pytest.mark.asyncio
    async def test_price_between_tp_and_sl_no_trigger(self) -> None:
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        await adapter.submit_order(_make_intent(side=Side.BUY, qty=Decimal("0.01")))

        await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("52000"),
            stop_loss=Decimal("48000"),
        )

        # Price moves but stays between TP and SL
        adapter.set_market_price("BTC/USDT", Decimal("51000"))
        assert "BTC/USDT" in adapter._trading_stops
        pos = adapter._positions.get("BTC/USDT")
        assert pos.qty != Decimal("0")

    @pytest.mark.asyncio
    async def test_no_position_clears_orphan_stop(self) -> None:
        adapter = _make_adapter()
        # Set a stop without a position
        await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("52000"),
        )
        adapter.set_market_price("BTC/USDT", Decimal("52000"))
        # Orphan stop should be cleared
        assert "BTC/USDT" not in adapter._trading_stops

    @pytest.mark.asyncio
    async def test_reset_clears_stops(self) -> None:
        adapter = _make_adapter()
        await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("52000"),
        )
        await adapter.reset()
        assert len(adapter._trading_stops) == 0
