"""Tests for PrivateStreamManager: private WebSocket stream handling.

Tests cover: order update parsing, fill event parsing, position update
parsing, balance update parsing, and stream lifecycle.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_trading.core.enums import Exchange, OrderStatus, Side
from agentic_trading.core.events import (
    BalanceUpdate,
    FillEvent,
    OrderUpdate,
    PositionUpdate,
)
from agentic_trading.data.private_streams import PrivateStreamManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_event_bus():
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.start = AsyncMock()
    bus.stop = AsyncMock()
    return bus


@pytest.fixture
def manager(mock_event_bus):
    return PrivateStreamManager(
        event_bus=mock_event_bus,
        exchange_configs=[],
        symbols=["BTC/USDT:USDT"],
    )


# ===========================================================================
# Order update parsing
# ===========================================================================

class TestParseOrderUpdate:
    def test_parse_filled_order(self, manager):
        raw = {
            "id": "order-123",
            "clientOrderId": "client-abc",
            "symbol": "BTC/USDT:USDT",
            "status": "closed",
            "filled": 0.01,
            "remaining": 0,
            "average": 50500.0,
        }
        event = manager._parse_order_update(Exchange.BYBIT, raw)

        assert isinstance(event, OrderUpdate)
        assert event.order_id == "order-123"
        assert event.client_order_id == "client-abc"
        assert event.symbol == "BTC/USDT:USDT"
        assert event.status == OrderStatus.FILLED
        assert event.filled_qty == Decimal("0.01")
        assert event.remaining_qty == Decimal("0")
        assert event.avg_fill_price == Decimal("50500.0")

    def test_parse_open_order(self, manager):
        raw = {
            "id": "order-456",
            "clientOrderId": "client-def",
            "symbol": "ETH/USDT:USDT",
            "status": "open",
            "filled": 0,
            "remaining": 0.5,
        }
        event = manager._parse_order_update(Exchange.BYBIT, raw)

        assert event.status == OrderStatus.SUBMITTED
        assert event.remaining_qty == Decimal("0.5")
        assert event.avg_fill_price is None

    def test_parse_cancelled_order(self, manager):
        raw = {
            "id": "order-789",
            "clientOrderId": "",
            "symbol": "BTC/USDT:USDT",
            "status": "canceled",
            "filled": 0,
            "remaining": 0.01,
        }
        event = manager._parse_order_update(Exchange.BYBIT, raw)
        assert event.status == OrderStatus.CANCELLED

    def test_parse_partially_filled(self, manager):
        raw = {
            "id": "order-partial",
            "clientOrderId": "cp-1",
            "symbol": "BTC/USDT:USDT",
            "status": "open",
            "filled": 0.005,
            "remaining": 0.005,
            "average": 49800.0,
        }
        event = manager._parse_order_update(Exchange.BYBIT, raw)
        assert event.filled_qty == Decimal("0.005")
        assert event.avg_fill_price == Decimal("49800.0")

    def test_parse_order_missing_fields(self, manager):
        """Parsing gracefully handles missing optional fields."""
        raw = {"id": "o-1", "status": "open"}
        event = manager._parse_order_update(Exchange.BYBIT, raw)
        assert event is not None
        assert event.order_id == "o-1"


# ===========================================================================
# Fill event parsing
# ===========================================================================

class TestParseFillEvent:
    def test_parse_buy_fill(self, manager):
        raw = {
            "id": "fill-001",
            "order": "order-123",
            "clientOrderId": "client-abc",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "price": 50000.0,
            "amount": 0.01,
            "fee": {"cost": 0.5, "currency": "USDT"},
            "takerOrMaker": "taker",
        }
        event = manager._parse_fill_event(Exchange.BYBIT, raw)

        assert isinstance(event, FillEvent)
        assert event.fill_id == "fill-001"
        assert event.order_id == "order-123"
        assert event.side == Side.BUY
        assert event.price == Decimal("50000.0")
        assert event.qty == Decimal("0.01")
        assert event.fee == Decimal("0.5")
        assert event.fee_currency == "USDT"
        assert event.is_maker is False

    def test_parse_sell_fill(self, manager):
        raw = {
            "id": "fill-002",
            "order": "order-456",
            "clientOrderId": "",
            "symbol": "ETH/USDT:USDT",
            "side": "sell",
            "price": 3000.0,
            "amount": 1.0,
            "fee": {"cost": 0.3, "currency": "USDT"},
            "takerOrMaker": "maker",
        }
        event = manager._parse_fill_event(Exchange.BYBIT, raw)
        assert event.side == Side.SELL
        assert event.is_maker is True

    def test_parse_fill_missing_fee(self, manager):
        """Fill with missing fee should default to 0."""
        raw = {
            "id": "fill-003",
            "order": "order-789",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "price": 50000,
            "amount": 0.01,
        }
        event = manager._parse_fill_event(Exchange.BYBIT, raw)
        assert event.fee == Decimal("0")

    def test_parse_fill_maker_flag(self, manager):
        """Fill with explicit 'maker' field."""
        raw = {
            "id": "fill-004",
            "order": "order-100",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "price": 50000,
            "amount": 0.01,
            "fee": {"cost": 0.1, "currency": "USDT"},
            "maker": True,
        }
        event = manager._parse_fill_event(Exchange.BYBIT, raw)
        assert event.is_maker is True


# ===========================================================================
# Position update parsing
# ===========================================================================

class TestParsePositionUpdate:
    def test_parse_long_position(self, manager):
        raw = {
            "symbol": "BTC/USDT:USDT",
            "contracts": 1,
            "contractSize": 0.01,
            "entryPrice": 50000,
            "markPrice": 51000,
            "unrealizedPnl": 10.0,
            "realizedPnl": 5.0,
            "leverage": 10,
        }
        event = manager._parse_position_update(Exchange.BYBIT, raw)

        assert isinstance(event, PositionUpdate)
        assert event.symbol == "BTC/USDT:USDT"
        assert event.qty == Decimal("0.01")
        assert event.entry_price == Decimal("50000")
        assert event.mark_price == Decimal("51000")
        assert event.unrealized_pnl == Decimal("10.0")
        assert event.leverage == 10

    def test_parse_zero_position(self, manager):
        """Zero-size positions should still be parsed (events track changes)."""
        raw = {
            "symbol": "BTC/USDT:USDT",
            "contracts": 0,
            "contractSize": 1,
            "entryPrice": 0,
            "markPrice": 0,
        }
        event = manager._parse_position_update(Exchange.BYBIT, raw)
        assert event is not None
        assert event.qty == Decimal("0")

    def test_parse_high_leverage_position(self, manager):
        raw = {
            "symbol": "ETH/USDT:USDT",
            "contracts": 10,
            "contractSize": 1,
            "entryPrice": 3000,
            "markPrice": 3050,
            "unrealizedPnl": 500,
            "leverage": 100,
        }
        event = manager._parse_position_update(Exchange.BYBIT, raw)
        assert event.qty == Decimal("10")
        assert event.leverage == 100


# ===========================================================================
# Lifecycle and properties
# ===========================================================================

class TestStreamManagerLifecycle:
    def test_initial_state(self, manager):
        assert not manager.is_running
        assert manager.active_task_count == 0

    @pytest.mark.asyncio
    async def test_start_without_configs(self, manager):
        """start() with no exchange configs should not crash."""
        await manager.start()
        assert manager.is_running
        assert manager.active_task_count == 0
        await manager.stop()
        assert not manager.is_running

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, manager):
        """Calling stop() multiple times should not raise."""
        await manager.stop()
        await manager.stop()
        assert not manager.is_running

    def test_topics_configurable(self, mock_event_bus):
        """Custom topics should be stored."""
        mgr = PrivateStreamManager(
            event_bus=mock_event_bus,
            exchange_configs=[],
            symbols=[],
            order_topic="custom.orders",
            fill_topic="custom.fills",
            position_topic="custom.positions",
            balance_topic="custom.balances",
        )
        assert mgr._order_topic == "custom.orders"
        assert mgr._fill_topic == "custom.fills"
        assert mgr._position_topic == "custom.positions"
        assert mgr._balance_topic == "custom.balances"


# ===========================================================================
# Edge cases
# ===========================================================================

class TestParsingEdgeCases:
    def test_order_update_unknown_status(self, manager):
        """Unknown status strings should default to SUBMITTED."""
        raw = {"id": "o-x", "status": "weird_status"}
        event = manager._parse_order_update(Exchange.BYBIT, raw)
        assert event.status == OrderStatus.SUBMITTED

    def test_fill_event_no_amount(self, manager):
        """Fill with missing amount defaults to 0."""
        raw = {
            "id": "f-x",
            "order": "o-x",
            "symbol": "BTC/USDT",
            "side": "buy",
            "price": 50000,
        }
        event = manager._parse_fill_event(Exchange.BYBIT, raw)
        assert event.qty == Decimal("0")

    def test_position_update_none_values(self, manager):
        """Position with None values should default to 0."""
        raw = {
            "symbol": "BTC/USDT",
            "contracts": None,
            "contractSize": None,
            "entryPrice": None,
            "markPrice": None,
            "unrealizedPnl": None,
            "leverage": None,
        }
        event = manager._parse_position_update(Exchange.BYBIT, raw)
        assert event.qty == Decimal("0")
        assert event.leverage == 1
