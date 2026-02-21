"""Tests for ExecutionEngine._resolve_fill_price.

Covers:
- PaperAdapter order record provides fill price.
- CCXTAdapter _last_fill_price provides fill price.
- Intent price provides fill price (limit orders).
- Adapter get_market_price provides fallback price.
- Raises OrderRejectedError when ALL sources fail (E1 fix).
- Priority ordering: response > paper > ccxt > intent > market.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentic_trading.core.enums import Exchange, OrderStatus, OrderType, Side, TimeInForce
from agentic_trading.core.errors import OrderRejectedError
from agentic_trading.core.events import OrderAck, OrderIntent
from agentic_trading.execution.engine import ExecutionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intent(
    price: Decimal | None = None,
    symbol: str = "BTC/USDT",
) -> OrderIntent:
    return OrderIntent(
        dedupe_key="test-dedupe",
        strategy_id="trend_following",
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=Decimal("0.01"),
        price=price,
        trace_id="test-trace-123",
    )


def _make_ack(order_id: str = "order-1") -> OrderAck:
    return OrderAck(
        order_id=order_id,
        client_order_id="test-dedupe",
        symbol="BTC/USDT",
        exchange=Exchange.BYBIT,
        status=OrderStatus.FILLED,
    )


def _make_engine(adapter: Any = None) -> ExecutionEngine:
    """Build a minimal ExecutionEngine for testing _resolve_fill_price."""
    if adapter is None:
        adapter = MagicMock()
    bus = MagicMock()
    bus.publish = MagicMock(return_value=None)
    risk = MagicMock()
    return ExecutionEngine(
        adapter=adapter,
        event_bus=bus,
        risk_manager=risk,
        kill_switch=lambda: False,
        portfolio_state_provider=lambda: None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResolveFillPrice:
    def test_response_avg_fill_price(self) -> None:
        """Response avg_fill_price has highest priority."""
        engine = _make_engine()
        price = engine._resolve_fill_price(
            _make_intent(),
            _make_ack(),
            response={"avg_fill_price": "50123.45"},
        )
        assert price == Decimal("50123.45")

    def test_paper_adapter_order_record(self) -> None:
        """PaperAdapter _orders provides fill price."""
        adapter = MagicMock()
        order_mock = MagicMock()
        order_mock.avg_fill_price = Decimal("50050")
        adapter._orders = {"order-1": order_mock}
        engine = _make_engine(adapter)

        price = engine._resolve_fill_price(
            _make_intent(),
            _make_ack(order_id="order-1"),
            response=None,
        )
        assert price == Decimal("50050")

    def test_ccxt_last_fill_price(self) -> None:
        """CCXTAdapter _last_fill_price provides fill price."""
        adapter = MagicMock()
        del adapter._orders  # No paper orders
        adapter._last_fill_price = Decimal("50100")
        engine = _make_engine(adapter)

        price = engine._resolve_fill_price(
            _make_intent(),
            _make_ack(),
            response=None,
        )
        assert price == Decimal("50100")

    def test_intent_price_for_limit_orders(self) -> None:
        """Intent price used for limit orders."""
        adapter = MagicMock()
        del adapter._orders
        del adapter._last_fill_price
        del adapter.get_market_price
        engine = _make_engine(adapter)

        price = engine._resolve_fill_price(
            _make_intent(price=Decimal("49900")),
            _make_ack(),
            response=None,
        )
        assert price == Decimal("49900")

    def test_market_price_fallback(self) -> None:
        """Adapter get_market_price used as last resort."""
        adapter = MagicMock()
        del adapter._orders
        del adapter._last_fill_price
        adapter.get_market_price = MagicMock(return_value=Decimal("50200"))
        engine = _make_engine(adapter)

        price = engine._resolve_fill_price(
            _make_intent(),  # Market order, no intent.price
            _make_ack(),
            response=None,
        )
        assert price == Decimal("50200")

    def test_all_sources_fail_raises_rejected(self) -> None:
        """Raises OrderRejectedError when all sources exhausted (E1 fix)."""
        adapter = MagicMock()
        del adapter._orders
        del adapter._last_fill_price
        del adapter.get_market_price
        engine = _make_engine(adapter)

        with pytest.raises(OrderRejectedError, match="all 5 price sources exhausted"):
            engine._resolve_fill_price(
                _make_intent(),  # Market order, no intent.price
                _make_ack(),
                response=None,
            )

    def test_priority_response_over_paper(self) -> None:
        """Response has higher priority than PaperAdapter."""
        adapter = MagicMock()
        order_mock = MagicMock()
        order_mock.avg_fill_price = Decimal("50050")
        adapter._orders = {"order-1": order_mock}
        engine = _make_engine(adapter)

        price = engine._resolve_fill_price(
            _make_intent(),
            _make_ack(order_id="order-1"),
            response={"avg_fill_price": "50123"},
        )
        assert price == Decimal("50123")  # Response wins

    def test_zero_response_falls_through(self) -> None:
        """Response with 0 value falls through to next source."""
        adapter = MagicMock()
        order_mock = MagicMock()
        order_mock.avg_fill_price = Decimal("50050")
        adapter._orders = {"order-1": order_mock}
        engine = _make_engine(adapter)

        price = engine._resolve_fill_price(
            _make_intent(),
            _make_ack(order_id="order-1"),
            response={"avg_fill_price": "0"},
        )
        assert price == Decimal("50050")  # Falls to paper
