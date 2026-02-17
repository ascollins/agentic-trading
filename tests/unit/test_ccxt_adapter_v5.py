"""Tests for CCXTAdapter V5-enhanced methods.

Tests cover: amend_order, batch_submit_orders, set_leverage,
set_position_mode, set_trading_stop, get_closed_pnl, and symbol conversion.
All exchange calls are mocked via AsyncMock.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.core.enums import (
    Exchange,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from agentic_trading.core.errors import ExchangeError
from agentic_trading.core.events import OrderAck, OrderIntent
from agentic_trading.execution.adapters.ccxt_adapter import CCXTAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ccxt_exchange():
    """Create a mock CCXT exchange object with all needed methods."""
    mock = AsyncMock()
    mock.load_markets = AsyncMock()
    mock.edit_order = AsyncMock()
    mock.create_orders = AsyncMock()
    mock.create_order = AsyncMock()
    mock.set_leverage = AsyncMock()
    mock.set_position_mode = AsyncMock()
    mock.cancel_order = AsyncMock()
    mock.fetch_my_trades = AsyncMock()
    mock.privatePostV5PositionTradingStop = AsyncMock()
    mock.privateGetV5PositionClosedPnl = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def adapter_bybit(mock_ccxt_exchange):
    """Create a CCXTAdapter for Bybit with mocked CCXT exchange."""
    with patch("agentic_trading.execution.adapters.ccxt_adapter.ccxt_async") as mock_ccxt:
        mock_ccxt.bybit = MagicMock(return_value=mock_ccxt_exchange)
        mock_ccxt.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_ccxt.RateLimitExceeded = type("RateLimitExceeded", (Exception,), {})
        mock_ccxt.InsufficientFunds = type("InsufficientFunds", (Exception,), {})
        mock_ccxt.BaseError = type("BaseError", (Exception,), {})

        adapter = CCXTAdapter(
            exchange_name="bybit",
            api_key="test_key",
            api_secret="test_secret",
            default_type="swap",
        )
        # Mark markets as loaded to skip load_markets
        adapter._markets_loaded = True
        return adapter


@pytest.fixture
def adapter_binance(mock_ccxt_exchange):
    """Create a CCXTAdapter for Binance with mocked CCXT exchange."""
    with patch("agentic_trading.execution.adapters.ccxt_adapter.ccxt_async") as mock_ccxt:
        mock_ccxt.binance = MagicMock(return_value=mock_ccxt_exchange)
        mock_ccxt.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_ccxt.RateLimitExceeded = type("RateLimitExceeded", (Exception,), {})
        mock_ccxt.InsufficientFunds = type("InsufficientFunds", (Exception,), {})
        mock_ccxt.BaseError = type("BaseError", (Exception,), {})

        adapter = CCXTAdapter(
            exchange_name="binance",
            api_key="test_key",
            api_secret="test_secret",
            default_type="swap",
        )
        adapter._markets_loaded = True
        return adapter


def _make_intent(
    symbol: str = "BTC/USDT:USDT",
    side: Side = Side.BUY,
    qty: Decimal = Decimal("0.01"),
    price: Decimal | None = Decimal("50000"),
    order_type: OrderType = OrderType.LIMIT,
    dedupe_key: str = "test-key-001",
) -> OrderIntent:
    return OrderIntent(
        dedupe_key=dedupe_key,
        strategy_id="test_strategy",
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=side,
        order_type=order_type,
        qty=qty,
        price=price,
    )


# ===========================================================================
# amend_order
# ===========================================================================

class TestAmendOrder:
    @pytest.mark.asyncio
    async def test_amend_order_price_and_qty(self, adapter_bybit, mock_ccxt_exchange):
        """amend_order should call CCXT edit_order with correct params."""
        mock_ccxt_exchange.edit_order.return_value = {
            "id": "order-123",
            "clientOrderId": "client-123",
            "status": "open",
        }

        ack = await adapter_bybit.amend_order(
            order_id="order-123",
            symbol="BTC/USDT:USDT",
            qty=Decimal("0.02"),
            price=Decimal("51000"),
        )

        assert isinstance(ack, OrderAck)
        assert ack.order_id == "order-123"
        assert ack.status == OrderStatus.SUBMITTED

        mock_ccxt_exchange.edit_order.assert_called_once()
        call_kwargs = mock_ccxt_exchange.edit_order.call_args
        assert call_kwargs.kwargs["id"] == "order-123"
        assert call_kwargs.kwargs["amount"] == 0.02
        assert call_kwargs.kwargs["price"] == 51000.0

    @pytest.mark.asyncio
    async def test_amend_order_price_only(self, adapter_bybit, mock_ccxt_exchange):
        """amend_order with only price should pass None for amount."""
        mock_ccxt_exchange.edit_order.return_value = {
            "id": "order-123",
            "clientOrderId": "client-123",
            "status": "open",
        }

        ack = await adapter_bybit.amend_order(
            order_id="order-123",
            symbol="BTC/USDT:USDT",
            price=Decimal("52000"),
        )

        assert ack.status == OrderStatus.SUBMITTED
        call_kwargs = mock_ccxt_exchange.edit_order.call_args
        assert call_kwargs.kwargs["amount"] is None
        assert call_kwargs.kwargs["price"] == 52000.0

    @pytest.mark.asyncio
    async def test_amend_order_with_stop_price(self, adapter_bybit, mock_ccxt_exchange):
        """amend_order with stop_price should pass it in params."""
        mock_ccxt_exchange.edit_order.return_value = {
            "id": "order-456",
            "clientOrderId": "",
            "status": "open",
        }

        ack = await adapter_bybit.amend_order(
            order_id="order-456",
            symbol="ETH/USDT:USDT",
            stop_price=Decimal("3000"),
        )

        assert ack.order_id == "order-456"
        call_kwargs = mock_ccxt_exchange.edit_order.call_args
        assert call_kwargs.kwargs["params"]["stopPrice"] == 3000.0

    @pytest.mark.asyncio
    async def test_amend_order_no_edit_order_support(self, adapter_binance, mock_ccxt_exchange):
        """amend_order should raise ExchangeError if edit_order not available."""
        del mock_ccxt_exchange.edit_order  # Remove the method

        with pytest.raises(ExchangeError, match="does not support edit_order"):
            await adapter_binance.amend_order(
                order_id="order-789",
                symbol="BTC/USDT",
                price=Decimal("50000"),
            )


# ===========================================================================
# batch_submit_orders
# ===========================================================================

class TestBatchSubmitOrders:
    @pytest.mark.asyncio
    async def test_batch_submit_native(self, adapter_bybit, mock_ccxt_exchange):
        """batch_submit_orders should use create_orders when available."""
        mock_ccxt_exchange.create_orders.return_value = [
            {"id": "b1", "clientOrderId": "ck1", "symbol": "BTC/USDT:USDT", "status": "open"},
            {"id": "b2", "clientOrderId": "ck2", "symbol": "ETH/USDT:USDT", "status": "open"},
        ]

        intents = [
            _make_intent(symbol="BTC/USDT:USDT", dedupe_key="ck1"),
            _make_intent(symbol="ETH/USDT:USDT", dedupe_key="ck2", qty=Decimal("0.1")),
        ]

        acks = await adapter_bybit.batch_submit_orders(intents)

        assert len(acks) == 2
        assert acks[0].order_id == "b1"
        assert acks[1].order_id == "b2"
        mock_ccxt_exchange.create_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_submit_sequential_fallback(self, adapter_bybit, mock_ccxt_exchange):
        """batch_submit_orders falls back to sequential when create_orders missing."""
        del mock_ccxt_exchange.create_orders

        mock_ccxt_exchange.create_order.side_effect = [
            {"id": "s1", "clientOrderId": "ck1", "status": "open"},
            {"id": "s2", "clientOrderId": "ck2", "status": "open"},
        ]

        intents = [
            _make_intent(dedupe_key="ck1"),
            _make_intent(dedupe_key="ck2"),
        ]

        acks = await adapter_bybit.batch_submit_orders(intents)

        assert len(acks) == 2
        assert mock_ccxt_exchange.create_order.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_submit_empty(self, adapter_bybit, mock_ccxt_exchange):
        """batch_submit_orders with empty list returns empty list."""
        mock_ccxt_exchange.create_orders.return_value = []
        acks = await adapter_bybit.batch_submit_orders([])
        assert acks == []

    @pytest.mark.asyncio
    async def test_batch_submit_bybit_params(self, adapter_bybit, mock_ccxt_exchange):
        """Batch orders for Bybit should include orderLinkId."""
        mock_ccxt_exchange.create_orders.return_value = [
            {"id": "b1", "clientOrderId": "dk1", "symbol": "BTC/USDT:USDT", "status": "open"},
        ]

        intents = [_make_intent(dedupe_key="dk1")]
        await adapter_bybit.batch_submit_orders(intents)

        call_args = mock_ccxt_exchange.create_orders.call_args[0][0]
        assert call_args[0]["params"]["orderLinkId"] == "dk1"

    @pytest.mark.asyncio
    async def test_batch_submit_with_reduce_only(self, adapter_bybit, mock_ccxt_exchange):
        """Batch orders should forward reduce_only and post_only params."""
        mock_ccxt_exchange.create_orders.return_value = [
            {"id": "b1", "clientOrderId": "dk1", "symbol": "BTC/USDT:USDT", "status": "open"},
        ]

        intent = _make_intent(dedupe_key="dk1")
        intent.reduce_only = True
        intent.post_only = True
        intent.time_in_force = TimeInForce.POST_ONLY

        await adapter_bybit.batch_submit_orders([intent])

        call_args = mock_ccxt_exchange.create_orders.call_args[0][0]
        assert call_args[0]["params"]["reduceOnly"] is True
        assert call_args[0]["params"]["postOnly"] is True
        assert call_args[0]["params"]["timeInForce"] == "PostOnly"


# ===========================================================================
# set_leverage
# ===========================================================================

class TestSetLeverage:
    @pytest.mark.asyncio
    async def test_set_leverage_success(self, adapter_bybit, mock_ccxt_exchange):
        """set_leverage should call CCXT set_leverage and return result."""
        mock_ccxt_exchange.set_leverage.return_value = {"retCode": 0, "retMsg": "OK"}

        result = await adapter_bybit.set_leverage("BTC/USDT:USDT", 10)

        assert result["retCode"] == 0
        mock_ccxt_exchange.set_leverage.assert_called_once_with(10, "BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_set_leverage_non_dict_result(self, adapter_bybit, mock_ccxt_exchange):
        """set_leverage wraps non-dict results."""
        mock_ccxt_exchange.set_leverage.return_value = "OK"

        result = await adapter_bybit.set_leverage("BTC/USDT:USDT", 5)

        assert result == {"result": "OK"}

    @pytest.mark.asyncio
    async def test_set_leverage_high_value(self, adapter_bybit, mock_ccxt_exchange):
        """set_leverage handles high leverage values (e.g., 100x)."""
        mock_ccxt_exchange.set_leverage.return_value = {"retCode": 0}

        result = await adapter_bybit.set_leverage("BTC/USDT:USDT", 100)
        mock_ccxt_exchange.set_leverage.assert_called_once_with(100, "BTC/USDT:USDT")


# ===========================================================================
# set_position_mode
# ===========================================================================

class TestSetPositionMode:
    @pytest.mark.asyncio
    async def test_set_hedge_mode(self, adapter_bybit, mock_ccxt_exchange):
        """set_position_mode('hedge') should call set_position_mode(True, ...)."""
        mock_ccxt_exchange.set_position_mode.return_value = {"retCode": 0}

        result = await adapter_bybit.set_position_mode("BTC/USDT:USDT", "hedge")

        mock_ccxt_exchange.set_position_mode.assert_called_once_with(True, "BTC/USDT:USDT")
        assert result["retCode"] == 0

    @pytest.mark.asyncio
    async def test_set_one_way_mode(self, adapter_bybit, mock_ccxt_exchange):
        """set_position_mode('one_way') should call set_position_mode(False, ...)."""
        mock_ccxt_exchange.set_position_mode.return_value = {"retCode": 0}

        result = await adapter_bybit.set_position_mode("BTC/USDT:USDT", "one_way")

        mock_ccxt_exchange.set_position_mode.assert_called_once_with(False, "BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_set_position_mode_non_dict_result(self, adapter_bybit, mock_ccxt_exchange):
        """set_position_mode wraps non-dict results."""
        mock_ccxt_exchange.set_position_mode.return_value = True

        result = await adapter_bybit.set_position_mode("BTC/USDT:USDT", "one_way")
        assert result == {"result": True}


# ===========================================================================
# set_trading_stop
# ===========================================================================

class TestSetTradingStop:
    @pytest.mark.asyncio
    async def test_set_trading_stop_bybit_tp_sl(self, adapter_bybit, mock_ccxt_exchange):
        """set_trading_stop on Bybit should call privatePostV5PositionTradingStop."""
        mock_ccxt_exchange.privatePostV5PositionTradingStop.return_value = {
            "retCode": 0, "retMsg": "OK"
        }

        result = await adapter_bybit.set_trading_stop(
            "BTC/USDT:USDT",
            take_profit=Decimal("55000"),
            stop_loss=Decimal("48000"),
        )

        assert result["retCode"] == 0
        call_args = mock_ccxt_exchange.privatePostV5PositionTradingStop.call_args[0][0]
        assert call_args["category"] == "linear"
        assert call_args["symbol"] == "BTCUSDT"
        # positionIdx and tpslMode are now required Bybit V5 params
        assert call_args["positionIdx"] == 0
        assert call_args["tpslMode"] == "Full"
        # Values are rounded to instrument tick precision (fallback 4dp)
        assert Decimal(call_args["takeProfit"]) == Decimal("55000")
        assert Decimal(call_args["stopLoss"]) == Decimal("48000")

    @pytest.mark.asyncio
    async def test_set_trading_stop_trailing(self, adapter_bybit, mock_ccxt_exchange):
        """set_trading_stop with trailing_stop should pass trailingStop param."""
        mock_ccxt_exchange.privatePostV5PositionTradingStop.return_value = {"retCode": 0}

        result = await adapter_bybit.set_trading_stop(
            "ETH/USDT:USDT",
            trailing_stop=Decimal("50"),
        )

        call_args = mock_ccxt_exchange.privatePostV5PositionTradingStop.call_args[0][0]
        assert Decimal(call_args["trailingStop"]) == Decimal("50")

    @pytest.mark.asyncio
    async def test_set_trading_stop_no_params_raises(self, adapter_bybit):
        """set_trading_stop with no TP/SL/trailing should raise ExchangeError."""
        with pytest.raises(ExchangeError, match="requires at least one"):
            await adapter_bybit.set_trading_stop("BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_set_trading_stop_unsupported_exchange(self, adapter_binance):
        """set_trading_stop on non-Bybit exchanges should raise ExchangeError."""
        with pytest.raises(ExchangeError, match="does not support"):
            await adapter_binance.set_trading_stop(
                "BTC/USDT",
                take_profit=Decimal("55000"),
            )


# ===========================================================================
# get_closed_pnl
# ===========================================================================

class TestGetClosedPnl:
    @pytest.mark.asyncio
    async def test_get_closed_pnl_bybit_native(self, adapter_bybit, mock_ccxt_exchange):
        """get_closed_pnl on Bybit should use native V5 endpoint."""
        mock_ccxt_exchange.privateGetV5PositionClosedPnl.return_value = {
            "result": {
                "list": [
                    {
                        "side": "Buy",
                        "qty": "0.01",
                        "avgEntryPrice": "50000",
                        "avgExitPrice": "51000",
                        "closedPnl": "10",
                        "fillCount": 2,
                        "leverage": "10",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700003600000",
                        "orderId": "order-1",
                    },
                ]
            }
        }

        records = await adapter_bybit.get_closed_pnl("BTC/USDT:USDT", limit=10)

        assert len(records) == 1
        assert records[0]["symbol"] == "BTC/USDT:USDT"
        assert records[0]["side"] == "Buy"
        assert records[0]["closed_pnl"] == "10"
        assert records[0]["entry_price"] == "50000"
        assert records[0]["exit_price"] == "51000"

    @pytest.mark.asyncio
    async def test_get_closed_pnl_fallback(self, adapter_binance, mock_ccxt_exchange):
        """get_closed_pnl on Binance should fall back to fetch_my_trades."""
        # Remove native Bybit endpoint
        del mock_ccxt_exchange.privateGetV5PositionClosedPnl

        mock_ccxt_exchange.fetch_my_trades.return_value = [
            {
                "id": "trade-1",
                "order": "order-1",
                "symbol": "BTC/USDT",
                "side": "buy",
                "amount": 0.01,
                "price": 50000,
                "fee": {"cost": 0.5, "currency": "USDT"},
                "timestamp": 1700000000000,
            },
        ]

        records = await adapter_binance.get_closed_pnl("BTC/USDT", limit=5)

        assert len(records) == 1
        assert records[0]["side"] == "buy"
        assert records[0]["trade_id"] == "trade-1"
        mock_ccxt_exchange.fetch_my_trades.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_closed_pnl_empty(self, adapter_bybit, mock_ccxt_exchange):
        """get_closed_pnl returns empty list when no records."""
        mock_ccxt_exchange.privateGetV5PositionClosedPnl.return_value = {
            "result": {"list": []}
        }

        records = await adapter_bybit.get_closed_pnl("BTC/USDT:USDT")
        assert records == []


# ===========================================================================
# Symbol conversion
# ===========================================================================

class TestSymbolConversion:
    def test_to_bybit_symbol_standard(self, adapter_bybit):
        """Convert unified symbol to Bybit format."""
        assert adapter_bybit._to_bybit_symbol("BTC/USDT:USDT") == "BTCUSDT"

    def test_to_bybit_symbol_no_settle(self, adapter_bybit):
        """Convert spot symbol (no settle currency) to Bybit format."""
        assert adapter_bybit._to_bybit_symbol("BTC/USDT") == "BTCUSDT"

    def test_to_bybit_symbol_eth(self, adapter_bybit):
        """Convert ETH pair to Bybit format."""
        assert adapter_bybit._to_bybit_symbol("ETH/USDT:USDT") == "ETHUSDT"

    def test_to_bybit_symbol_sol(self, adapter_bybit):
        """Convert SOL pair to Bybit format."""
        assert adapter_bybit._to_bybit_symbol("SOL/USDT:USDT") == "SOLUSDT"


# ===========================================================================
# Error handling
# ===========================================================================

class TestV5ErrorHandling:
    @pytest.mark.asyncio
    async def test_amend_order_exchange_error(self, adapter_bybit, mock_ccxt_exchange):
        """amend_order wraps CCXT exceptions as ExchangeError."""
        mock_ccxt_exchange.edit_order.side_effect = Exception("Network timeout")

        with pytest.raises(ExchangeError, match="Network timeout"):
            await adapter_bybit.amend_order(
                "order-1", "BTC/USDT:USDT", price=Decimal("50000")
            )

    @pytest.mark.asyncio
    async def test_set_leverage_error(self, adapter_bybit, mock_ccxt_exchange):
        """set_leverage wraps exceptions as ExchangeError."""
        mock_ccxt_exchange.set_leverage.side_effect = Exception("Invalid leverage")

        with pytest.raises(ExchangeError, match="Invalid leverage"):
            await adapter_bybit.set_leverage("BTC/USDT:USDT", 200)

    @pytest.mark.asyncio
    async def test_batch_submit_error(self, adapter_bybit, mock_ccxt_exchange):
        """batch_submit_orders wraps exceptions as ExchangeError."""
        mock_ccxt_exchange.create_orders.side_effect = Exception("Batch failed")

        with pytest.raises(ExchangeError, match="Batch failed"):
            await adapter_bybit.batch_submit_orders([_make_intent()])

    @pytest.mark.asyncio
    async def test_get_closed_pnl_error(self, adapter_bybit, mock_ccxt_exchange):
        """get_closed_pnl wraps exceptions as ExchangeError."""
        mock_ccxt_exchange.privateGetV5PositionClosedPnl.side_effect = Exception("API error")

        with pytest.raises(ExchangeError, match="API error"):
            await adapter_bybit.get_closed_pnl("BTC/USDT:USDT")
