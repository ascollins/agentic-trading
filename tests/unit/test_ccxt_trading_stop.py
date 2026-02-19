"""Tests for CCXTAdapter.set_trading_stop — Bybit V5 parameter correctness.

Verifies:
- Required Bybit params ``positionIdx`` and ``tpslMode`` are included.
- TP/SL/trailing values are correctly converted to string params.
- Trigger types (tpTriggerBy, slTriggerBy) are always present.
- Raises ExchangeError when no TP/SL/trailing is provided.
- Symbol conversion from CCXT unified format to Bybit format.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.core.enums import Exchange
from agentic_trading.core.errors import ExchangeError


@pytest.fixture
def adapter():
    """Build a CCXTAdapter with mocked internals."""
    from agentic_trading.execution.adapters.ccxt_adapter import CCXTAdapter

    # We need to construct the adapter without calling __init__ fully,
    # since it requires real CCXT exchange setup.  Instead, mock.
    with patch.object(CCXTAdapter, "__init__", lambda self, **kw: None):
        a = CCXTAdapter.__new__(CCXTAdapter)
        a._ccxt = AsyncMock()
        a._exchange_name = "bybit"
        a._exchange_enum = Exchange.BYBIT
        a._default_type = "swap"
        a._markets_loaded = True
        # Mock _ensure_markets as no-op
        a._ensure_markets = AsyncMock()
        # Mock _to_bybit_symbol
        a._to_bybit_symbol = lambda sym: sym.replace("/", "").split(":")[0]
        # Mock _to_swap_symbol (for tick-size lookup)
        a._to_swap_symbol = lambda sym: sym + ":USDT" if ":" not in sym else sym
        # Mock _wrap_error
        a._wrap_error = lambda exc: ExchangeError(str(exc))
        # Mock market() to return tick-size precision (4 decimals by default)
        a._ccxt.market = MagicMock(return_value={
            "precision": {"price": 4, "amount": 6},
        })
    return a


class TestBybitTradingStopParams:
    """Verify the Bybit V5 trading-stop request params."""

    @pytest.mark.asyncio
    async def test_includes_position_idx(self, adapter):
        """positionIdx=0 must be sent for one-way mode."""
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "BTC/USDT", take_profit=Decimal("100000"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        assert params["positionIdx"] == 0

    @pytest.mark.asyncio
    async def test_includes_tpsl_mode(self, adapter):
        """tpslMode=Full must be sent."""
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "BTC/USDT", stop_loss=Decimal("90000"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        assert params["tpslMode"] == "Full"

    @pytest.mark.asyncio
    async def test_includes_category_linear(self, adapter):
        """category=linear for perpetual futures."""
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "ETH/USDT", take_profit=Decimal("5000"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        assert params["category"] == "linear"

    @pytest.mark.asyncio
    async def test_symbol_converted_to_bybit_format(self, adapter):
        """Symbol like 'XRP/USDT' becomes 'XRPUSDT'."""
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "XRP/USDT", take_profit=Decimal("3"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        assert params["symbol"] == "XRPUSDT"

    @pytest.mark.asyncio
    async def test_trigger_types_always_present(self, adapter):
        """tpTriggerBy and slTriggerBy must always be 'MarkPrice' for USDT perps.

        MarkPrice avoids wick-triggered stop hunts that can occur with LastPrice.
        """
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "BTC/USDT", take_profit=Decimal("100000"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        assert params["tpTriggerBy"] == "MarkPrice"
        assert params["slTriggerBy"] == "MarkPrice"

    @pytest.mark.asyncio
    async def test_tp_sl_values_as_strings(self, adapter):
        """TP/SL/trailing values must be stringified and rounded for Bybit API."""
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("105000.50"),
            stop_loss=Decimal("95000.25"),
            trailing_stop=Decimal("500"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        # Values are rounded to instrument tick precision (4dp in fixture)
        assert params["takeProfit"] == "105000.5000"
        assert params["stopLoss"] == "95000.2500"
        assert params["trailingStop"] == "500.0000"

    @pytest.mark.asyncio
    async def test_partial_params_tp_only(self, adapter):
        """Only TP provided — stopLoss and trailingStop absent from params."""
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "BTC/USDT", take_profit=Decimal("110000"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        assert "takeProfit" in params
        assert "stopLoss" not in params
        assert "trailingStop" not in params

    @pytest.mark.asyncio
    async def test_partial_params_sl_only(self, adapter):
        """Only SL provided — takeProfit and trailingStop absent."""
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "BTC/USDT", stop_loss=Decimal("90000"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        assert "stopLoss" in params
        assert "takeProfit" not in params
        assert "trailingStop" not in params

    @pytest.mark.asyncio
    async def test_raises_when_no_values(self, adapter):
        """Raises ExchangeError when no TP/SL/trailing provided."""
        with pytest.raises(ExchangeError, match="requires at least one"):
            await adapter.set_trading_stop("BTC/USDT")

    @pytest.mark.asyncio
    async def test_full_request_param_set(self, adapter):
        """Verify complete set of params for a full TP+SL+trailing request."""
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "SOL/USDT",
            take_profit=Decimal("200"),
            stop_loss=Decimal("150"),
            trailing_stop=Decimal("5"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        expected_keys = {
            "category", "symbol", "positionIdx", "tpslMode",
            "tpTriggerBy", "slTriggerBy",
            "takeProfit", "stopLoss", "trailingStop",
        }
        assert set(params.keys()) == expected_keys
        assert params["category"] == "linear"
        assert params["symbol"] == "SOLUSDT"
        assert params["positionIdx"] == 0
        assert params["tpslMode"] == "Full"

    @pytest.mark.asyncio
    async def test_active_price_included_with_trailing(self, adapter):
        """activePrice is sent alongside trailingStop when both are provided.

        Bybit V5 uses activePrice to defer trailing stop activation until
        the market reaches a specific level (breakeven activation pattern).
        """
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "BTC/USDT",
            trailing_stop=Decimal("500"),
            active_price=Decimal("97000"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        assert params["trailingStop"] == "500.0000"
        assert params["activePrice"] == "97000.0000"

    @pytest.mark.asyncio
    async def test_active_price_not_sent_without_trailing(self, adapter):
        """activePrice is NOT sent when there is no trailingStop.

        Sending activePrice without a trailing stop would be meaningless noise
        in the Bybit API request.
        """
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("110000"),
            active_price=Decimal("97000"),  # ignored when no trailing_stop
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        assert "takeProfit" in params
        assert "activePrice" not in params


class TestPriceRounding:
    """Verify TP/SL prices are rounded to instrument tick size."""

    @pytest.mark.asyncio
    async def test_rounds_to_tick_precision(self, adapter):
        """Prices with excess decimals are rounded to instrument precision."""
        # Market returns 4 decimal places of precision (default fixture)
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("100123.456789"),
            stop_loss=Decimal("95000.123456"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        assert params["takeProfit"] == "100123.4568"  # rounded to 4 dp
        assert params["stopLoss"] == "95000.1235"  # rounded to 4 dp

    @pytest.mark.asyncio
    async def test_rounds_to_2dp_for_xrp(self, adapter):
        """XRP/USDT has 4 decimal places on Bybit."""
        adapter._ccxt.market = MagicMock(return_value={
            "precision": {"price": 4, "amount": 1},
        })
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "XRP/USDT",
            take_profit=Decimal("1.5678901"),
            stop_loss=Decimal("1.234567"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        assert params["takeProfit"] == "1.5679"
        assert params["stopLoss"] == "1.2346"

    @pytest.mark.asyncio
    async def test_fallback_precision_on_market_error(self, adapter):
        """If market() throws, fallback to 4 decimal places."""
        adapter._ccxt.market = MagicMock(side_effect=Exception("not found"))
        adapter._ccxt.privatePostV5PositionTradingStop = AsyncMock(
            return_value={"retCode": 0, "retMsg": "OK"}
        )
        await adapter.set_trading_stop(
            "UNKNOWN/USDT",
            take_profit=Decimal("50.123456789"),
        )
        call_args = adapter._ccxt.privatePostV5PositionTradingStop.call_args
        params = call_args[0][0]
        # Fallback precision = 4
        assert params["takeProfit"] == "50.1235"
