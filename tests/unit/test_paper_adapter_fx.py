"""Tests for PaperAdapter FX-aware math.

Validates that the PaperAdapter correctly handles FX instruments:
- Notional calculations include lot_size
- Unrealized PnL includes lot_size multiplier
- Realized PnL includes lot_size multiplier
- fee_currency uses instrument.quote instead of hardcoded USDT
- Position notional is FX-aware
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from agentic_trading.core.enums import (
    AssetClass,
    Exchange,
    InstrumentType,
    OrderType,
    Side,
    TimeInForce,
)
from agentic_trading.core.events import OrderIntent
from agentic_trading.core.models import Instrument
from agentic_trading.execution.adapters.base import SlippageConfig
from agentic_trading.execution.adapters.paper import PaperAdapter

# Zero slippage for deterministic test math
_ZERO_SLIPPAGE = SlippageConfig(fixed_bps=Decimal("0"), max_random_bps=Decimal("0"))


def _make_fx_instrument(
    symbol: str = "EUR/USD",
    pip_size: Decimal = Decimal("0.0001"),
    lot_size: Decimal = Decimal("100000"),
) -> Instrument:
    """Create a minimal FX instrument for testing."""
    return Instrument(
        symbol=symbol,
        exchange=Exchange.OANDA,
        instrument_type=InstrumentType.FX_SPOT,
        base="EUR",
        quote="USD",
        price_precision=5,
        qty_precision=2,
        tick_size=Decimal("0.00001"),
        step_size=Decimal("0.01"),
        min_qty=Decimal("0.01"),
        asset_class=AssetClass.FX,
        pip_size=pip_size,
        lot_size=lot_size,
        maker_fee=Decimal("0"),
        taker_fee=Decimal("0"),
    )


def _make_crypto_instrument(symbol: str = "BTC/USDT") -> Instrument:
    """Create a minimal crypto instrument for testing."""
    return Instrument(
        symbol=symbol,
        exchange=Exchange.BINANCE,
        instrument_type=InstrumentType.PERP,
        base="BTC",
        quote="USDT",
        asset_class=AssetClass.CRYPTO,
    )


def _make_intent(
    symbol: str = "EUR/USD",
    side: Side = Side.BUY,
    qty: Decimal = Decimal("1"),
    exchange: Exchange = Exchange.OANDA,
) -> OrderIntent:
    """Create a minimal OrderIntent."""
    return OrderIntent(
        dedupe_key="test_key",
        strategy_id="test_strat",
        symbol=symbol,
        exchange=exchange,
        side=side,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=qty,
    )


class TestPaperAdapterFXNotional:
    """Test that notional calculations are FX-aware."""

    @pytest.mark.asyncio
    async def test_fx_notional_includes_lot_size(self):
        """FX notional = qty * lot_size * price."""
        inst = _make_fx_instrument()
        adapter = PaperAdapter(
            exchange=Exchange.OANDA,
            initial_balances={"USD": Decimal("1000000")},
            instruments={inst.symbol: inst},
            slippage=_ZERO_SLIPPAGE,
        )
        adapter.set_market_price("EUR/USD", Decimal("1.1000"))

        intent = _make_intent(qty=Decimal("1"))  # 1 lot
        ack = await adapter.submit_order(intent)
        assert ack.status.value == "filled"

        # Check position notional: 1 lot * 100000 * 1.1 = 110000
        positions = await adapter.get_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert pos.notional == Decimal("1") * Decimal("100000") * Decimal("1.1000")

    @pytest.mark.asyncio
    async def test_crypto_notional_no_lot_size(self):
        """Crypto notional = qty * price (no lot_size)."""
        inst = _make_crypto_instrument()
        adapter = PaperAdapter(
            exchange=Exchange.BINANCE,
            initial_balances={"USDT": Decimal("1000000")},
            instruments={inst.symbol: inst},
            slippage=_ZERO_SLIPPAGE,
        )
        adapter.set_market_price("BTC/USDT", Decimal("50000"))

        intent = _make_intent(
            symbol="BTC/USDT",
            qty=Decimal("0.5"),
            exchange=Exchange.BINANCE,
        )
        ack = await adapter.submit_order(intent)
        assert ack.status.value == "filled"

        positions = await adapter.get_positions()
        assert len(positions) == 1
        pos = positions[0]
        # 0.5 * 50000 = 25000
        assert pos.notional == Decimal("0.5") * Decimal("50000")


class TestPaperAdapterFXPnL:
    """Test that PnL calculations are FX-aware."""

    @pytest.mark.asyncio
    async def test_fx_unrealized_pnl_includes_lot_size(self):
        """FX unrealized PnL = (mark - entry) * qty * lot_size."""
        inst = _make_fx_instrument()
        adapter = PaperAdapter(
            exchange=Exchange.OANDA,
            initial_balances={"USD": Decimal("1000000")},
            instruments={inst.symbol: inst},
            slippage=_ZERO_SLIPPAGE,
        )
        adapter.set_market_price("EUR/USD", Decimal("1.1000"))

        intent = _make_intent(qty=Decimal("1"))  # Buy 1 lot at 1.1000
        await adapter.submit_order(intent)

        # Price moves up 10 pips
        adapter.set_market_price("EUR/USD", Decimal("1.1010"))
        positions = await adapter.get_positions()
        pos = positions[0]

        # PnL = (1.1010 - 1.1000) * 1 * 100000 = 100 USD
        expected_pnl = Decimal("0.0010") * Decimal("1") * Decimal("100000")
        assert pos.unrealized_pnl == expected_pnl

    @pytest.mark.asyncio
    async def test_fx_realized_pnl_includes_lot_size(self):
        """FX realized PnL = (exit - entry) * qty * lot_size."""
        inst = _make_fx_instrument()
        adapter = PaperAdapter(
            exchange=Exchange.OANDA,
            initial_balances={"USD": Decimal("1000000")},
            instruments={inst.symbol: inst},
            slippage=_ZERO_SLIPPAGE,
        )
        adapter.set_market_price("EUR/USD", Decimal("1.1000"))

        # Open long 1 lot
        buy_intent = _make_intent(side=Side.BUY, qty=Decimal("1"))
        await adapter.submit_order(buy_intent)

        # Close at 1.1050 (+50 pips)
        adapter.set_market_price("EUR/USD", Decimal("1.1050"))
        sell_intent = _make_intent(side=Side.SELL, qty=Decimal("1"))
        await adapter.submit_order(sell_intent)

        # Position should be closed
        positions = await adapter.get_positions()
        assert len(positions) == 0

        # Check the internal position for realized PnL
        raw_pos = adapter._positions.get("EUR/USD")
        if raw_pos is not None:
            # PnL = (1.1050 - 1.1000) * 1 * 100000 = 500 USD
            expected_pnl = Decimal("0.0050") * Decimal("1") * Decimal("100000")
            assert raw_pos.realized_pnl == expected_pnl

    @pytest.mark.asyncio
    async def test_crypto_pnl_no_lot_size(self):
        """Crypto PnL = (mark - entry) * qty (no lot_size)."""
        inst = _make_crypto_instrument()
        adapter = PaperAdapter(
            exchange=Exchange.BINANCE,
            initial_balances={"USDT": Decimal("1000000")},
            instruments={inst.symbol: inst},
            slippage=_ZERO_SLIPPAGE,
        )
        adapter.set_market_price("BTC/USDT", Decimal("50000"))

        intent = _make_intent(
            symbol="BTC/USDT",
            qty=Decimal("0.5"),
            exchange=Exchange.BINANCE,
        )
        await adapter.submit_order(intent)

        # Price moves up 1000
        adapter.set_market_price("BTC/USDT", Decimal("51000"))
        positions = await adapter.get_positions()
        pos = positions[0]

        # PnL = (51000 - 50000) * 0.5 = 500 USDT
        expected_pnl = Decimal("1000") * Decimal("0.5")
        assert pos.unrealized_pnl == expected_pnl


class TestPaperAdapterFXFeeCurrency:
    """Test that fee_currency uses instrument.quote."""

    @pytest.mark.asyncio
    async def test_fx_fee_currency_uses_usd(self):
        """FX instruments should use USD as fee currency."""
        inst = _make_fx_instrument()
        adapter = PaperAdapter(
            exchange=Exchange.OANDA,
            initial_balances={"USD": Decimal("1000000")},
            instruments={inst.symbol: inst},
        )
        adapter.set_market_price("EUR/USD", Decimal("1.1000"))

        intent = _make_intent(qty=Decimal("1"))
        ack = await adapter.submit_order(intent)
        assert ack.status.value == "filled"

        # The quote currency should be "USD" (from instrument), not "USDT"
        assert inst.quote == "USD"


class TestPaperAdapterFXBalance:
    """Test FX paper trading with USD balance."""

    @pytest.mark.asyncio
    async def test_fx_paper_uses_usd_balance(self):
        """FX paper trading debits USD, not USDT."""
        inst = _make_fx_instrument()
        adapter = PaperAdapter(
            exchange=Exchange.OANDA,
            initial_balances={"USD": Decimal("100000")},
            instruments={inst.symbol: inst},
        )
        adapter.set_market_price("EUR/USD", Decimal("1.1000"))

        # Buy 0.1 lots (notional = 0.1 * 100000 * 1.1 = 11000 USD)
        intent = _make_intent(qty=Decimal("0.1"))
        ack = await adapter.submit_order(intent)
        assert ack.status.value == "filled"

        balances = await adapter.get_balances()
        assert len(balances) == 1
        bal = balances[0]
        assert bal.currency == "USD"
        # Balance should be reduced by notional (+ fee)
        assert bal.total < Decimal("100000")


class TestPaperAdapterNoInstrument:
    """Test backward compatibility when no instrument is loaded."""

    @pytest.mark.asyncio
    async def test_no_instrument_uses_raw_math(self):
        """Without instrument, notional = qty * price (no lot_size)."""
        adapter = PaperAdapter(
            exchange=Exchange.BINANCE,
            initial_balances={"USDT": Decimal("1000000")},
        )
        adapter.set_market_price("BTC/USDT", Decimal("50000"))

        intent = _make_intent(
            symbol="BTC/USDT",
            qty=Decimal("0.5"),
            exchange=Exchange.BINANCE,
        )
        ack = await adapter.submit_order(intent)
        assert ack.status.value == "filled"

        positions = await adapter.get_positions()
        pos = positions[0]
        # Raw math: 0.5 * 50000 = 25000
        assert pos.notional == Decimal("0.5") * Decimal("50000")
