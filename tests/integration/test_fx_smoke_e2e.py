"""End-to-end smoke test for the FX paper trading pipeline.

Simulates the full bootstrap path from main.py:
1. Build G10 instruments from hardcoded definitions
2. Create PaperAdapter with USD balance and loaded instruments
3. Set market prices
4. Submit FX orders
5. Verify positions, PnL, balances

This validates the entire wiring is correct without needing
an actual broker connection.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from agentic_trading.core.enums import (
    AssetClass,
    Exchange,
    OrderType,
    Side,
    TimeInForce,
)
from agentic_trading.core.events import OrderIntent
from agentic_trading.core.fx_instruments import build_fx_instruments
from agentic_trading.execution.adapters.base import SlippageConfig
from agentic_trading.execution.adapters.paper import PaperAdapter

_ZERO_SLIPPAGE = SlippageConfig(fixed_bps=Decimal("0"), max_random_bps=Decimal("0"))


def _intent(
    symbol: str,
    side: Side,
    qty: Decimal,
    exchange: Exchange = Exchange.OANDA,
) -> OrderIntent:
    return OrderIntent(
        dedupe_key=f"smoke_{symbol}_{side.value}",
        strategy_id="smoke_test",
        symbol=symbol,
        exchange=exchange,
        side=side,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=qty,
    )


class TestFXSmokeE2E:
    """Full pipeline smoke test."""

    @pytest.mark.asyncio
    async def test_bootstrap_instruments_and_trade(self):
        """Simulate main.py bootstrap: build instruments, create adapter, trade."""
        # Step 1: Build instruments (same path as main.py)
        symbols = ["EUR/USD", "GBP/USD", "USD/JPY"]
        instruments = build_fx_instruments(symbols, exchange=Exchange.OANDA)
        assert len(instruments) == 3

        # Step 2: Create PaperAdapter with USD balance (same as main.py)
        adapter = PaperAdapter(
            exchange=Exchange.OANDA,
            initial_balances={"USD": Decimal("100000")},
            instruments=instruments,
            slippage=_ZERO_SLIPPAGE,
        )

        # Step 3: Load instruments into adapter (same as main.py)
        for inst in instruments.values():
            adapter.load_instrument(inst)

        # Step 4: Set market prices (in production, these come from FXFeedManager)
        adapter.set_market_price("EUR/USD", Decimal("1.0850"))
        adapter.set_market_price("GBP/USD", Decimal("1.2650"))
        adapter.set_market_price("USD/JPY", Decimal("149.50"))

        # Step 5: Buy 0.1 lots of EUR/USD
        intent = _intent("EUR/USD", Side.BUY, Decimal("0.1"))
        ack = await adapter.submit_order(intent)
        assert ack.status.value == "filled"

        # Step 6: Verify position
        positions = await adapter.get_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert pos.symbol == "EUR/USD"
        assert pos.qty == Decimal("0.1")

        # Notional = 0.1 lots * 100000 * 1.0850 = 10850 USD
        expected_notional = Decimal("0.1") * Decimal("100000") * Decimal("1.0850")
        assert pos.notional == expected_notional

        # Step 7: Verify USD balance decreased
        balances = await adapter.get_balances()
        assert len(balances) == 1
        assert balances[0].currency == "USD"
        assert balances[0].total < Decimal("100000")

    @pytest.mark.asyncio
    async def test_multi_pair_portfolio(self):
        """Trade multiple FX pairs simultaneously.

        Note: USD/JPY has quote=JPY, so buying USD/JPY requires JPY balance.
        We use EUR/USD + GBP/USD (both quote=USD) and sell USD/JPY
        (selling requires base currency margin, which the simplified
        adapter allows from USD).
        """
        instruments = build_fx_instruments(
            ["EUR/USD", "GBP/USD", "USD/JPY"],
            exchange=Exchange.OANDA,
        )
        adapter = PaperAdapter(
            exchange=Exchange.OANDA,
            initial_balances={"USD": Decimal("500000"), "JPY": Decimal("50000000")},
            instruments=instruments,
            slippage=_ZERO_SLIPPAGE,
        )
        for inst in instruments.values():
            adapter.load_instrument(inst)

        adapter.set_market_price("EUR/USD", Decimal("1.0850"))
        adapter.set_market_price("GBP/USD", Decimal("1.2650"))
        adapter.set_market_price("USD/JPY", Decimal("149.50"))

        # Open positions in 3 pairs
        await adapter.submit_order(
            _intent("EUR/USD", Side.BUY, Decimal("0.5"))
        )
        await adapter.submit_order(
            OrderIntent(
                dedupe_key="smoke_gbp",
                strategy_id="smoke_test",
                symbol="GBP/USD",
                exchange=Exchange.OANDA,
                side=Side.SELL,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                qty=Decimal("0.3"),
            )
        )
        await adapter.submit_order(
            OrderIntent(
                dedupe_key="smoke_jpy",
                strategy_id="smoke_test",
                symbol="USD/JPY",
                exchange=Exchange.OANDA,
                side=Side.BUY,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                qty=Decimal("0.2"),
            )
        )

        positions = await adapter.get_positions()
        assert len(positions) == 3

        symbols = {p.symbol for p in positions}
        assert symbols == {"EUR/USD", "GBP/USD", "USD/JPY"}

    @pytest.mark.asyncio
    async def test_pnl_after_price_move(self):
        """Verify PnL calculation after price movement."""
        instruments = build_fx_instruments(["EUR/USD"], exchange=Exchange.OANDA)
        adapter = PaperAdapter(
            exchange=Exchange.OANDA,
            initial_balances={"USD": Decimal("200000")},  # 1 lot = ~$108k notional
            instruments=instruments,
            slippage=_ZERO_SLIPPAGE,
        )
        for inst in instruments.values():
            adapter.load_instrument(inst)

        # Buy 1 lot at 1.0850
        adapter.set_market_price("EUR/USD", Decimal("1.0850"))
        await adapter.submit_order(
            _intent("EUR/USD", Side.BUY, Decimal("1"))
        )

        # Price rises 50 pips to 1.0900
        adapter.set_market_price("EUR/USD", Decimal("1.0900"))
        positions = await adapter.get_positions()
        pos = positions[0]

        # Unrealized PnL = (1.0900 - 1.0850) * 1 * 100000 = 500 USD
        expected_pnl = Decimal("0.0050") * Decimal("100000")
        assert pos.unrealized_pnl == expected_pnl

    @pytest.mark.asyncio
    async def test_instrument_properties_correct(self):
        """Verify instruments have correct FX-specific properties."""
        instruments = build_fx_instruments(
            ["EUR/USD", "USD/JPY"],
            exchange=Exchange.OANDA,
        )

        eurusd = instruments["EUR/USD"]
        assert eurusd.asset_class == AssetClass.FX
        assert eurusd.lot_size == Decimal("100000")
        assert eurusd.pip_size == Decimal("0.0001")
        assert eurusd.weekend_close is True
        assert eurusd.rollover_enabled is True
        assert eurusd.venue_symbol == "EUR_USD"
        assert eurusd.exchange == Exchange.OANDA
        assert len(eurusd.trading_sessions) > 0

        usdjpy = instruments["USD/JPY"]
        assert usdjpy.pip_size == Decimal("0.01")
        assert usdjpy.price_precision == 3
        assert usdjpy.venue_symbol == "USD_JPY"

    @pytest.mark.asyncio
    async def test_round_trip_trade(self):
        """Open and close a position, verify realized PnL."""
        instruments = build_fx_instruments(["GBP/USD"], exchange=Exchange.OANDA)
        adapter = PaperAdapter(
            exchange=Exchange.OANDA,
            initial_balances={"USD": Decimal("100000")},
            instruments=instruments,
            slippage=_ZERO_SLIPPAGE,
        )
        for inst in instruments.values():
            adapter.load_instrument(inst)

        # Open long 0.5 lots at 1.2650
        adapter.set_market_price("GBP/USD", Decimal("1.2650"))
        await adapter.submit_order(
            OrderIntent(
                dedupe_key="rt_open",
                strategy_id="smoke_test",
                symbol="GBP/USD",
                exchange=Exchange.OANDA,
                side=Side.BUY,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                qty=Decimal("0.5"),
            )
        )

        # Close at 1.2700 (+50 pips)
        adapter.set_market_price("GBP/USD", Decimal("1.2700"))
        await adapter.submit_order(
            OrderIntent(
                dedupe_key="rt_close",
                strategy_id="smoke_test",
                symbol="GBP/USD",
                exchange=Exchange.OANDA,
                side=Side.SELL,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                qty=Decimal("0.5"),
            )
        )

        # Position should be closed
        positions = await adapter.get_positions()
        assert len(positions) == 0

        # Balance should reflect profit:
        # PnL = (1.2700 - 1.2650) * 0.5 * 100000 = 250 USD
        balances = await adapter.get_balances()
        bal = balances[0]
        assert bal.currency == "USD"
        assert bal.total > Decimal("100000")  # profit added back
