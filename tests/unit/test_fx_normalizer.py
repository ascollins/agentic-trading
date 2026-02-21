"""Tests for FX normalization utilities."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentic_trading.core.enums import (
    AssetClass,
    Exchange,
    InstrumentType,
    QtyUnit,
)
from agentic_trading.core.fx_normalizer import (
    is_session_open,
    normalize_order_qty,
    notional_usd,
    pip_value_in_account_ccy,
    rollover_cost,
    round_to_tick,
    spread_in_pips,
)
from agentic_trading.core.models import Instrument, TradingSession


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_eurusd() -> Instrument:
    return Instrument(
        symbol="EUR/USD",
        exchange=Exchange.OANDA,
        instrument_type=InstrumentType.FX_SPOT,
        base="EUR",
        quote="USD",
        asset_class=AssetClass.FX,
        tick_size=Decimal("0.00001"),
        step_size=Decimal("0.01"),
        min_qty=Decimal("0.01"),
        pip_size=Decimal("0.0001"),
        pip_value_per_lot=Decimal("10"),
        lot_size=Decimal("100000"),
        weekend_close=True,
        trading_sessions=[
            TradingSession(name="london", open_utc="08:00", close_utc="17:00"),
            TradingSession(name="new_york", open_utc="13:00", close_utc="22:00"),
        ],
    )


def _make_usdjpy() -> Instrument:
    return Instrument(
        symbol="USD/JPY",
        exchange=Exchange.OANDA,
        instrument_type=InstrumentType.FX_SPOT,
        base="USD",
        quote="JPY",
        asset_class=AssetClass.FX,
        tick_size=Decimal("0.001"),
        step_size=Decimal("0.01"),
        min_qty=Decimal("0.01"),
        pip_size=Decimal("0.01"),
        pip_value_per_lot=None,
        lot_size=Decimal("100000"),
        weekend_close=True,
        trading_sessions=[
            TradingSession(name="tokyo", open_utc="00:00", close_utc="09:00"),
        ],
    )


def _make_crypto() -> Instrument:
    return Instrument(
        symbol="BTC/USDT",
        exchange=Exchange.BYBIT,
        instrument_type=InstrumentType.PERP,
        base="BTC",
        quote="USDT",
        tick_size=Decimal("0.1"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
    )


# ---------------------------------------------------------------------------
# spread_in_pips
# ---------------------------------------------------------------------------


class TestSpreadInPips:
    def test_eurusd_spread(self):
        inst = _make_eurusd()
        result = spread_in_pips(
            Decimal("1.08450"), Decimal("1.08470"), inst
        )
        assert result == Decimal("2.0")

    def test_zero_spread(self):
        inst = _make_eurusd()
        result = spread_in_pips(
            Decimal("1.0850"), Decimal("1.0850"), inst
        )
        assert result == Decimal("0")

    def test_crypto_returns_zero(self):
        inst = _make_crypto()
        result = spread_in_pips(Decimal("50000"), Decimal("50001"), inst)
        assert result == Decimal("0")


# ---------------------------------------------------------------------------
# notional_usd
# ---------------------------------------------------------------------------


class TestNotionalUsd:
    def test_eurusd_quote_is_usd(self):
        inst = _make_eurusd()
        # 0.1 lots * 100,000 * 1.0850 = 10,850 USD
        result = notional_usd(Decimal("0.1"), Decimal("1.0850"), inst)
        assert result == Decimal("10850.00")

    def test_usdjpy_base_is_usd(self):
        inst = _make_usdjpy()
        # Base is USD: notional = 0.1 * 100,000 = 10,000 USD
        result = notional_usd(Decimal("0.1"), Decimal("150.00"), inst)
        assert result == Decimal("10000.0")


# ---------------------------------------------------------------------------
# pip_value_in_account_ccy
# ---------------------------------------------------------------------------


class TestPipValue:
    def test_eurusd_pip_value(self):
        inst = _make_eurusd()
        # pip_value_per_lot=10, 0.5 lots → 5 USD per pip
        result = pip_value_in_account_ccy(inst, Decimal("0.5"))
        assert result == Decimal("5.0")

    def test_usdjpy_pip_value_fallback(self):
        inst = _make_usdjpy()
        # No pip_value_per_lot; fallback: pip_size * lot_size * lots
        # 0.01 * 100,000 * 1.0 = 1000 JPY per pip
        result = pip_value_in_account_ccy(
            inst, Decimal("1.0"), account_ccy="JPY"
        )
        assert result == Decimal("1000.0")


# ---------------------------------------------------------------------------
# is_session_open
# ---------------------------------------------------------------------------


class TestIsSessionOpen:
    def test_london_session_open(self):
        inst = _make_eurusd()
        # Wednesday 10:30 UTC — London open
        assert is_session_open(inst, 10, 30, 3) is True

    def test_outside_all_sessions(self):
        inst = _make_eurusd()
        # Wednesday 02:00 UTC — no session
        assert is_session_open(inst, 2, 0, 3) is False

    def test_weekend_closed(self):
        inst = _make_eurusd()
        # Saturday 14:00 UTC — weekend
        assert is_session_open(inst, 14, 0, 6) is False

    def test_crypto_always_open(self):
        inst = _make_crypto()
        # Saturday 03:00 — crypto is 24/7
        assert is_session_open(inst, 3, 0, 6) is True

    def test_ny_session_open(self):
        inst = _make_eurusd()
        # Wednesday 20:00 UTC — NY open
        assert is_session_open(inst, 20, 0, 3) is True

    def test_overlap_open(self):
        inst = _make_eurusd()
        # Wednesday 14:00 — London + NY overlap
        assert is_session_open(inst, 14, 0, 3) is True


# ---------------------------------------------------------------------------
# rollover_cost
# ---------------------------------------------------------------------------


class TestRolloverCost:
    def test_rollover_disabled(self):
        inst = _make_eurusd()
        assert inst.rollover_enabled is False
        result = rollover_cost(inst, Decimal("1.0"), is_long=True)
        assert result == Decimal("0")

    def test_rollover_enabled(self):
        inst = _make_eurusd().model_copy(
            update={
                "rollover_enabled": True,
                "long_swap_rate": Decimal("-0.000015"),
                "short_swap_rate": Decimal("0.000005"),
            }
        )
        # Long: 1.0 lots * 100,000 * -0.000015 = -1.5
        result = rollover_cost(inst, Decimal("1.0"), is_long=True)
        assert result == Decimal("-1.5000")

    def test_rollover_short(self):
        inst = _make_eurusd().model_copy(
            update={
                "rollover_enabled": True,
                "long_swap_rate": Decimal("-0.000015"),
                "short_swap_rate": Decimal("0.000005"),
            }
        )
        # Short: 1.0 lots * 100,000 * 0.000005 = 0.5
        result = rollover_cost(inst, Decimal("1.0"), is_long=False)
        assert result == Decimal("0.5000")


# ---------------------------------------------------------------------------
# round_to_tick
# ---------------------------------------------------------------------------


class TestRoundToTick:
    def test_round_price(self):
        result = round_to_tick(Decimal("1.08567"), Decimal("0.00001"))
        assert result == Decimal("1.08567")

    def test_round_to_pip(self):
        result = round_to_tick(Decimal("1.08567"), Decimal("0.0001"))
        # 1.08567 / 0.0001 = 10856.7 → round to 10857 → 1.0857
        assert result == Decimal("1.0857")


# ---------------------------------------------------------------------------
# normalize_order_qty
# ---------------------------------------------------------------------------


class TestNormalizeOrderQty:
    def test_lots_to_rounded_base(self):
        inst = _make_eurusd()
        # 0.015 lots → 1500 base → round to step_size 0.01 → 0.02 lots
        # Actually: normalize_order_qty rounds the LOT quantity
        # 0.015 lots * 100000 = 1500 base, round_qty(1500) with step 0.01 → 1500.00
        result = normalize_order_qty(Decimal("0.015"), QtyUnit.LOTS, inst)
        # 0.015 * 100000 = 1500, round_qty(1500, step=0.01) = 1500.00
        assert result == Decimal("1500.00")

    def test_base_rounded(self):
        inst = _make_eurusd()
        result = normalize_order_qty(Decimal("1500.005"), QtyUnit.BASE, inst)
        assert result == Decimal("1500.00") or result == Decimal("1500.01")
