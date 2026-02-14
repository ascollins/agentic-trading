"""Test position sizing methods."""

from decimal import Decimal

import pytest

from agentic_trading.portfolio.sizing import (
    fixed_fractional_size,
    kelly_size,
    volatility_adjusted_size,
)


class TestVolatilityAdjustedSize:
    def test_produces_positive_qty(self):
        qty = volatility_adjusted_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            atr=500.0,
            price=67000.0,
        )
        assert qty > Decimal("0")

    def test_higher_atr_reduces_size(self):
        qty_low = volatility_adjusted_size(
            capital=100_000, risk_per_trade_pct=0.01, atr=200.0, price=67000.0
        )
        qty_high = volatility_adjusted_size(
            capital=100_000, risk_per_trade_pct=0.01, atr=800.0, price=67000.0
        )
        assert qty_high < qty_low

    def test_zero_atr_returns_zero(self):
        qty = volatility_adjusted_size(
            capital=100_000, risk_per_trade_pct=0.01, atr=0, price=67000.0
        )
        assert qty == Decimal("0")

    def test_zero_price_returns_zero(self):
        qty = volatility_adjusted_size(
            capital=100_000, risk_per_trade_pct=0.01, atr=500.0, price=0
        )
        assert qty == Decimal("0")

    def test_with_instrument_rounding(self, sample_instrument):
        qty = volatility_adjusted_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            atr=500.0,
            price=67000.0,
            instrument=sample_instrument,
        )
        # Should be rounded to step_size (0.001)
        assert qty >= sample_instrument.min_qty
        # Check it is a multiple of step_size
        remainder = qty % sample_instrument.step_size
        assert remainder == Decimal("0")


class TestFixedFractionalSize:
    def test_produces_positive_qty(self):
        qty = fixed_fractional_size(
            capital=100_000, fraction=0.05, price=67000.0
        )
        assert qty > Decimal("0")

    def test_known_calculation(self):
        qty = fixed_fractional_size(
            capital=100_000, fraction=0.10, price=50000.0
        )
        # 100_000 * 0.10 / 50_000 = 0.2
        assert qty == Decimal("0.2")

    def test_zero_price_returns_zero(self):
        qty = fixed_fractional_size(capital=100_000, fraction=0.05, price=0)
        assert qty == Decimal("0")

    def test_with_instrument_rounding(self, sample_instrument):
        qty = fixed_fractional_size(
            capital=100_000,
            fraction=0.05,
            price=67000.0,
            instrument=sample_instrument,
        )
        assert qty >= sample_instrument.min_qty


class TestKellySize:
    def test_produces_positive_qty_with_edge(self):
        qty = kelly_size(
            capital=100_000,
            win_rate=0.6,
            avg_win=1.5,
            avg_loss=1.0,
            price=67000.0,
        )
        assert qty > Decimal("0")

    def test_no_edge_returns_zero(self):
        """win_rate=0.3 with avg_win/avg_loss=1 => negative Kelly => zero."""
        qty = kelly_size(
            capital=100_000,
            win_rate=0.3,
            avg_win=1.0,
            avg_loss=1.0,
            price=67000.0,
        )
        assert qty == Decimal("0")

    def test_zero_win_rate_returns_zero(self):
        qty = kelly_size(
            capital=100_000, win_rate=0, avg_win=1.0, avg_loss=1.0, price=67000.0
        )
        assert qty == Decimal("0")

    def test_kelly_is_fractional(self):
        """Default fractional Kelly (0.25) produces smaller size than full Kelly."""
        full = kelly_size(
            capital=100_000,
            win_rate=0.6,
            avg_win=2.0,
            avg_loss=1.0,
            price=67000.0,
            kelly_fraction=1.0,
        )
        quarter = kelly_size(
            capital=100_000,
            win_rate=0.6,
            avg_win=2.0,
            avg_loss=1.0,
            price=67000.0,
            kelly_fraction=0.25,
        )
        assert quarter < full

    def test_reasonable_range(self):
        """Kelly size should not exceed 20% of capital."""
        qty = kelly_size(
            capital=100_000,
            win_rate=0.9,
            avg_win=5.0,
            avg_loss=1.0,
            price=67000.0,
            kelly_fraction=1.0,
        )
        notional = float(qty) * 67000.0
        assert notional <= 100_000 * 0.20 + 1  # 20% cap + rounding tolerance
