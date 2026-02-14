"""Test position sizing methods."""

from decimal import Decimal

import pytest

from agentic_trading.portfolio.sizing import (
    fixed_fractional_size,
    kelly_size,
    scaled_entry_size,
    stop_loss_based_size,
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


class TestStopLossBasedSize:
    def test_produces_positive_qty(self):
        qty = stop_loss_based_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entry_price=67000.0,
            stop_loss_price=65500.0,
        )
        assert qty > Decimal("0")

    def test_known_calculation(self):
        # 100k * 1% / |100 - 90| = 1000 / 10 = 100 units
        qty = stop_loss_based_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entry_price=100.0,
            stop_loss_price=90.0,
        )
        assert qty == Decimal("100.0")

    def test_larger_stop_distance_reduces_size(self):
        qty_tight = stop_loss_based_size(
            capital=100_000, risk_per_trade_pct=0.01,
            entry_price=100.0, stop_loss_price=98.0,
        )
        qty_wide = stop_loss_based_size(
            capital=100_000, risk_per_trade_pct=0.01,
            entry_price=100.0, stop_loss_price=90.0,
        )
        assert qty_wide < qty_tight

    def test_short_direction(self):
        # entry=90, stop=100 → short → |90-100| = 10
        qty = stop_loss_based_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entry_price=90.0,
            stop_loss_price=100.0,
        )
        assert qty == Decimal("100.0")

    def test_zero_entry_returns_zero(self):
        qty = stop_loss_based_size(
            capital=100_000, risk_per_trade_pct=0.01,
            entry_price=0, stop_loss_price=90.0,
        )
        assert qty == Decimal("0")

    def test_entry_equals_stop_returns_zero(self):
        qty = stop_loss_based_size(
            capital=100_000, risk_per_trade_pct=0.01,
            entry_price=100.0, stop_loss_price=100.0,
        )
        assert qty == Decimal("0")

    def test_with_instrument_rounding(self, sample_instrument):
        qty = stop_loss_based_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entry_price=67000.0,
            stop_loss_price=65500.0,
            instrument=sample_instrument,
        )
        assert qty >= sample_instrument.min_qty
        remainder = qty % sample_instrument.step_size
        assert remainder == Decimal("0")


class TestScaledEntrySize:
    def test_two_entries_equal_split(self):
        result = scaled_entry_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entries=[(100.0, 0.5), (95.0, 0.5)],
            stop_loss_price=90.0,
        )
        assert len(result) == 2
        for qty, price in result:
            assert qty > Decimal("0")
            assert price in (100.0, 95.0)

    def test_unequal_allocations(self):
        result = scaled_entry_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entries=[(100.0, 0.7), (95.0, 0.3)],
            stop_loss_price=90.0,
        )
        assert len(result) == 2
        # First entry should get more units
        assert result[0][0] > result[1][0]

    def test_three_entries(self):
        result = scaled_entry_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entries=[(100.0, 0.4), (95.0, 0.35), (90.5, 0.25)],
            stop_loss_price=88.0,
        )
        assert len(result) == 3

    def test_single_entry_matches_stop_loss_based(self):
        scaled = scaled_entry_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entries=[(100.0, 1.0)],
            stop_loss_price=90.0,
        )
        single = stop_loss_based_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entry_price=100.0,
            stop_loss_price=90.0,
        )
        assert len(scaled) == 1
        assert abs(float(scaled[0][0]) - float(single)) < 0.01

    def test_empty_entries_returns_empty(self):
        result = scaled_entry_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entries=[],
            stop_loss_price=90.0,
        )
        assert result == []

    def test_allocations_normalised(self):
        # Weights sum to 10, not 1.0 — should still work via normalisation
        result = scaled_entry_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entries=[(100.0, 5.0), (95.0, 5.0)],
            stop_loss_price=90.0,
        )
        assert len(result) == 2

    def test_with_instrument_rounding(self, sample_instrument):
        result = scaled_entry_size(
            capital=100_000,
            risk_per_trade_pct=0.01,
            entries=[(67000.0, 0.5), (65500.0, 0.5)],
            stop_loss_price=63000.0,
            instrument=sample_instrument,
        )
        for qty, _ in result:
            assert qty >= sample_instrument.min_qty
