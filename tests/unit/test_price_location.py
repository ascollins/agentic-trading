"""Tests for premium / discount zone classification."""

from __future__ import annotations

import pytest

from agentic_trading.features.smc.price_location import (
    PriceLocation,
    PriceZone,
    classify_price_location,
    compute_dealing_range,
)
from agentic_trading.features.smc.swing_detection import SwingPoint, SwingType


def _make_swing_high(index: int, price: float) -> SwingPoint:
    return SwingPoint(index=index, price=price, swing_type=SwingType.HIGH)


def _make_swing_low(index: int, price: float) -> SwingPoint:
    return SwingPoint(index=index, price=price, swing_type=SwingType.LOW)


class TestComputeDealingRange:
    """Tests for compute_dealing_range."""

    def test_basic_range_from_swings(self):
        """Most recent swing high and low form the range."""
        swings = [
            _make_swing_low(5, 100.0),
            _make_swing_high(10, 200.0),
        ]
        result = compute_dealing_range(swings)
        assert result is not None
        assert result == (200.0, 100.0)

    def test_uses_most_recent_swings(self):
        """Uses the most recent swing of each type."""
        swings = [
            _make_swing_low(1, 90.0),
            _make_swing_high(5, 150.0),
            _make_swing_low(10, 100.0),
            _make_swing_high(15, 200.0),
        ]
        result = compute_dealing_range(swings)
        assert result is not None
        assert result == (200.0, 100.0)

    def test_requires_minimum_range_size(self):
        """Small ranges are filtered by ATR."""
        swings = [
            _make_swing_low(5, 99.5),
            _make_swing_high(10, 100.5),
        ]
        # Range = 1.0, ATR = 2.0, min_range_atr = 2.0 -> threshold = 4.0
        result = compute_dealing_range(swings, min_range_atr=2.0, atr_value=2.0)
        assert result is None

    def test_no_swings_returns_none(self):
        """Empty swing list returns None."""
        assert compute_dealing_range([]) is None

    def test_single_swing_returns_none(self):
        """Single swing cannot form a range."""
        assert compute_dealing_range([_make_swing_high(5, 100.0)]) is None

    def test_only_same_type_returns_none(self):
        """Two swings of the same type cannot form a range."""
        swings = [
            _make_swing_high(5, 100.0),
            _make_swing_high(10, 110.0),
        ]
        assert compute_dealing_range(swings) is None

    def test_high_below_low_returns_none(self):
        """When swing high price < swing low price, returns None."""
        swings = [
            _make_swing_high(10, 95.0),
            _make_swing_low(5, 100.0),
        ]
        assert compute_dealing_range(swings) is None

    def test_no_atr_accepts_any_range(self):
        """Without ATR, any non-zero range is accepted."""
        swings = [
            _make_swing_low(5, 99.9),
            _make_swing_high(10, 100.1),
        ]
        result = compute_dealing_range(swings, atr_value=None)
        assert result is not None


class TestClassifyPriceLocation:
    """Tests for classify_price_location."""

    def test_deep_discount(self):
        """Price well below equilibrium is DEEP_DISCOUNT."""
        loc = classify_price_location(current_price=102.0, dealing_range_high=200.0, dealing_range_low=100.0)
        assert loc.zone == PriceZone.DEEP_DISCOUNT
        assert loc.range_position_pct < 0.214

    def test_discount(self):
        """Price below equilibrium but above 21.4% is DISCOUNT."""
        loc = classify_price_location(current_price=130.0, dealing_range_high=200.0, dealing_range_low=100.0)
        assert loc.zone == PriceZone.DISCOUNT
        assert 0.214 <= loc.range_position_pct < 0.45

    def test_equilibrium(self):
        """Price near 50% is EQUILIBRIUM."""
        loc = classify_price_location(current_price=150.0, dealing_range_high=200.0, dealing_range_low=100.0)
        assert loc.zone == PriceZone.EQUILIBRIUM
        assert 0.45 <= loc.range_position_pct <= 0.55

    def test_premium(self):
        """Price above equilibrium but below 78.6% is PREMIUM."""
        loc = classify_price_location(current_price=170.0, dealing_range_high=200.0, dealing_range_low=100.0)
        assert loc.zone == PriceZone.PREMIUM
        assert 0.55 < loc.range_position_pct <= 0.786

    def test_deep_premium(self):
        """Price well above equilibrium is DEEP_PREMIUM."""
        loc = classify_price_location(current_price=195.0, dealing_range_high=200.0, dealing_range_low=100.0)
        assert loc.zone == PriceZone.DEEP_PREMIUM
        assert loc.range_position_pct > 0.786

    def test_ote_zone_for_longs(self):
        """Price at 61.8-78.6% retrace from high is in OTE for longs."""
        # 61.8% retrace from 200 to 100 = 200 - 0.618*100 = 138.2
        # 78.6% retrace from 200 to 100 = 200 - 0.786*100 = 121.4
        # OTE long zone: 121.4 to 138.2
        loc = classify_price_location(current_price=130.0, dealing_range_high=200.0, dealing_range_low=100.0)
        assert loc.in_ote is True
        # Price is in discount → long OTE
        assert loc.range_position_pct < 0.5

    def test_ote_zone_for_shorts(self):
        """Price at 61.8-78.6% from low is in OTE for shorts."""
        # OTE short zone: range_position 0.618 to 0.786
        # = 100 + 0.618*100 = 161.8 to 100 + 0.786*100 = 178.6
        loc = classify_price_location(current_price=170.0, dealing_range_high=200.0, dealing_range_low=100.0)
        assert loc.in_ote is True
        assert loc.range_position_pct > 0.5

    def test_deviation_sign_and_magnitude(self):
        """Deviation is positive above eq, negative below."""
        loc_above = classify_price_location(170.0, 200.0, 100.0)
        assert loc_above.deviation_pct > 0

        loc_below = classify_price_location(130.0, 200.0, 100.0)
        assert loc_below.deviation_pct < 0

        loc_eq = classify_price_location(150.0, 200.0, 100.0)
        assert abs(loc_eq.deviation_pct) < 0.01

    def test_range_position_boundaries(self):
        """0.0 at low, 1.0 at high."""
        loc_low = classify_price_location(100.0, 200.0, 100.0)
        assert loc_low.range_position_pct == pytest.approx(0.0)

        loc_high = classify_price_location(200.0, 200.0, 100.0)
        assert loc_high.range_position_pct == pytest.approx(1.0)

    def test_price_outside_range(self):
        """Price beyond range edges still gets a classification."""
        loc_below = classify_price_location(80.0, 200.0, 100.0)
        assert loc_below.zone == PriceZone.DEEP_DISCOUNT
        assert loc_below.range_position_pct < 0.0

        loc_above = classify_price_location(250.0, 200.0, 100.0)
        assert loc_above.zone == PriceZone.DEEP_PREMIUM
        assert loc_above.range_position_pct > 1.0

    def test_equilibrium_calculation(self):
        """Equilibrium is exactly 50% of the range."""
        loc = classify_price_location(150.0, 200.0, 100.0)
        assert loc.equilibrium == pytest.approx(150.0)
        assert loc.dealing_range_size == pytest.approx(100.0)

    def test_degenerate_range_returns_equilibrium(self):
        """Zero-size range returns EQUILIBRIUM zone."""
        loc = classify_price_location(100.0, 100.0, 100.0)
        assert loc.zone == PriceZone.EQUILIBRIUM
        assert loc.dealing_range_size == 0.0

    def test_zone_encoding_values(self):
        """PriceZone enum values match expected strings."""
        assert PriceZone.DEEP_DISCOUNT.value == "deep_discount"
        assert PriceZone.DISCOUNT.value == "discount"
        assert PriceZone.EQUILIBRIUM.value == "equilibrium"
        assert PriceZone.PREMIUM.value == "premium"
        assert PriceZone.DEEP_PREMIUM.value == "deep_premium"
