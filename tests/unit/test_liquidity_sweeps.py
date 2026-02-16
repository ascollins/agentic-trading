"""Tests for liquidity sweep detection (BSL / SSL)."""

from __future__ import annotations

import numpy as np
import pytest

from agentic_trading.features.smc.liquidity_sweeps import (
    LiquiditySweep,
    SweepType,
    detect_liquidity_sweeps,
)
from agentic_trading.features.smc.swing_detection import SwingPoint, SwingType


def _make_swing_high(index: int, price: float) -> SwingPoint:
    return SwingPoint(index=index, price=price, swing_type=SwingType.HIGH)


def _make_swing_low(index: int, price: float) -> SwingPoint:
    return SwingPoint(index=index, price=price, swing_type=SwingType.LOW)


def _make_arrays(n: int, base: float = 100.0) -> tuple:
    """Create flat OHLC arrays at a base price."""
    opens = np.full(n, base)
    highs = np.full(n, base + 0.5)
    lows = np.full(n, base - 0.5)
    closes = np.full(n, base)
    return highs, lows, opens, closes


class TestLiquiditySweeps:
    """Tests for detect_liquidity_sweeps."""

    def test_detect_bsl_sweep_above_swing_high(self):
        """Bar wick above swing high, body closes below = BSL."""
        n = 20
        highs, lows, opens, closes = _make_arrays(n, base=100.0)
        swings = [_make_swing_high(5, 101.0)]

        # Bar 10: wick above 101.0, close below
        highs[10] = 101.5
        closes[10] = 100.5

        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, swings, min_penetration_atr=0.0,
        )
        assert len(sweeps) == 1
        assert sweeps[0].sweep_type == SweepType.BSL
        assert sweeps[0].swing_level == 101.0
        assert sweeps[0].index == 10
        assert sweeps[0].wick_price == 101.5
        assert sweeps[0].close_price == 100.5
        assert sweeps[0].penetration_pct > 0

    def test_detect_ssl_sweep_below_swing_low(self):
        """Bar wick below swing low, body closes above = SSL."""
        n = 20
        highs, lows, opens, closes = _make_arrays(n, base=100.0)
        swings = [_make_swing_low(5, 99.0)]

        # Bar 10: wick below 99.0, close above
        lows[10] = 98.5
        closes[10] = 99.5

        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, swings, min_penetration_atr=0.0,
        )
        assert len(sweeps) == 1
        assert sweeps[0].sweep_type == SweepType.SSL
        assert sweeps[0].swing_level == 99.0
        assert sweeps[0].index == 10

    def test_no_sweep_when_body_closes_beyond(self):
        """Close beyond the swing level = breakout, not sweep."""
        n = 20
        highs, lows, opens, closes = _make_arrays(n, base=100.0)
        swings = [_make_swing_high(5, 101.0)]

        # Bar 10: wick above AND close above = genuine breakout
        highs[10] = 102.0
        closes[10] = 101.5

        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, swings, min_penetration_atr=0.0,
        )
        assert len(sweeps) == 0

    def test_reversal_confirmation_flag(self):
        """Reversal confirmed when next bar closes opposite direction."""
        n = 20
        highs, lows, opens, closes = _make_arrays(n, base=100.0)
        swings = [_make_swing_high(5, 101.0)]

        # Bar 10: BSL sweep
        highs[10] = 101.5
        closes[10] = 100.5

        # Bar 11: closes lower = reversal confirmed
        closes[11] = 100.0

        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, swings, min_penetration_atr=0.0,
        )
        assert len(sweeps) == 1
        assert sweeps[0].reversal_confirmed is True

    def test_no_reversal_when_next_bar_continues(self):
        """No reversal when next bar continues in sweep direction."""
        n = 20
        highs, lows, opens, closes = _make_arrays(n, base=100.0)
        swings = [_make_swing_high(5, 101.0)]

        # Bar 10: BSL sweep
        highs[10] = 101.5
        closes[10] = 100.5

        # Bar 11: closes higher = no reversal
        closes[11] = 101.0

        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, swings, min_penetration_atr=0.0,
        )
        assert len(sweeps) == 1
        assert sweeps[0].reversal_confirmed is False

    def test_no_sweeps_in_flat_data(self):
        """Flat data with no swings returns empty."""
        n = 20
        highs, lows, opens, closes = _make_arrays(n, base=100.0)
        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, [], min_penetration_atr=0.0,
        )
        assert len(sweeps) == 0

    def test_max_age_bars_limits_search(self):
        """Sweeps beyond age window are excluded."""
        n = 30
        highs, lows, opens, closes = _make_arrays(n, base=100.0)
        swings = [_make_swing_high(2, 101.0)]

        # Bar 25 (23 bars after swing): would be a sweep
        highs[25] = 101.5
        closes[25] = 100.5

        # max_age_bars=10: should NOT find the sweep
        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, swings,
            max_age_bars=10, min_penetration_atr=0.0,
        )
        assert len(sweeps) == 0

        # max_age_bars=30: should find it
        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, swings,
            max_age_bars=30, min_penetration_atr=0.0,
        )
        assert len(sweeps) == 1

    def test_min_penetration_filter(self):
        """Tiny wicks below ATR threshold are excluded."""
        n = 20
        highs, lows, opens, closes = _make_arrays(n, base=100.0)
        atr = np.full(n, 2.0)
        swings = [_make_swing_high(5, 101.0)]

        # Bar 10: very tiny penetration (0.01 above swing)
        highs[10] = 101.01
        closes[10] = 100.5

        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, swings,
            min_penetration_atr=0.1, atr=atr,
        )
        # penetration=0.01, threshold=0.1*2.0=0.2 -> filtered
        assert len(sweeps) == 0

    def test_multiple_sweeps_only_first_per_level(self):
        """Only the first sweep per swing level is detected."""
        n = 30
        highs, lows, opens, closes = _make_arrays(n, base=100.0)
        swings = [_make_swing_high(5, 101.0)]

        # Two bars sweep above the same swing high
        highs[10] = 101.5
        closes[10] = 100.5
        highs[15] = 101.8
        closes[15] = 100.3

        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, swings, min_penetration_atr=0.0,
        )
        assert len(sweeps) == 1
        assert sweeps[0].index == 10  # First one

    def test_last_sweep_type_values(self):
        """SweepType enum values are BSL and SSL."""
        assert SweepType.BSL.value == "BSL"
        assert SweepType.SSL.value == "SSL"

    def test_empty_arrays_returns_empty(self):
        """Very short arrays return empty list."""
        sweeps = detect_liquidity_sweeps(
            np.array([100.0]), np.array([99.0]),
            np.array([99.5]), np.array([100.0]),
            [_make_swing_high(0, 101.0)],
        )
        assert len(sweeps) == 0

    def test_sorted_by_index(self):
        """Sweeps are returned sorted by index."""
        n = 30
        highs, lows, opens, closes = _make_arrays(n, base=100.0)
        swings = [
            _make_swing_low(3, 99.0),
            _make_swing_high(5, 101.0),
        ]

        # SSL at bar 15
        lows[15] = 98.5
        closes[15] = 99.5

        # BSL at bar 10
        highs[10] = 101.5
        closes[10] = 100.5

        sweeps = detect_liquidity_sweeps(
            highs, lows, opens, closes, swings, min_penetration_atr=0.0,
        )
        assert len(sweeps) == 2
        assert sweeps[0].index < sweeps[1].index
