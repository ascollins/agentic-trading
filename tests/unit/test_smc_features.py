"""Tests for Smart Money Concepts (SMC) feature detection.

Tests swing detection, order blocks, fair value gaps, structure breaks,
and the integrated SMCFeatureComputer.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.models import Candle
from agentic_trading.features.smc.swing_detection import (
    StructureLabel,
    SwingPoint,
    SwingType,
    classify_structure,
    compute_swing_bias,
    detect_all_swings,
    detect_swing_highs,
    detect_swing_lows,
)
from agentic_trading.features.smc.order_blocks import (
    BlockType,
    GapType,
    detect_fvgs,
    detect_order_blocks,
)
from agentic_trading.features.smc.structure_breaks import (
    BreakDirection,
    BreakType,
    StructureBreak,
    detect_all_structure_breaks,
    detect_structure_breaks,
)
from agentic_trading.features.smc.computer import SMCFeatureComputer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(
    i: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
) -> Candle:
    """Create a Candle with minimal required fields."""
    return Candle(
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M1,
        timestamp=datetime(2024, 1, 1, 0, i, tzinfo=timezone.utc),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        quote_volume=0.0,
        trades=0,
        is_closed=True,
    )


def _uptrend_highs_lows(n: int = 50):
    """Generate a simple uptrend series for testing."""
    highs = np.array([100.0 + i * 2 + np.sin(i * 0.5) * 5 for i in range(n)])
    lows = highs - 3.0
    closes = (highs + lows) / 2
    opens = closes - 0.5
    return opens, highs, lows, closes


def _generate_swing_data():
    """Generate price data with clear swing highs and lows.

    Creates a zigzag pattern:
      Bars 0-9: rise to peak at bar 5 (high=110)
      Bars 10-19: fall to trough at bar 15 (low=90)
      Bars 20-29: rise to peak at bar 25 (high=115)
      Bars 30-39: fall to trough at bar 35 (low=85)
      Bars 40-49: rise to peak at bar 45 (high=120)
    """
    n = 50
    highs = np.full(n, 100.0)
    lows = np.full(n, 98.0)

    # First peak: index 5
    for i in range(11):
        highs[i] = 100.0 + (10.0 if i == 5 else min(i, 10 - i) * 1.5)
        lows[i] = highs[i] - 2.0

    # First trough: index 15
    for i in range(10, 21):
        offset = i - 10
        lows[i] = 100.0 - (10.0 if i == 15 else min(offset, 10 - offset) * 1.5)
        highs[i] = lows[i] + 2.0

    # Second peak: index 25
    for i in range(20, 31):
        offset = i - 20
        highs[i] = 100.0 + (15.0 if i == 25 else min(offset, 10 - offset) * 2.0)
        lows[i] = highs[i] - 2.0

    # Second trough: index 35
    for i in range(30, 41):
        offset = i - 30
        lows[i] = 100.0 - (15.0 if i == 35 else min(offset, 10 - offset) * 2.0)
        highs[i] = lows[i] + 2.0

    # Third peak: index 45
    for i in range(40, 50):
        offset = i - 40
        highs[i] = 100.0 + (20.0 if i == 45 else min(offset, 10 - offset) * 2.5)
        lows[i] = highs[i] - 2.0

    closes = (highs + lows) / 2
    opens = closes - 0.5
    volumes = np.full(n, 1000.0)

    return opens, highs, lows, closes, volumes


# ===========================================================================
# Swing Detection Tests
# ===========================================================================


class TestSwingDetection:
    """Tests for swing_detection module."""

    def test_detect_swing_highs_basic(self):
        """Swing highs detected at local maxima."""
        opens, highs, lows, closes, volumes = _generate_swing_data()
        swings = detect_swing_highs(highs, lookback=3)

        assert len(swings) > 0
        for s in swings:
            assert s.swing_type == SwingType.HIGH
            assert s.price > 0

    def test_detect_swing_lows_basic(self):
        """Swing lows detected at local minima."""
        opens, highs, lows, closes, volumes = _generate_swing_data()
        swings = detect_swing_lows(lows, lookback=3)

        assert len(swings) > 0
        for s in swings:
            assert s.swing_type == SwingType.LOW
            assert s.price > 0

    def test_detect_all_swings_sorted_by_index(self):
        """Combined swings should be sorted by index."""
        opens, highs, lows, closes, volumes = _generate_swing_data()
        swings = detect_all_swings(highs, lows, lookback=3)

        for i in range(1, len(swings)):
            assert swings[i].index >= swings[i - 1].index

    def test_detect_all_swings_alternating(self):
        """Ideally, swings should alternate between highs and lows."""
        opens, highs, lows, closes, volumes = _generate_swing_data()
        swings = detect_all_swings(highs, lows, lookback=3)

        # Should have both types
        types = set(s.swing_type for s in swings)
        assert SwingType.HIGH in types
        assert SwingType.LOW in types

    def test_swing_highs_with_flat_data_returns_empty(self):
        """Flat data has no swing points."""
        highs = np.full(20, 100.0)
        swings = detect_swing_highs(highs, lookback=3)
        assert len(swings) == 0

    def test_swing_lows_with_flat_data_returns_empty(self):
        """Flat data has no swing points."""
        lows = np.full(20, 100.0)
        swings = detect_swing_lows(lows, lookback=3)
        assert len(swings) == 0

    def test_swing_detection_respects_lookback(self):
        """Larger lookback should detect fewer (more significant) swings."""
        opens, highs, lows, closes, volumes = _generate_swing_data()

        swings_3 = detect_all_swings(highs, lows, lookback=3)
        swings_5 = detect_all_swings(highs, lows, lookback=5)

        # Larger lookback = fewer or equal number of swings
        assert len(swings_5) <= len(swings_3)

    def test_swing_detection_short_array(self):
        """Short arrays (< 2*lookback+1) should return empty."""
        highs = np.array([100.0, 105.0, 100.0])
        swings = detect_swing_highs(highs, lookback=5)
        assert len(swings) == 0


class TestStructureClassification:
    """Tests for structure classification (HH/HL/LH/LL)."""

    def test_classify_uptrend_structure(self):
        """Uptrend swings should classify as HH/HL."""
        swings = [
            SwingPoint(index=0, price=100.0, swing_type=SwingType.HIGH),
            SwingPoint(index=5, price=95.0, swing_type=SwingType.LOW),
            SwingPoint(index=10, price=110.0, swing_type=SwingType.HIGH),
            SwingPoint(index=15, price=105.0, swing_type=SwingType.LOW),
        ]

        classes = classify_structure(swings)
        assert len(classes) == 2

        # Second high (110) > first high (100) → HH
        assert classes[0].label == StructureLabel.HH
        # Second low (105) > first low (95) → HL
        assert classes[1].label == StructureLabel.HL

    def test_classify_downtrend_structure(self):
        """Downtrend swings should classify as LH/LL."""
        swings = [
            SwingPoint(index=0, price=110.0, swing_type=SwingType.HIGH),
            SwingPoint(index=5, price=105.0, swing_type=SwingType.LOW),
            SwingPoint(index=10, price=100.0, swing_type=SwingType.HIGH),
            SwingPoint(index=15, price=95.0, swing_type=SwingType.LOW),
        ]

        classes = classify_structure(swings)
        assert len(classes) == 2

        # Second high (100) < first high (110) → LH
        assert classes[0].label == StructureLabel.LH
        # Second low (95) < first low (105) → LL
        assert classes[1].label == StructureLabel.LL

    def test_classify_empty_returns_empty(self):
        """Single or no swings can't be classified."""
        assert classify_structure([]) == []
        assert classify_structure([
            SwingPoint(index=0, price=100.0, swing_type=SwingType.HIGH)
        ]) == []

    def test_compute_swing_bias_bullish(self):
        """Bullish classifications produce positive bias."""
        swings = [
            SwingPoint(index=0, price=100.0, swing_type=SwingType.HIGH),
            SwingPoint(index=5, price=95.0, swing_type=SwingType.LOW),
            SwingPoint(index=10, price=110.0, swing_type=SwingType.HIGH),
            SwingPoint(index=15, price=105.0, swing_type=SwingType.LOW),
        ]
        classes = classify_structure(swings)
        bias = compute_swing_bias(classes)
        assert bias > 0.0

    def test_compute_swing_bias_bearish(self):
        """Bearish classifications produce negative bias."""
        swings = [
            SwingPoint(index=0, price=110.0, swing_type=SwingType.HIGH),
            SwingPoint(index=5, price=105.0, swing_type=SwingType.LOW),
            SwingPoint(index=10, price=100.0, swing_type=SwingType.HIGH),
            SwingPoint(index=15, price=95.0, swing_type=SwingType.LOW),
        ]
        classes = classify_structure(swings)
        bias = compute_swing_bias(classes)
        assert bias < 0.0

    def test_compute_swing_bias_empty(self):
        """No classifications → zero bias."""
        assert compute_swing_bias([]) == 0.0


# ===========================================================================
# Order Block Tests
# ===========================================================================


class TestOrderBlocks:
    """Tests for order block detection."""

    def test_detect_bullish_order_block(self):
        """Bearish candle before bullish displacement = bullish OB."""
        n = 30
        opens = np.full(n, 100.0)
        highs = np.full(n, 102.0)
        lows = np.full(n, 98.0)
        closes = np.full(n, 100.0)
        volumes = np.full(n, 1000.0)
        atr = np.full(n, 2.0)

        # Bar 14: bearish candle (close < open)
        opens[14] = 101.0
        closes[14] = 99.0
        highs[14] = 101.5
        lows[14] = 98.5

        # Bar 15: strong bullish displacement (range > 2x ATR = 4)
        opens[15] = 99.0
        closes[15] = 106.0
        highs[15] = 106.5
        lows[15] = 98.5  # range = 8 > 4

        obs = detect_order_blocks(opens, highs, lows, closes, volumes, atr)

        bullish = [ob for ob in obs if ob.block_type == BlockType.BULLISH]
        assert len(bullish) >= 1
        # The OB should be at bar 14
        assert any(ob.index == 14 for ob in bullish)

    def test_detect_bearish_order_block(self):
        """Bullish candle before bearish displacement = bearish OB."""
        n = 30
        opens = np.full(n, 100.0)
        highs = np.full(n, 102.0)
        lows = np.full(n, 98.0)
        closes = np.full(n, 100.0)
        volumes = np.full(n, 1000.0)
        atr = np.full(n, 2.0)

        # Bar 14: bullish candle (close > open)
        opens[14] = 99.0
        closes[14] = 101.0
        highs[14] = 101.5
        lows[14] = 98.5

        # Bar 15: strong bearish displacement
        opens[15] = 101.0
        closes[15] = 94.0
        highs[15] = 101.5
        lows[15] = 93.5  # range = 8 > 4

        obs = detect_order_blocks(opens, highs, lows, closes, volumes, atr)

        bearish = [ob for ob in obs if ob.block_type == BlockType.BEARISH]
        assert len(bearish) >= 1
        assert any(ob.index == 14 for ob in bearish)

    def test_no_order_blocks_without_displacement(self):
        """Small candles don't create order blocks."""
        n = 30
        opens = np.full(n, 100.0)
        highs = np.full(n, 101.0)  # tiny range
        lows = np.full(n, 99.0)
        closes = np.full(n, 100.0)
        volumes = np.full(n, 1000.0)
        atr = np.full(n, 2.0)  # Range 2 < 2*ATR = 4

        obs = detect_order_blocks(opens, highs, lows, closes, volumes, atr)
        assert len(obs) == 0

    def test_short_data_returns_empty(self):
        """Very short arrays shouldn't crash."""
        obs = detect_order_blocks(
            np.array([100.0]), np.array([101.0]), np.array([99.0]),
            np.array([100.0]), np.array([1000.0]), np.array([2.0]),
        )
        assert obs == []


# ===========================================================================
# Fair Value Gap Tests
# ===========================================================================


class TestFairValueGaps:
    """Tests for FVG detection."""

    def test_detect_bullish_fvg(self):
        """Bullish FVG: candle_3.low > candle_1.high."""
        n = 20
        highs = np.full(n, 102.0)
        lows = np.full(n, 98.0)
        closes = np.full(n, 100.0)

        # Create bullish FVG at index 10 (middle candle)
        # candle_1 (index 9): high = 102
        # candle_2 (index 10): big up move
        # candle_3 (index 11): low = 104 > candle_1 high = 102
        highs[10] = 106.0
        lows[10] = 101.0
        closes[10] = 105.0
        highs[11] = 108.0
        lows[11] = 104.0  # 104 > 102 (candle 9 high)
        closes[11] = 107.0

        fvgs = detect_fvgs(highs, lows, closes)

        bullish = [g for g in fvgs if g.gap_type == GapType.BULLISH]
        assert len(bullish) >= 1
        gap = bullish[0]
        assert gap.gap_size > 0
        # gap_low should be candle_1 high, gap_high should be candle_3 low
        assert gap.gap_low == 102.0
        assert gap.gap_high == 104.0

    def test_detect_bearish_fvg(self):
        """Bearish FVG: candle_1.low > candle_3.high."""
        n = 20
        highs = np.full(n, 102.0)
        lows = np.full(n, 98.0)
        closes = np.full(n, 100.0)

        # Create bearish FVG at index 10 (middle candle)
        # candle_1 (index 9): low = 98
        # candle_2 (index 10): big down move
        # candle_3 (index 11): high = 96 < candle_1 low = 98
        highs[10] = 99.0
        lows[10] = 94.0
        closes[10] = 95.0
        highs[11] = 96.0  # 96 < 98 (candle 9 low)
        lows[11] = 93.0
        closes[11] = 94.0

        fvgs = detect_fvgs(highs, lows, closes)

        bearish = [g for g in fvgs if g.gap_type == GapType.BEARISH]
        assert len(bearish) >= 1
        gap = bearish[0]
        assert gap.gap_size > 0

    def test_no_fvg_in_flat_data(self):
        """Flat data should have no FVGs."""
        n = 30
        highs = np.full(n, 102.0)
        lows = np.full(n, 98.0)
        closes = np.full(n, 100.0)

        fvgs = detect_fvgs(highs, lows, closes)
        assert len(fvgs) == 0

    def test_fvg_fill_detection(self):
        """FVG should be marked as filled when price returns."""
        n = 25
        highs = np.full(n, 102.0)
        lows = np.full(n, 98.0)
        closes = np.full(n, 100.0)

        # Bullish FVG at index 10
        highs[10] = 106.0
        lows[10] = 101.0
        closes[10] = 105.0
        highs[11] = 108.0
        lows[11] = 104.0
        closes[11] = 107.0

        # Price fills the gap later (drops back into it)
        highs[15] = 105.0
        lows[15] = 101.0  # drops below gap_high (104)
        closes[15] = 102.0

        fvgs = detect_fvgs(highs, lows, closes)
        bullish = [g for g in fvgs if g.gap_type == GapType.BULLISH]

        if bullish:
            # Should be marked as filled
            assert bullish[0].is_filled is True

    def test_short_data_no_crash(self):
        """Short arrays shouldn't crash."""
        fvgs = detect_fvgs(np.array([100.0]), np.array([98.0]), np.array([99.0]))
        assert fvgs == []


# ===========================================================================
# Structure Break Tests
# ===========================================================================


class TestStructureBreaks:
    """Tests for BOS/CHoCH detection."""

    def test_detect_bullish_bos(self):
        """Close above swing high in uptrend = bullish BOS."""
        swings = [
            SwingPoint(index=0, price=100.0, swing_type=SwingType.HIGH),
            SwingPoint(index=5, price=95.0, swing_type=SwingType.LOW),
            SwingPoint(index=10, price=110.0, swing_type=SwingType.HIGH),
            SwingPoint(index=15, price=105.0, swing_type=SwingType.LOW),
            SwingPoint(index=20, price=115.0, swing_type=SwingType.HIGH),
            SwingPoint(index=25, price=110.0, swing_type=SwingType.LOW),
        ]

        # Price closes above the last swing high (115) at bar 27
        closes = np.full(30, 112.0)
        closes[27] = 118.0  # Breaks above 115

        breaks = detect_structure_breaks(swings, closes)
        bullish = [b for b in breaks if b.direction == BreakDirection.BULLISH]
        assert len(bullish) >= 1

    def test_detect_bearish_break(self):
        """Close below swing low = bearish break."""
        swings = [
            SwingPoint(index=0, price=110.0, swing_type=SwingType.HIGH),
            SwingPoint(index=5, price=105.0, swing_type=SwingType.LOW),
            SwingPoint(index=10, price=100.0, swing_type=SwingType.HIGH),
            SwingPoint(index=15, price=95.0, swing_type=SwingType.LOW),
        ]

        closes = np.full(30, 97.0)
        closes[20] = 92.0  # Breaks below 95

        breaks = detect_structure_breaks(swings, closes)
        bearish = [b for b in breaks if b.direction == BreakDirection.BEARISH]
        assert len(bearish) >= 1

    def test_detect_all_structure_breaks_returns_breaks(self):
        """Historical walk-through should detect structure breaks."""
        opens, highs, lows, closes, volumes = _generate_swing_data()
        swings = detect_all_swings(highs, lows, lookback=3)

        breaks = detect_all_structure_breaks(swings, closes)
        # With zigzag data, should find some structure breaks
        # (exact count depends on data shape)
        assert isinstance(breaks, list)
        for b in breaks:
            assert isinstance(b.break_type, BreakType)
            assert isinstance(b.direction, BreakDirection)

    def test_insufficient_swings_returns_empty(self):
        """Too few swings should return empty."""
        swings = [
            SwingPoint(index=0, price=100.0, swing_type=SwingType.HIGH),
        ]
        closes = np.array([100.0, 105.0])
        breaks = detect_structure_breaks(swings, closes)
        assert breaks == []

    def test_break_type_consistency(self):
        """Break type must be either BOS or CHoCH."""
        opens, highs, lows, closes, volumes = _generate_swing_data()
        swings = detect_all_swings(highs, lows, lookback=3)
        breaks = detect_all_structure_breaks(swings, closes)

        for b in breaks:
            assert b.break_type in (BreakType.BOS, BreakType.CHOCH)


# ===========================================================================
# SMCFeatureComputer Integration Tests
# ===========================================================================


class TestSMCFeatureComputer:
    """Tests for the integrated SMC feature computer."""

    @pytest.fixture
    def computer(self):
        return SMCFeatureComputer(
            swing_lookback=3,
            displacement_mult=2.0,
            min_candles=20,
        )

    def _make_candles(self, n: int = 100) -> list[Candle]:
        """Generate n candles with a simple uptrend + noise."""
        candles = []
        for i in range(n):
            base = 100.0 + i * 0.5 + np.sin(i * 0.3) * 3
            o = base
            h = base + np.random.uniform(1, 3)
            l = base - np.random.uniform(1, 3)
            c = base + np.random.uniform(-1, 1)
            candles.append(_make_candle(i % 60, o, h, l, c, volume=1000.0 + i * 10))
        return candles

    def test_compute_returns_all_feature_keys(self, computer):
        """Computer should return all expected SMC feature keys."""
        np.random.seed(42)
        candles = self._make_candles(100)
        features = computer.compute(candles)

        expected_keys = [
            "smc_swing_bias",
            "smc_swing_count",
            "smc_hh_count",
            "smc_hl_count",
            "smc_lh_count",
            "smc_ll_count",
            "smc_ob_count_bullish",
            "smc_ob_count_bearish",
            "smc_ob_unmitigated_bullish",
            "smc_ob_unmitigated_bearish",
            "smc_nearest_demand_distance",
            "smc_nearest_supply_distance",
            "smc_fvg_count_bullish",
            "smc_fvg_count_bearish",
            "smc_fvg_count_total",
            "smc_bos_bullish",
            "smc_bos_bearish",
            "smc_choch_bullish",
            "smc_choch_bearish",
            "smc_last_break_direction",
            "smc_last_break_is_choch",
            "smc_confluence_score",
        ]

        for key in expected_keys:
            assert key in features, f"Missing feature key: {key}"
            assert isinstance(features[key], float), f"{key} should be float"

    def test_compute_returns_empty_for_short_data(self, computer):
        """Insufficient data should return zeroed features."""
        candles = self._make_candles(5)
        features = computer.compute(candles)

        # All features should be 0.0
        for key, value in features.items():
            assert value == 0.0, f"{key} should be 0.0 for short data"

    def test_feature_values_are_finite(self, computer):
        """No feature should be NaN or infinite."""
        np.random.seed(123)
        candles = self._make_candles(200)
        features = computer.compute(candles)

        for key, value in features.items():
            assert np.isfinite(value), f"{key}={value} is not finite"

    def test_swing_bias_bounded(self, computer):
        """Swing bias should be in [-1, +1]."""
        np.random.seed(42)
        candles = self._make_candles(100)
        features = computer.compute(candles)

        assert -1.0 <= features["smc_swing_bias"] <= 1.0

    def test_confluence_score_bounded(self, computer):
        """Confluence score should be in [0, 14]."""
        np.random.seed(42)
        candles = self._make_candles(100)
        features = computer.compute(candles)

        assert 0.0 <= features["smc_confluence_score"] <= 14.0

    def test_counts_non_negative(self, computer):
        """All count features should be >= 0."""
        np.random.seed(42)
        candles = self._make_candles(100)
        features = computer.compute(candles)

        count_keys = [
            "smc_swing_count",
            "smc_hh_count", "smc_hl_count", "smc_lh_count", "smc_ll_count",
            "smc_ob_count_bullish", "smc_ob_count_bearish",
            "smc_ob_unmitigated_bullish", "smc_ob_unmitigated_bearish",
            "smc_fvg_count_bullish", "smc_fvg_count_bearish",
            "smc_fvg_count_total",
            "smc_bos_bullish", "smc_bos_bearish",
            "smc_choch_bullish", "smc_choch_bearish",
        ]

        for key in count_keys:
            assert features[key] >= 0.0, f"{key}={features[key]} should be >= 0"

    def test_last_break_direction_values(self, computer):
        """Last break direction should be -1, 0, or +1."""
        np.random.seed(42)
        candles = self._make_candles(100)
        features = computer.compute(candles)

        assert features["smc_last_break_direction"] in (-1.0, 0.0, 1.0)

    def test_last_break_is_choch_values(self, computer):
        """Last break CHoCH flag should be 0 or 1."""
        np.random.seed(42)
        candles = self._make_candles(100)
        features = computer.compute(candles)

        assert features["smc_last_break_is_choch"] in (0.0, 1.0)

    def test_empty_features_matches_compute_keys(self, computer):
        """Empty features should have same keys as computed features."""
        empty = computer._empty_features()
        np.random.seed(42)
        computed = computer.compute(self._make_candles(100))

        assert set(empty.keys()) == set(computed.keys())


# ===========================================================================
# ATR Computation Tests
# ===========================================================================


class TestSMCATR:
    """Tests for the internal ATR computation in SMCFeatureComputer."""

    def test_atr_computation(self):
        """ATR should be computed correctly."""
        highs = np.array([102.0] * 20)
        lows = np.array([98.0] * 20)
        closes = np.array([100.0] * 20)

        atr = SMCFeatureComputer._compute_atr(highs, lows, closes, period=14)

        # For constant range of 4.0, ATR should converge to 4.0
        # (first period values will be NaN)
        valid = atr[~np.isnan(atr)]
        assert len(valid) > 0
        assert abs(valid[-1] - 4.0) < 0.1

    def test_atr_short_data(self):
        """ATR with insufficient data should be all NaN."""
        highs = np.array([102.0, 103.0, 101.0])
        lows = np.array([98.0, 99.0, 97.0])
        closes = np.array([100.0, 101.0, 99.0])

        atr = SMCFeatureComputer._compute_atr(highs, lows, closes, period=14)
        assert all(np.isnan(atr))
