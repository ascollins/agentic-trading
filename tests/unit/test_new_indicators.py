"""Tests for new indicator functions added for CMT strategy support.

Covers: Stochastic, OBV, Keltner Channel, BBW, Fibonacci, ROC.
"""

from __future__ import annotations

import numpy as np

from agentic_trading.features.indicators import (
    compute_bbw,
    compute_fibonacci_extensions,
    compute_fibonacci_levels,
    compute_keltner,
    compute_obv,
    compute_roc,
    compute_stochastic,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n: int = 100, start: float = 100.0, trend: float = 0.1) -> np.ndarray:
    """Generate a simple trending price series."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.5, n)
    return np.array([start + i * trend + noise[i] for i in range(n)])


def _make_ohlcv(n: int = 100, start: float = 100.0):
    """Generate OHLCV arrays."""
    rng = np.random.default_rng(42)
    closes = _make_prices(n, start)
    highs = closes + np.abs(rng.normal(0.5, 0.3, n))
    lows = closes - np.abs(rng.normal(0.5, 0.3, n))
    opens = closes + rng.normal(0, 0.2, n)
    volumes = np.abs(rng.normal(1000, 200, n))
    return opens, highs, lows, closes, volumes


# ===========================================================================
# Stochastic Oscillator
# ===========================================================================


class TestStochastic:
    def test_output_shape(self):
        _, highs, lows, closes, _ = _make_ohlcv(100)
        k, d = compute_stochastic(highs, lows, closes, 14, 3)
        assert len(k) == 100
        assert len(d) == 100

    def test_k_range(self):
        _, highs, lows, closes, _ = _make_ohlcv(100)
        k, _ = compute_stochastic(highs, lows, closes, 14, 3)
        valid_k = k[~np.isnan(k)]
        assert all(0 <= v <= 100 for v in valid_k)

    def test_d_range(self):
        _, highs, lows, closes, _ = _make_ohlcv(100)
        _, d = compute_stochastic(highs, lows, closes, 14, 3)
        valid_d = d[~np.isnan(d)]
        assert all(0 <= v <= 100 for v in valid_d)

    def test_short_input(self):
        _, highs, lows, closes, _ = _make_ohlcv(5)
        k, d = compute_stochastic(highs, lows, closes, 14, 3)
        assert len(k) == 5
        assert all(np.isnan(v) for v in k)

    def test_warmup_period(self):
        _, highs, lows, closes, _ = _make_ohlcv(100)
        k, _ = compute_stochastic(highs, lows, closes, 14, 3)
        # First 13 bars should be NaN (period - 1)
        assert all(np.isnan(v) for v in k[:13])
        assert not np.isnan(k[13])

    def test_flat_price_gives_50(self):
        n = 30
        flat = np.ones(n) * 100.0
        k, _ = compute_stochastic(flat, flat, flat, 14, 3)
        valid_k = k[~np.isnan(k)]
        assert all(v == 50.0 for v in valid_k)


# ===========================================================================
# On-Balance Volume (OBV)
# ===========================================================================


class TestOBV:
    def test_output_shape(self):
        closes = np.array([10.0, 11.0, 10.5, 12.0, 11.5])
        volumes = np.array([100.0, 150.0, 120.0, 200.0, 80.0])
        obv = compute_obv(closes, volumes)
        assert len(obv) == 5

    def test_first_value_zero(self):
        closes = np.array([10.0, 11.0])
        volumes = np.array([100.0, 150.0])
        obv = compute_obv(closes, volumes)
        assert obv[0] == 0.0

    def test_up_close_adds_volume(self):
        closes = np.array([10.0, 11.0])
        volumes = np.array([100.0, 150.0])
        obv = compute_obv(closes, volumes)
        assert obv[1] == 150.0

    def test_down_close_subtracts_volume(self):
        closes = np.array([10.0, 9.0])
        volumes = np.array([100.0, 150.0])
        obv = compute_obv(closes, volumes)
        assert obv[1] == -150.0

    def test_flat_close_no_change(self):
        closes = np.array([10.0, 10.0])
        volumes = np.array([100.0, 150.0])
        obv = compute_obv(closes, volumes)
        assert obv[1] == 0.0

    def test_cumulative_calculation(self):
        closes = np.array([10.0, 11.0, 10.5, 12.0])
        volumes = np.array([100.0, 200.0, 150.0, 300.0])
        obv = compute_obv(closes, volumes)
        # Up: +200, Down: -150, Up: +300
        assert obv[1] == 200.0
        assert obv[2] == 200.0 - 150.0
        assert obv[3] == 200.0 - 150.0 + 300.0


# ===========================================================================
# Keltner Channel
# ===========================================================================


class TestKeltnerChannel:
    def test_output_shape(self):
        _, highs, lows, closes, _ = _make_ohlcv(100)
        upper, middle, lower = compute_keltner(highs, lows, closes)
        assert len(upper) == 100
        assert len(middle) == 100
        assert len(lower) == 100

    def test_upper_above_middle_above_lower(self):
        _, highs, lows, closes, _ = _make_ohlcv(100)
        upper, middle, lower = compute_keltner(highs, lows, closes)
        for i in range(len(upper)):
            if not any(np.isnan(v) for v in (upper[i], middle[i], lower[i])):
                assert upper[i] >= middle[i]
                assert middle[i] >= lower[i]

    def test_middle_is_ema(self):
        _, highs, lows, closes, _ = _make_ohlcv(100)
        from agentic_trading.features.indicators import compute_ema
        upper, middle, lower = compute_keltner(highs, lows, closes, ema_period=20)
        ema = compute_ema(closes, 20)
        np.testing.assert_array_almost_equal(middle, ema)


# ===========================================================================
# Bollinger Band Width
# ===========================================================================


class TestBBW:
    def test_output_shape(self):
        closes = _make_prices(100)
        bbw = compute_bbw(closes, 20, 2.0)
        assert len(bbw) == 100

    def test_positive_values(self):
        closes = _make_prices(100)
        bbw = compute_bbw(closes, 20, 2.0)
        valid = bbw[~np.isnan(bbw)]
        assert all(v > 0 for v in valid)

    def test_warmup_nans(self):
        closes = _make_prices(100)
        bbw = compute_bbw(closes, 20, 2.0)
        assert all(np.isnan(v) for v in bbw[:19])

    def test_flat_price_near_zero(self):
        closes = np.ones(50) * 100.0
        bbw = compute_bbw(closes, 20, 2.0)
        valid = bbw[~np.isnan(bbw)]
        assert all(v < 0.001 for v in valid)


# ===========================================================================
# Fibonacci Levels
# ===========================================================================


class TestFibonacciLevels:
    def test_retracement_levels(self):
        levels = compute_fibonacci_levels(100.0, 50.0)
        assert levels["fib_0"] == 100.0
        assert levels["fib_1000"] == 50.0
        assert levels["fib_500"] == 75.0
        assert abs(levels["fib_618"] - (100.0 - 0.618 * 50.0)) < 0.01
        assert abs(levels["fib_382"] - (100.0 - 0.382 * 50.0)) < 0.01

    def test_all_levels_between_high_low(self):
        levels = compute_fibonacci_levels(200.0, 100.0)
        for name, val in levels.items():
            assert 100.0 <= val <= 200.0, f"{name}={val} out of range"

    def test_extensions(self):
        ext = compute_fibonacci_extensions(100.0, 50.0, 70.0)
        assert ext["ext_1000"] == 70.0 + 50.0  # 120
        assert abs(ext["ext_1618"] - (70.0 + 1.618 * 50.0)) < 0.01


# ===========================================================================
# Rate of Change
# ===========================================================================


class TestROC:
    def test_output_shape(self):
        closes = _make_prices(50)
        roc = compute_roc(closes, 12)
        assert len(roc) == 50

    def test_warmup_nans(self):
        closes = _make_prices(50)
        roc = compute_roc(closes, 12)
        assert all(np.isnan(v) for v in roc[:12])

    def test_flat_price_zero_roc(self):
        closes = np.ones(30) * 100.0
        roc = compute_roc(closes, 12)
        valid = roc[~np.isnan(roc)]
        assert all(abs(v) < 0.001 for v in valid)

    def test_positive_trend_positive_roc(self):
        closes = np.array([100.0 + i for i in range(30)])
        roc = compute_roc(closes, 10)
        valid = roc[~np.isnan(roc)]
        assert all(v > 0 for v in valid)

    def test_short_input(self):
        closes = _make_prices(5)
        roc = compute_roc(closes, 12)
        assert all(np.isnan(v) for v in roc)
