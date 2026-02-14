"""Test technical indicator functions with known data."""

import numpy as np
import pytest

from agentic_trading.features.indicators import (
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_rsi,
    compute_sma,
)


class TestComputeEMA:
    def test_output_shape_matches_input(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = compute_ema(close, period=3)
        assert result.shape == close.shape

    def test_leading_nans(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = compute_ema(close, period=5)
        # First 4 values should be NaN (period - 1 = 4)
        assert np.isnan(result[0])
        assert np.isnan(result[3])
        # The period-th value and beyond should be computed
        assert not np.isnan(result[4])

    def test_ema_tracks_uptrend(self):
        close = np.arange(1.0, 21.0)
        result = compute_ema(close, period=5)
        # In a steady uptrend, EMA should be below the latest value
        valid = result[~np.isnan(result)]
        assert valid[-1] < close[-1]

    def test_short_array_returns_all_nans(self):
        close = np.array([1.0, 2.0])
        result = compute_ema(close, period=5)
        assert np.all(np.isnan(result))


class TestComputeRSI:
    def test_output_shape_matches_input(self):
        close = np.arange(1.0, 31.0)
        result = compute_rsi(close, period=14)
        assert result.shape == close.shape

    def test_leading_nans(self):
        close = np.arange(1.0, 31.0)
        result = compute_rsi(close, period=14)
        # First 14 values should be NaN
        assert np.isnan(result[0])
        assert not np.isnan(result[14])

    def test_rsi_values_bounded(self):
        np.random.seed(42)
        close = np.cumsum(np.random.randn(100)) + 100
        result = compute_rsi(close, period=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_monotonic_up_gives_high_rsi(self):
        close = np.arange(1.0, 31.0)  # Monotonically increasing
        result = compute_rsi(close, period=14)
        valid = result[~np.isnan(result)]
        # All gains, no losses -> RSI should be 100
        assert valid[-1] == 100.0


class TestComputeBollingerBands:
    def test_output_shapes(self):
        close = np.arange(1.0, 51.0, dtype=np.float64)
        upper, middle, lower = compute_bollinger_bands(close, period=20, std=2.0)
        assert upper.shape == close.shape
        assert middle.shape == close.shape
        assert lower.shape == close.shape

    def test_band_ordering(self):
        np.random.seed(42)
        close = np.cumsum(np.random.randn(50)) + 100
        upper, middle, lower = compute_bollinger_bands(close, period=20, std=2.0)
        # Where all three are valid, upper >= middle >= lower
        for i in range(19, 50):
            if not (np.isnan(upper[i]) or np.isnan(middle[i]) or np.isnan(lower[i])):
                assert upper[i] >= middle[i]
                assert middle[i] >= lower[i]

    def test_leading_nans(self):
        close = np.arange(1.0, 51.0, dtype=np.float64)
        upper, middle, lower = compute_bollinger_bands(close, period=20, std=2.0)
        assert np.isnan(upper[0])
        assert not np.isnan(upper[19])


class TestComputeATR:
    def test_output_shape_matches_input(self):
        n = 30
        high = np.random.RandomState(42).uniform(100, 110, n)
        low = high - np.random.RandomState(42).uniform(1, 5, n)
        close = (high + low) / 2
        result = compute_atr(high, low, close, period=14)
        assert result.shape == (n,)

    def test_leading_nans(self):
        n = 30
        rng = np.random.RandomState(42)
        close = np.cumsum(rng.randn(n)) + 100
        high = close + abs(rng.randn(n))
        low = close - abs(rng.randn(n))
        result = compute_atr(high, low, close, period=14)
        assert np.isnan(result[0])
        assert not np.isnan(result[14])

    def test_atr_always_positive(self):
        n = 50
        rng = np.random.RandomState(42)
        close = np.cumsum(rng.randn(n)) + 100
        high = close + abs(rng.randn(n)) + 0.1
        low = close - abs(rng.randn(n)) - 0.1
        result = compute_atr(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)


class TestComputeSMA:
    def test_output_shape(self):
        close = np.arange(1.0, 11.0)
        result = compute_sma(close, period=3)
        assert result.shape == close.shape

    def test_known_values(self):
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_sma(close, period=3)
        assert result[2] == pytest.approx(2.0)  # (1+2+3)/3
        assert result[3] == pytest.approx(3.0)  # (2+3+4)/3
        assert result[4] == pytest.approx(4.0)  # (3+4+5)/3
