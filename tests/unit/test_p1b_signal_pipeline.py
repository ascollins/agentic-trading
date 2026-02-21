"""Tests for P1b sprint: signal pipeline enrichment.

Covers:
- ARIMA forecast features (Task #9)
- Fourier/FFT features (Task #10)
- Feature version hashing (Task #11)
- MBO microstructure features (Task #12)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from agentic_trading.core.enums import Timeframe
from agentic_trading.core.models import Candle
from agentic_trading.intelligence.features.arima import ARIMAForecaster
from agentic_trading.intelligence.features.engine import FeatureEngine
from agentic_trading.intelligence.features.fourier import FourierExtractor
from agentic_trading.intelligence.features.orderbook import OrderbookEngine


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_price_series(n: int = 200, base: float = 50000.0, seed: int = 42) -> np.ndarray:
    """Generate a synthetic price series with trend + noise."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0.0002, 0.01, n)
    prices = base * np.cumprod(1.0 + returns)
    return prices


def _make_candles(
    n: int = 200, symbol: str = "BTCUSDT", seed: int = 42,
) -> list[Candle]:
    """Create a list of Candle objects from synthetic data."""
    rng = np.random.RandomState(seed)
    base = 50000.0
    candles = []
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for i in range(n):
        close = base * (1.0 + rng.normal(0.0002, 0.01))
        high = close * (1.0 + abs(rng.normal(0, 0.003)))
        low = close * (1.0 - abs(rng.normal(0, 0.003)))
        open_ = (high + low) / 2.0
        volume = abs(rng.normal(100, 30))
        candles.append(Candle(
            symbol=symbol,
            exchange="bybit",
            timeframe=Timeframe.M1,
            timestamp=ts.replace(minute=i % 60, hour=i // 60),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            is_closed=True,
        ))
        base = close

    return candles


def _make_orderbook(
    mid: float = 50000.0, levels: int = 20, spread_bps: float = 5.0,
) -> tuple[list[list[float]], list[list[float]]]:
    """Generate a synthetic orderbook."""
    half_spread = mid * spread_bps / 20_000.0
    bids = []
    asks = []
    for i in range(levels):
        bid_price = mid - half_spread - i * 10.0
        ask_price = mid + half_spread + i * 10.0
        bid_qty = 1.0 + i * 0.5
        ask_qty = 1.0 + i * 0.3
        bids.append([bid_price, bid_qty])
        asks.append([ask_price, ask_qty])
    return bids, asks


# ===============================================================
# Task #9: ARIMA Forecast Features
# ===============================================================


class TestARIMAForecaster:
    """Test ARIMA forecaster."""

    def test_insufficient_data_returns_empty(self):
        forecaster = ARIMAForecaster(min_observations=60)
        result = forecaster.compute(np.array([100.0, 101.0, 102.0]))
        assert result == {}

    def test_fallback_produces_forecast(self):
        """Test the exponential smoothing fallback."""
        forecaster = ARIMAForecaster(min_observations=30)
        prices = _make_price_series(n=60)
        result = forecaster.compute(prices, symbol="TEST")

        assert "arima_forecast" in result
        assert "arima_lower" in result
        assert "arima_upper" in result
        assert "arima_residual_std" in result
        assert "arima_forecast_return" in result
        assert result["arima_lower"] <= result["arima_forecast"] <= result["arima_upper"]

    def test_forecast_reasonable_range(self):
        """Forecast should be within reasonable range of last price."""
        forecaster = ARIMAForecaster(min_observations=30)
        prices = _make_price_series(n=100, base=50000.0)
        result = forecaster.compute(prices, symbol="BTC")

        last_price = prices[-1]
        forecast = result["arima_forecast"]
        # Forecast should be within 10% of last price
        assert abs(forecast - last_price) / last_price < 0.10

    def test_forecast_return_sign(self):
        """Forecast return should be consistent with forecast vs last price."""
        forecaster = ARIMAForecaster(min_observations=30)
        prices = _make_price_series(n=100)
        result = forecaster.compute(prices, symbol="BTC")

        last_price = float(prices[-1])
        forecast = result["arima_forecast"]
        expected_return = forecast / last_price - 1.0
        assert result["arima_forecast_return"] == pytest.approx(expected_return, rel=1e-6)

    def test_max_window_trims_series(self):
        """With max_window=50, only last 50 observations are used."""
        forecaster = ARIMAForecaster(min_observations=30, max_window=50)
        prices = _make_price_series(n=200)
        result = forecaster.compute(prices, symbol="BTC")
        assert "arima_forecast" in result

    def test_clear_cache(self):
        forecaster = ARIMAForecaster(min_observations=30)
        forecaster._fitted_orders["BTC"] = (1, 1, 1)
        forecaster.clear_cache("BTC")
        assert "BTC" not in forecaster._fitted_orders


# ===============================================================
# Task #10: Fourier/FFT Features
# ===============================================================


class TestFourierExtractor:
    """Test FFT spectral feature extraction."""

    def test_insufficient_data_returns_empty(self):
        extractor = FourierExtractor(min_window=64)
        result = extractor.compute(np.array([1.0, 2.0, 3.0]))
        assert result == {}

    def test_produces_expected_keys(self):
        extractor = FourierExtractor(
            min_window=32, window_size=64, num_components=3,
        )
        prices = _make_price_series(n=64)
        result = extractor.compute(prices)

        assert "fft_mag_1" in result
        assert "fft_phase_1" in result
        assert "fft_period_1" in result
        assert "fft_mag_3" in result
        assert "fft_dominant_period" in result
        assert "fft_spectral_entropy" in result

    def test_magnitudes_sorted_descending(self):
        extractor = FourierExtractor(min_window=32, window_size=64, num_components=5)
        prices = _make_price_series(n=128)
        result = extractor.compute(prices)

        mags = [result[f"fft_mag_{i}"] for i in range(1, 6)]
        for i in range(len(mags) - 1):
            assert mags[i] >= mags[i + 1]

    def test_spectral_entropy_in_range(self):
        """Normalised entropy should be in [0, 1]."""
        extractor = FourierExtractor(min_window=32, window_size=64)
        prices = _make_price_series(n=128)
        result = extractor.compute(prices)

        assert 0.0 <= result["fft_spectral_entropy"] <= 1.0

    def test_pure_sine_low_entropy(self):
        """A pure sine wave should have low spectral entropy."""
        extractor = FourierExtractor(
            min_window=32, window_size=128, num_components=5, detrend=False,
        )
        t = np.arange(128)
        sine = np.sin(2 * np.pi * t / 16.0)  # Period = 16 bars
        result = extractor.compute(sine)

        # Entropy should be low (energy concentrated in one frequency)
        assert result["fft_spectral_entropy"] < 0.5

    def test_dominant_period_matches_sine(self):
        """For a pure sine wave, dominant period should match input frequency."""
        extractor = FourierExtractor(
            min_window=32, window_size=128, num_components=5, detrend=False,
        )
        period = 32
        t = np.arange(128)
        sine = np.sin(2 * np.pi * t / period)
        result = extractor.compute(sine)

        # Dominant period should be close to input period
        assert abs(result["fft_dominant_period"] - period) < 2.0

    def test_detrend_removes_linear_trend(self):
        """Detrending should reduce low-frequency dominance from trends."""
        prices_trending = np.linspace(100, 200, 128)

        ext_no_detrend = FourierExtractor(
            min_window=32, window_size=128, detrend=False,
        )
        ext_detrend = FourierExtractor(
            min_window=32, window_size=128, detrend=True,
        )

        r_no = ext_no_detrend.compute(prices_trending)
        r_yes = ext_detrend.compute(prices_trending)

        # Without detrending, dominant period should be very long (trend)
        # With detrending, the trend is removed
        assert r_no["fft_mag_1"] > r_yes["fft_mag_1"]


# ===============================================================
# Task #11: Feature Version Hashing
# ===============================================================


class TestFeatureVersionHashing:
    """Test feature version deterministic hashing."""

    def test_version_populated(self):
        engine = FeatureEngine()
        assert engine.feature_version
        assert len(engine.feature_version) == 16  # content_hash default length

    def test_same_config_same_hash(self):
        e1 = FeatureEngine()
        e2 = FeatureEngine()
        assert e1.feature_version == e2.feature_version

    def test_different_config_different_hash(self):
        e1 = FeatureEngine(indicator_config={"rsi_period": 14})
        e2 = FeatureEngine(indicator_config={"rsi_period": 21})
        assert e1.feature_version != e2.feature_version

    def test_feature_version_in_output(self):
        """FeatureVector should carry the feature_version."""
        engine = FeatureEngine()
        candles = _make_candles(n=30)
        fv = engine.compute_features("BTCUSDT", Timeframe.M1, candles)
        assert fv.feature_version == engine.feature_version
        assert fv.feature_version != ""

    def test_arima_toggle_changes_hash(self):
        e1 = FeatureEngine(indicator_config={"arima_enabled": True})
        e2 = FeatureEngine(indicator_config={"arima_enabled": False})
        assert e1.feature_version != e2.feature_version


# ===============================================================
# Task #12: MBO Microstructure Features
# ===============================================================


class TestMBOMicrostructure:
    """Test extended orderbook microstructure features."""

    def test_depth_at_bps_levels(self):
        engine = OrderbookEngine(depth_bps_levels=[10, 50, 100])
        bids, asks = _make_orderbook(mid=50000.0, levels=20)
        features = engine.compute_orderbook_features("BTC", bids, asks)

        assert "ob_bid_depth_10bps" in features
        assert "ob_ask_depth_10bps" in features
        assert "ob_bid_depth_50bps" in features
        assert "ob_bid_depth_100bps" in features
        assert "ob_imbalance_10bps" in features

    def test_depth_monotonically_increasing(self):
        """Wider bps levels should have >= depth than narrower levels."""
        engine = OrderbookEngine(depth_bps_levels=[10, 50, 100])
        bids, asks = _make_orderbook(mid=50000.0, levels=50)
        features = engine.compute_orderbook_features("BTC", bids, asks)

        assert features["ob_bid_depth_10bps"] <= features["ob_bid_depth_50bps"]
        assert features["ob_bid_depth_50bps"] <= features["ob_bid_depth_100bps"]

    def test_imbalance_bps_range(self):
        """Depth imbalance should be in [-1, 1]."""
        engine = OrderbookEngine(depth_bps_levels=[50])
        bids, asks = _make_orderbook(mid=50000.0, levels=20)
        features = engine.compute_orderbook_features("BTC", bids, asks)

        assert -1.0 <= features["ob_imbalance_50bps"] <= 1.0

    def test_level_counts(self):
        engine = OrderbookEngine()
        bids, asks = _make_orderbook(mid=50000.0, levels=15)
        features = engine.compute_orderbook_features("BTC", bids, asks)

        assert "ob_bid_level_count" in features
        assert "ob_ask_level_count" in features
        assert features["ob_bid_level_count"] > 0
        assert features["ob_ask_level_count"] > 0

    def test_weighted_mid_and_microprice(self):
        engine = OrderbookEngine()
        # Asymmetric book: large bid, small ask
        bids = [[50000.0, 10.0], [49990.0, 5.0]]
        asks = [[50010.0, 1.0], [50020.0, 2.0]]
        features = engine.compute_orderbook_features("BTC", bids, asks)

        assert "ob_weighted_mid" in features
        assert "ob_microprice_offset_bps" in features
        # With large bid and small ask, weighted mid should lean towards ask
        mid = (50000.0 + 50010.0) / 2.0
        assert features["ob_weighted_mid"] > mid  # Pulled towards ask

    def test_snapshot_interval_tracking(self):
        engine = OrderbookEngine()
        bids, asks = _make_orderbook()
        engine.compute_orderbook_features("BTC", bids, asks)
        time.sleep(0.01)  # Small delay
        features = engine.compute_orderbook_features("BTC", bids, asks)

        assert "ob_snapshot_interval_ms" in features
        assert features["ob_snapshot_interval_ms"] > 0

    def test_trade_intensity_no_trades(self):
        engine = OrderbookEngine()
        result = engine.get_trade_intensity("BTC")
        assert result["ob_trade_intensity"] == 0.0
        assert result["ob_trade_count_window"] == 0.0

    def test_trade_intensity_with_trades(self):
        engine = OrderbookEngine()
        now = time.monotonic()
        for i in range(10):
            engine.record_trade("BTC", timestamp=now - 30 + i)
        result = engine.get_trade_intensity("BTC", window_seconds=60.0)

        assert result["ob_trade_count_window"] == 10.0
        assert result["ob_trade_intensity"] > 0
        assert result["ob_avg_inter_trade_ms"] > 0

    def test_trade_intensity_window_filter(self):
        """Old trades outside the window should be excluded."""
        engine = OrderbookEngine()
        now = time.monotonic()
        # 5 old trades (120s ago) + 3 recent trades
        for i in range(5):
            engine.record_trade("BTC", timestamp=now - 120 + i)
        for i in range(3):
            engine.record_trade("BTC", timestamp=now - 5 + i)
        result = engine.get_trade_intensity("BTC", window_seconds=60.0)

        assert result["ob_trade_count_window"] == 3.0


# ===============================================================
# Integration: FeatureEngine with ARIMA + FFT
# ===============================================================


class TestFeatureEngineIntegration:
    """Test that ARIMA and FFT features appear in FeatureVector output."""

    def test_arima_features_in_output(self):
        """With enough candles, ARIMA features should appear."""
        engine = FeatureEngine(indicator_config={
            "arima_enabled": True,
            "arima_min_observations": 30,
            "fft_enabled": False,
            "smc_enabled": False,
        })
        candles = _make_candles(n=80)
        fv = engine.compute_features("BTCUSDT", Timeframe.M1, candles)

        assert "arima_forecast" in fv.features
        assert "arima_forecast_return" in fv.features

    def test_fft_features_in_output(self):
        """With enough candles, FFT features should appear."""
        engine = FeatureEngine(indicator_config={
            "arima_enabled": False,
            "fft_enabled": True,
            "fft_min_window": 32,
            "fft_window_size": 64,
            "smc_enabled": False,
        })
        candles = _make_candles(n=80)
        fv = engine.compute_features("BTCUSDT", Timeframe.M1, candles)

        assert "fft_mag_1" in fv.features
        assert "fft_dominant_period" in fv.features
        assert "fft_spectral_entropy" in fv.features

    def test_insufficient_candles_no_arima(self):
        """With too few candles, no ARIMA features."""
        engine = FeatureEngine(indicator_config={
            "arima_enabled": True,
            "arima_min_observations": 100,
            "fft_enabled": False,
            "smc_enabled": False,
        })
        candles = _make_candles(n=30)
        fv = engine.compute_features("BTCUSDT", Timeframe.M1, candles)

        assert "arima_forecast" not in fv.features

    def test_disabled_modules(self):
        """Disabled modules should produce no features."""
        engine = FeatureEngine(indicator_config={
            "arima_enabled": False,
            "fft_enabled": False,
            "smc_enabled": False,
        })
        candles = _make_candles(n=200)
        fv = engine.compute_features("BTCUSDT", Timeframe.M1, candles)

        assert "arima_forecast" not in fv.features
        assert "fft_mag_1" not in fv.features
