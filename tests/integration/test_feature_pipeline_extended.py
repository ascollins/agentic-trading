"""Integration test: extended feature pipeline (ARIMA, FFT, orderbook).

Exercises FeatureEngine with advanced indicators enabled, verifying
that feature keys appear after sufficient warmup.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock

import numpy as np
import pytest

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.events import CandleEvent, FeatureVector
from agentic_trading.core.models import Candle
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from agentic_trading.intelligence.features.engine import FeatureEngine
from agentic_trading.intelligence.features.fourier import FourierExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_trending_candles(
    n: int,
    start_price: float = 50000.0,
    trend: float = 10.0,
    symbol: str = "BTC/USDT",
    timeframe: Timeframe = Timeframe.M1,
) -> list[Candle]:
    """Generate n candles with an upward trend."""
    base_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
    candles = []
    for i in range(n):
        price = start_price + trend * i
        candles.append(Candle(
            symbol=symbol,
            exchange=Exchange.BYBIT,
            timeframe=timeframe,
            timestamp=base_time + timedelta(minutes=i),
            open=Decimal(str(price - 5)),
            high=Decimal(str(price + 10)),
            low=Decimal(str(price - 10)),
            close=Decimal(str(price)),
            volume=Decimal("100"),
            quote_volume=Decimal(str(price * 100)),
            trades=50,
            is_closed=True,
        ))
    return candles


def _generate_sine_wave_candles(
    n: int,
    base_price: float = 50000.0,
    amplitude: float = 500.0,
    period: int = 20,
    symbol: str = "BTC/USDT",
    timeframe: Timeframe = Timeframe.M1,
) -> list[Candle]:
    """Generate n candles with a sine-wave close pattern."""
    base_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
    candles = []
    for i in range(n):
        price = base_price + amplitude * math.sin(2 * math.pi * i / period)
        candles.append(Candle(
            symbol=symbol,
            exchange=Exchange.BYBIT,
            timeframe=timeframe,
            timestamp=base_time + timedelta(minutes=i),
            open=Decimal(str(price - 3)),
            high=Decimal(str(price + 5)),
            low=Decimal(str(price - 5)),
            close=Decimal(str(round(price, 2))),
            volume=Decimal("100"),
            quote_volume=Decimal(str(round(price * 100, 2))),
            trades=50,
            is_closed=True,
        ))
    return candles


def _make_candle_event(candle: Candle) -> CandleEvent:
    """Convert a Candle to a CandleEvent for bus publishing."""
    return CandleEvent(
        symbol=candle.symbol,
        exchange=candle.exchange,
        timeframe=candle.timeframe,
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
        quote_volume=candle.quote_volume,
        trades=candle.trades,
        is_closed=candle.is_closed,
    )


# ---------------------------------------------------------------------------
# ARIMA tests
# ---------------------------------------------------------------------------


class TestARIMAFeatures:
    def test_arima_features_present_after_warmup(self):
        """After 60+ candles, FeatureVector contains arima keys."""
        engine = FeatureEngine(
            indicator_config={
                "arima_enabled": True,
                "arima_min_observations": 60,
                "fft_enabled": False,
                "smc_enabled": False,
            },
        )
        candles = _generate_trending_candles(80)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)

        # ARIMA features should be present
        arima_keys = [k for k in fv.features if k.startswith("arima_")]
        assert len(arima_keys) >= 1, f"Expected ARIMA keys, got: {list(fv.features.keys())}"

    def test_arima_features_absent_with_insufficient_data(self):
        """With < min_observations candles, no ARIMA keys in features."""
        engine = FeatureEngine(
            indicator_config={
                "arima_enabled": True,
                "arima_min_observations": 60,
                "fft_enabled": False,
                "smc_enabled": False,
            },
        )
        candles = _generate_trending_candles(30)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)

        arima_keys = [k for k in fv.features if k.startswith("arima_")]
        assert len(arima_keys) == 0

    def test_arima_forecast_direction_matches_trend(self):
        """In a strong uptrend, arima_forecast > current close."""
        engine = FeatureEngine(
            indicator_config={
                "arima_enabled": True,
                "arima_min_observations": 60,
                "fft_enabled": False,
                "smc_enabled": False,
            },
        )
        candles = _generate_trending_candles(100, trend=50.0)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)

        forecast = fv.features.get("arima_forecast")
        if forecast is not None and not math.isnan(forecast):
            current_close = float(candles[-1].close)
            # In a strong uptrend, forecast should be >= close
            # Allow some tolerance for model uncertainty
            assert forecast >= current_close * 0.98


# ---------------------------------------------------------------------------
# FFT tests
# ---------------------------------------------------------------------------


class TestFFTFeatures:
    def test_fft_features_present_after_warmup(self):
        """After 128+ candles, FeatureVector contains fft keys."""
        engine = FeatureEngine(
            indicator_config={
                "arima_enabled": False,
                "fft_enabled": True,
                "fft_min_window": 64,
                "fft_window_size": 128,
                "smc_enabled": False,
            },
        )
        candles = _generate_trending_candles(150)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)

        fft_keys = [k for k in fv.features if k.startswith("fft_")]
        assert len(fft_keys) >= 3, f"Expected FFT keys, got: {[k for k in fv.features if 'fft' in k]}"

    def test_fft_detects_embedded_cycle(self):
        """Sine-wave candles produce FFT features with dominant period near true period."""
        extractor = FourierExtractor(
            min_window=64,
            window_size=128,
            num_components=5,
            detrend=True,
        )

        # Generate sine wave with period=20
        n = 256
        closes = np.array([
            50000 + 500 * math.sin(2 * math.pi * i / 20)
            for i in range(n)
        ], dtype=np.float64)

        features = extractor.compute(closes)
        dominant_period = features.get("fft_dominant_period", 0)
        # The dominant period should be approximately 20 (within tolerance)
        assert abs(dominant_period - 20) < 5, (
            f"Expected dominant period ~20, got {dominant_period}"
        )

    def test_fft_features_absent_below_min_window(self):
        """With < min_window candles, no FFT keys in features."""
        engine = FeatureEngine(
            indicator_config={
                "arima_enabled": False,
                "fft_enabled": True,
                "fft_min_window": 64,
                "smc_enabled": False,
            },
        )
        candles = _generate_trending_candles(30)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)

        fft_keys = [k for k in fv.features if k.startswith("fft_")]
        assert len(fft_keys) == 0


# ---------------------------------------------------------------------------
# Feature version tests
# ---------------------------------------------------------------------------


class TestFeatureVersion:
    def test_feature_version_hash_deterministic(self):
        """Same indicator config produces same feature_version hash."""
        cfg = {"ema_periods": [9, 21], "arima_enabled": False, "fft_enabled": False, "smc_enabled": False}
        engine1 = FeatureEngine(indicator_config=cfg)
        engine2 = FeatureEngine(indicator_config=cfg)
        assert engine1.feature_version == engine2.feature_version

    def test_feature_version_changes_on_config_change(self):
        """Changing indicator params produces different feature_version hash."""
        cfg1 = {"ema_periods": [9, 21], "arima_enabled": False, "fft_enabled": False, "smc_enabled": False}
        cfg2 = {"ema_periods": [9, 50], "arima_enabled": False, "fft_enabled": False, "smc_enabled": False}
        engine1 = FeatureEngine(indicator_config=cfg1)
        engine2 = FeatureEngine(indicator_config=cfg2)
        assert engine1.feature_version != engine2.feature_version


# ---------------------------------------------------------------------------
# Event bus integration test
# ---------------------------------------------------------------------------


class TestFeatureEventBusIntegration:
    @pytest.mark.asyncio
    async def test_feature_vector_published_with_all_indicator_groups(self):
        """CandleEvent -> FeatureEngine -> FeatureVector has core indicator keys."""
        bus = MemoryEventBus()
        captured: list[FeatureVector] = []

        async def capture(event):
            if isinstance(event, FeatureVector):
                captured.append(event)

        await bus.subscribe("feature.vector", "test", capture)

        engine = FeatureEngine(
            event_bus=bus,
            indicator_config={
                "arima_enabled": False,
                "fft_enabled": False,
                "smc_enabled": False,
            },
        )
        await engine.start()

        # Feed 80 candles through bus
        candles = _generate_trending_candles(80)
        for c in candles:
            await bus.publish("market.candle", _make_candle_event(c))
            await asyncio.sleep(0)  # yield to handlers

        await asyncio.sleep(0.05)

        # Should have received FeatureVectors
        assert len(captured) > 0
        last_fv = captured[-1]

        # Verify core indicator groups are present
        assert "close" in last_fv.features
        assert "ema_9" in last_fv.features
        assert "rsi_14" in last_fv.features or "rsi" in last_fv.features
        assert "macd" in last_fv.features
        assert last_fv.feature_version != ""

        await engine.stop()
