"""Tests for the FeatureEngine computing new indicators (stochastic, OBV, etc.).

Ensures the engine integrates all new indicator functions into the
FeatureVector output.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.models import Candle
from agentic_trading.features.engine import FeatureEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candles(n: int = 100, start_price: float = 100.0) -> list[Candle]:
    """Generate a list of candles with realistic-ish OHLCV."""
    rng = np.random.default_rng(42)
    candles = []
    price = start_price
    for i in range(n):
        change = rng.normal(0, 0.5)
        open_ = price
        close = price + change
        high = max(open_, close) + abs(rng.normal(0.3, 0.1))
        low = min(open_, close) - abs(rng.normal(0.3, 0.1))
        volume = abs(rng.normal(1000, 200))
        candles.append(Candle(
            symbol="BTC/USDT",
            exchange=Exchange.BYBIT,
            timeframe=Timeframe.H1,
            timestamp=datetime(2024, 1, 1, i % 24, tzinfo=UTC),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            is_closed=True,
        ))
        price = close
    return candles


# ===========================================================================
# FeatureEngine: new indicators
# ===========================================================================


class TestFeatureEngineNewIndicators:
    """Verify FeatureEngine computes the new indicators."""

    def test_stochastic_computed(self):
        engine = FeatureEngine()
        candles = _make_candles(50)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "stoch_k" in fv.features
        assert "stoch_d" in fv.features
        assert not np.isnan(fv.features["stoch_k"])

    def test_stochastic_prev_values(self):
        engine = FeatureEngine()
        candles = _make_candles(50)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "stoch_k_prev" in fv.features
        assert "stoch_d_prev" in fv.features

    def test_obv_computed(self):
        engine = FeatureEngine()
        candles = _make_candles(50)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "obv" in fv.features
        assert not np.isnan(fv.features["obv"])

    def test_obv_ema_computed(self):
        engine = FeatureEngine()
        candles = _make_candles(50)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "obv_ema_20" in fv.features

    def test_keltner_channel_computed(self):
        engine = FeatureEngine()
        candles = _make_candles(50)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "keltner_upper" in fv.features
        assert "keltner_middle" in fv.features
        assert "keltner_lower" in fv.features

    def test_keltner_ordering(self):
        engine = FeatureEngine()
        candles = _make_candles(50)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        upper = fv.features["keltner_upper"]
        middle = fv.features["keltner_middle"]
        lower = fv.features["keltner_lower"]
        if not any(np.isnan(v) for v in (upper, middle, lower)):
            assert upper >= middle >= lower

    def test_bbw_computed(self):
        engine = FeatureEngine()
        candles = _make_candles(50)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "bbw" in fv.features
        assert not np.isnan(fv.features["bbw"])

    def test_bbw_percentile_not_computed_short_history(self):
        engine = FeatureEngine()
        candles = _make_candles(50)  # Less than 120
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "bbw_percentile" not in fv.features or np.isnan(
            fv.features.get("bbw_percentile", float("nan"))
        )

    def test_bbw_percentile_computed_long_history(self):
        engine = FeatureEngine()
        candles = _make_candles(150)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "bbw_percentile" in fv.features
        pct = fv.features["bbw_percentile"]
        assert 0.0 <= pct <= 1.0

    def test_roc_computed(self):
        engine = FeatureEngine()
        candles = _make_candles(50)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "roc_12" in fv.features

    def test_macd_prev_values(self):
        engine = FeatureEngine()
        candles = _make_candles(50)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "macd_prev" in fv.features
        assert "macd_signal_prev" in fv.features

    def test_longer_returns_computed(self):
        engine = FeatureEngine()
        candles = _make_candles(70)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "return_60" in fv.features
        assert not np.isnan(fv.features["return_60"])

    def test_existing_features_still_present(self):
        """Regression: all pre-existing features should still be computed."""
        engine = FeatureEngine()
        candles = _make_candles(50)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)

        expected = [
            "open", "high", "low", "close", "volume",
            "ema_12", "ema_26", "ema_50",
            "sma_20", "sma_50",
            "rsi", "rsi_14",
            "bb_upper", "bb_middle", "bb_lower", "bb_pct_b",
            "adx", "adx_14",
            "atr", "atr_14",
            "macd", "macd_signal", "macd_histogram",
            "donchian_upper", "donchian_lower",
            "vwap",
            "volume_ratio",
            "return_1",
        ]
        for key in expected:
            assert key in fv.features, f"Missing expected feature: {key}"

    def test_empty_candles(self):
        engine = FeatureEngine()
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, [])
        assert fv.features == {}

    def test_single_candle(self):
        engine = FeatureEngine()
        candles = _make_candles(1)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        assert "close" in fv.features
        assert "obv" in fv.features
