"""Test FeatureEngine produces 20+ features from sample candles."""

import math

from agentic_trading.core.enums import Timeframe
from agentic_trading.features.engine import FeatureEngine


class TestFeatureEngine:
    def test_compute_features_returns_feature_vector(self, sample_candles):
        engine = FeatureEngine()
        candles = sample_candles(n=100)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)
        assert fv.symbol == "BTC/USDT"
        assert fv.timeframe == Timeframe.M1

    def test_compute_features_produces_20_plus_features(self, sample_candles):
        engine = FeatureEngine()
        candles = sample_candles(n=100)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)
        assert len(fv.features) >= 20

    def test_features_include_ohlcv(self, sample_candles):
        engine = FeatureEngine()
        candles = sample_candles(n=100)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)
        assert "open" in fv.features
        assert "high" in fv.features
        assert "low" in fv.features
        assert "close" in fv.features
        assert "volume" in fv.features

    def test_features_include_ema(self, sample_candles):
        engine = FeatureEngine()
        candles = sample_candles(n=100)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)
        assert "ema_9" in fv.features
        assert "ema_12" in fv.features
        assert "ema_21" in fv.features
        assert "ema_50" in fv.features

    def test_features_include_rsi(self, sample_candles):
        engine = FeatureEngine()
        candles = sample_candles(n=100)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)
        assert "rsi_14" in fv.features

    def test_features_include_bollinger_bands(self, sample_candles):
        engine = FeatureEngine()
        candles = sample_candles(n=100)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)
        assert "bb_upper" in fv.features
        assert "bb_middle" in fv.features
        assert "bb_lower" in fv.features

    def test_features_include_macd(self, sample_candles):
        engine = FeatureEngine()
        candles = sample_candles(n=100)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)
        assert "macd" in fv.features
        assert "macd_signal" in fv.features
        assert "macd_histogram" in fv.features

    def test_features_include_atr(self, sample_candles):
        engine = FeatureEngine()
        candles = sample_candles(n=100)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)
        assert "atr_14" in fv.features

    def test_features_include_returns(self, sample_candles):
        engine = FeatureEngine()
        candles = sample_candles(n=100)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)
        assert "return_1" in fv.features
        assert "return_5" in fv.features
        assert "return_20" in fv.features

    def test_empty_candles_returns_empty_features(self):
        engine = FeatureEngine()
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, [])
        assert fv.features == {}

    def test_feature_values_are_finite_or_nan(self, sample_candles):
        engine = FeatureEngine()
        candles = sample_candles(n=100)
        fv = engine.compute_features("BTC/USDT", Timeframe.M1, candles)
        for key, val in fv.features.items():
            assert isinstance(val, float), f"{key} is not a float"
            assert not math.isinf(val), f"{key} is infinite"

    def test_add_candle_and_compute(self, sample_candles):
        engine = FeatureEngine()
        candles = sample_candles(n=50)
        for c in candles:
            engine.add_candle(c)
        buf = engine.get_buffer("BTC/USDT", Timeframe.M1)
        assert len(buf) == 50
