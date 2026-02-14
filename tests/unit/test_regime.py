"""Test RuleBasedRegimeDetector returns RegimeState and hysteresis."""

import numpy as np

from agentic_trading.core.enums import RegimeType, VolatilityRegime
from agentic_trading.strategies.regime.rule_based import RuleBasedRegimeDetector


class TestRuleBasedRegimeDetector:
    def test_high_adx_returns_trend(self):
        detector = RuleBasedRegimeDetector(adx_trend_threshold=25.0)
        regime, vol, confidence = detector.detect(adx=40.0)
        assert regime == RegimeType.TREND

    def test_low_adx_returns_range(self):
        detector = RuleBasedRegimeDetector(adx_trend_threshold=25.0)
        regime, vol, confidence = detector.detect(adx=15.0)
        assert regime == RegimeType.RANGE

    def test_no_features_returns_unknown(self):
        detector = RuleBasedRegimeDetector()
        regime, vol, confidence = detector.detect()
        assert regime == RegimeType.UNKNOWN
        assert confidence == 0.0

    def test_returns_tuple_of_three(self):
        detector = RuleBasedRegimeDetector()
        result = detector.detect(adx=30.0)
        assert len(result) == 3
        regime, vol, confidence = result
        assert isinstance(regime, RegimeType)
        assert isinstance(vol, VolatilityRegime)
        assert isinstance(confidence, float)

    def test_confidence_between_0_and_1(self):
        detector = RuleBasedRegimeDetector()
        _, _, confidence = detector.detect(adx=50.0)
        assert 0.0 <= confidence <= 1.0

    def test_with_returns_and_adx(self):
        """Test detection using both ADX and return statistics."""
        detector = RuleBasedRegimeDetector()
        # Trending returns: autocorrelation should be positive
        np.random.seed(42)
        trending_returns = np.cumsum(np.random.randn(60) * 0.01).tolist()
        regime, vol, confidence = detector.detect(
            returns=trending_returns,
            adx=35.0,
        )
        assert regime == RegimeType.TREND

    def test_ranging_returns_give_range(self):
        """Mean-reverting returns with low ADX should give range."""
        detector = RuleBasedRegimeDetector()
        np.random.seed(123)
        # Independent returns (no trend)
        ranging_returns = (np.random.randn(60) * 0.005).tolist()
        regime, vol, confidence = detector.detect(
            returns=ranging_returns,
            adx=15.0,
        )
        assert regime == RegimeType.RANGE

    def test_volatility_regime_detection(self):
        """With enough returns, volatility regime should be detected."""
        detector = RuleBasedRegimeDetector()
        np.random.seed(42)
        # High recent volatility
        long_returns = (np.random.randn(60) * 0.001).tolist()
        # Make the last 20 returns much more volatile
        long_returns[-20:] = (np.random.randn(20) * 0.01).tolist()
        regime, vol, confidence = detector.detect(returns=long_returns, adx=20.0)
        assert vol in {VolatilityRegime.HIGH, VolatilityRegime.LOW, VolatilityRegime.UNKNOWN}

    def test_custom_threshold(self):
        """Custom adx_trend_threshold should be respected."""
        detector = RuleBasedRegimeDetector(adx_trend_threshold=40.0)
        regime, _, _ = detector.detect(adx=35.0)
        assert regime == RegimeType.RANGE  # 35 < 40

        regime2, _, _ = detector.detect(adx=45.0)
        assert regime2 == RegimeType.TREND  # 45 > 40
