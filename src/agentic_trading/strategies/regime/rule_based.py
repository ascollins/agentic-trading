"""Rule-based regime detection fallback.

Simple heuristic when HMM data is insufficient or unavailable.
Uses ADX, Bollinger Band width, and return statistics.
"""

from __future__ import annotations

import numpy as np

from agentic_trading.core.enums import RegimeType, VolatilityRegime


class RuleBasedRegimeDetector:
    """Simple rule-based regime detection.

    Rules:
    - ADX > 25 → trend, else range
    - BB width > 2 * median → high vol, else low vol
    - Return autocorrelation > 0.3 → trend confirmation
    """

    def __init__(
        self,
        adx_trend_threshold: float = 25.0,
        vol_high_multiplier: float = 1.5,
    ) -> None:
        self._adx_threshold = adx_trend_threshold
        self._vol_multiplier = vol_high_multiplier

    def detect(
        self,
        returns: list[float] | None = None,
        adx: float | None = None,
        bb_width: float | None = None,
        volume_ratio: float | None = None,
    ) -> tuple[RegimeType, VolatilityRegime, float]:
        """Detect regime using available features.

        Returns (regime_type, vol_regime, confidence).
        """
        regime = RegimeType.UNKNOWN
        vol_regime = VolatilityRegime.UNKNOWN
        confidence = 0.5

        votes_trend = 0
        votes_range = 0
        total_votes = 0

        # ADX rule
        if adx is not None:
            total_votes += 1
            if adx > self._adx_threshold:
                votes_trend += 1
            else:
                votes_range += 1

        # Return statistics
        if returns and len(returns) >= 20:
            arr = np.array(returns[-20:])
            total_votes += 1

            # Autocorrelation at lag 1
            if len(arr) > 1:
                autocorr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
                if not np.isnan(autocorr):
                    if autocorr > 0.2:
                        votes_trend += 1
                    else:
                        votes_range += 1
                else:
                    votes_range += 1

        # BB width / volatility
        if returns and len(returns) >= 20:
            arr = np.array(returns[-20:])
            current_vol = float(np.std(arr))
            total_votes += 1

            # Compare to longer-term vol if available
            if len(returns) >= 60:
                long_vol = float(np.std(np.array(returns[-60:])))
                if current_vol > long_vol * self._vol_multiplier:
                    vol_regime = VolatilityRegime.HIGH
                else:
                    vol_regime = VolatilityRegime.LOW
            else:
                vol_regime = VolatilityRegime.UNKNOWN

            # High vol slightly favors trend detection
            if vol_regime == VolatilityRegime.HIGH:
                votes_trend += 0.5

        # Determine regime
        if total_votes > 0:
            trend_pct = votes_trend / total_votes
            if trend_pct > 0.5:
                regime = RegimeType.TREND
                confidence = min(1.0, trend_pct)
            else:
                regime = RegimeType.RANGE
                confidence = min(1.0, 1.0 - trend_pct)
        else:
            confidence = 0.0

        return regime, vol_regime, round(confidence, 3)
