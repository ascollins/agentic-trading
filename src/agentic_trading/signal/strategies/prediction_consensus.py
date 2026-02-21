"""Prediction market consensus strategy.

Trades crypto based on aggregate prediction market directional bias.
Orthogonal to all technical strategies — captures alpha from
forward-looking market-priced probabilities.

Signal logic:
  LONG:  consensus > +threshold AND trending upward
  SHORT: consensus < -threshold AND trending downward
  EXIT:  consensus crosses back toward neutral

Confidence scaled by consensus magnitude and rate of change.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from agentic_trading.core.enums import SignalDirection, Timeframe
from agentic_trading.core.events import FeatureVector, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle

from .base import BaseStrategy
from .registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy("prediction_consensus")
class PredictionConsensusStrategy(BaseStrategy):
    """Trades based on aggregate prediction market directional bias.

    Parameters
    ----------
    consensus_threshold : float
        Minimum |consensus_score| to generate a signal (default 0.3).
    exit_threshold : float
        Consensus must drop below this to trigger exit (default 0.15).
    min_markets : int
        Minimum number of PM markets for signal validity (default 2).
    min_volume_usd : float
        Minimum average PM volume for signal validity (default 100_000).
    trend_lookback : int
        Number of consensus readings for trend detection (default 3).
    min_confidence : float
        Minimum confidence to emit signal (default 0.3).
    """

    def __init__(
        self,
        strategy_id: str = "prediction_consensus",
        params: dict[str, Any] | None = None,
    ) -> None:
        defaults = {
            "consensus_threshold": 0.3,
            "exit_threshold": 0.15,
            "min_markets": 2,
            "min_volume_usd": 100_000,
            "trend_lookback": 3,
            "min_confidence": 0.3,
            "signal_cooldown_minutes": 60,  # Longer cooldown for PM strategy
        }
        merged = {**defaults, **(params or {})}
        super().__init__(strategy_id, merged)

        # Track consensus history per symbol for trend detection
        self._consensus_history: dict[str, list[tuple[datetime, float]]] = (
            defaultdict(list)
        )

    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        """Check prediction market consensus and generate signals.

        The consensus data is injected into FeatureVector by the
        PredictionMarketAgent via the FeatureEngine. We read
        ``pm_consensus_score`` and ``pm_market_count`` from features.
        """
        f = features.features
        symbol = candle.symbol

        # Read PM features (injected by PredictionMarketAgent)
        consensus_score = f.get("pm_consensus_score")
        market_count = f.get("pm_market_count")
        avg_volume = f.get("pm_avg_volume_usd")
        event_risk = f.get("pm_event_risk_level", 0.0)

        # If no PM data available, nothing to do
        if consensus_score is None or market_count is None:
            return None

        min_markets = self._get_param("min_markets")
        min_volume = self._get_param("min_volume_usd")

        # Validity checks
        if market_count < min_markets:
            return None
        if avg_volume is not None and avg_volume < min_volume:
            return None

        # Update consensus history for trend detection
        self._consensus_history[symbol].append(
            (candle.timestamp, consensus_score)
        )
        # Keep only recent history
        lookback = self._get_param("trend_lookback")
        self._consensus_history[symbol] = (
            self._consensus_history[symbol][-(lookback + 5):]
        )

        # ---- EXIT CHECK ----
        current_pos = self._position_direction(symbol)
        if current_pos is not None:
            exit_threshold = self._get_param("exit_threshold")
            should_exit = False
            exit_reasons = []

            if current_pos == "long" and consensus_score < exit_threshold:
                should_exit = True
                exit_reasons.append(
                    f"PM consensus weakened to {consensus_score:.2f} "
                    f"(below exit threshold {exit_threshold})"
                )
            elif current_pos == "short" and consensus_score > -exit_threshold:
                should_exit = True
                exit_reasons.append(
                    f"PM consensus recovered to {consensus_score:.2f} "
                    f"(above exit threshold {-exit_threshold})"
                )

            if should_exit:
                self._record_exit(symbol)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction=SignalDirection.FLAT,
                    confidence=0.5,
                    rationale=" | ".join(exit_reasons),
                    features_used={
                        "pm_consensus_score": consensus_score,
                        "pm_market_count": market_count,
                    },
                    timeframe=candle.timeframe,
                    risk_constraints={},
                )
            return None  # Already positioned, don't re-signal

        # ---- ENTRY CHECK ----
        if self._on_cooldown(symbol, candle.timestamp):
            return None

        threshold = self._get_param("consensus_threshold")
        abs_score = abs(consensus_score)

        if abs_score < threshold:
            return None  # Consensus not strong enough

        # Check trend: consensus must be trending in signal direction
        if not self._is_trending(symbol, consensus_score):
            return None

        # Skip entries during high event risk (uncertain binary events)
        if event_risk is not None and event_risk > 0.7:
            logger.info(
                "PM consensus signal suppressed for %s: event_risk=%.2f",
                symbol, event_risk,
            )
            return None

        # Determine direction
        if consensus_score > threshold:
            direction = SignalDirection.LONG
        elif consensus_score < -threshold:
            direction = SignalDirection.SHORT
        else:
            return None

        # Confidence: scaled by consensus magnitude
        # |consensus| of 0.3 → confidence 0.3, |consensus| of 1.0 → confidence 0.8
        confidence = min(0.8, abs_score * 0.8)

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        rationale_parts = [
            f"PM consensus={consensus_score:+.2f} "
            f"({int(market_count)} markets, avg_vol=${avg_volume:,.0f})"
            if avg_volume
            else f"PM consensus={consensus_score:+.2f} ({int(market_count)} markets)",
        ]

        # Risk constraints: use ATR from technical features for stops
        atr = f.get("atr", 0)
        price = candle.close
        risk_constraints: dict[str, Any] = {
            "price": price,
            "sizing_method": "volatility_adjusted" if atr > 0 else "fixed_fractional",
        }
        if atr > 0:
            risk_constraints["atr"] = atr

        # TP/SL based on ATR (standard 2:1 R:R)
        take_profit = None
        stop_loss = None
        if atr > 0:
            sl_distance = atr * 2.5
            tp_distance = sl_distance * 2.0
            if direction == SignalDirection.LONG:
                stop_loss = Decimal(str(price - sl_distance))
                take_profit = Decimal(str(price + tp_distance))
            else:
                stop_loss = Decimal(str(price + sl_distance))
                take_profit = Decimal(str(price - tp_distance))

        features_used = {
            "pm_consensus_score": consensus_score,
            "pm_market_count": market_count,
            "close": price,
        }
        if avg_volume is not None:
            features_used["pm_avg_volume_usd"] = avg_volume
        if atr > 0:
            features_used["atr"] = atr

        self._record_entry(symbol, direction.value)
        self._record_signal_time(symbol, candle.timestamp)

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 3),
            rationale=" | ".join(rationale_parts),
            features_used=features_used,
            timeframe=candle.timeframe,
            risk_constraints=risk_constraints,
            take_profit=take_profit,
            stop_loss=stop_loss,
        )

    def _is_trending(self, symbol: str, current_score: float) -> bool:
        """Check if consensus is trending in the signal direction.

        Requires the last N readings to show a consistent directional trend.
        """
        history = self._consensus_history.get(symbol, [])
        lookback = self._get_param("trend_lookback")

        if len(history) < lookback:
            return True  # Not enough history, allow signal

        recent_scores = [score for _, score in history[-lookback:]]

        if current_score > 0:
            # Bullish: scores should be generally increasing
            return recent_scores[-1] >= recent_scores[0]
        else:
            # Bearish: scores should be generally decreasing
            return recent_scores[-1] <= recent_scores[0]
