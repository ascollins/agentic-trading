"""RSI Divergence with Trend Confirmation (CMT Strategy 3).

Detects bullish and bearish RSI divergence with trend context:
  Bullish divergence: Price makes lower low, RSI makes higher low (RSI < 35)
  Bearish divergence: Price makes higher high, RSI makes lower high (RSI > 65)

Entry requires trendline break confirmation. Exit on RSI mean reversion
or opposite divergence.

Signal logic:
  LONG:  Bullish RSI divergence + RSI < 35 + min 5 bars between lows
  SHORT: Bearish RSI divergence + RSI > 65 + min 5 bars between highs
  EXIT:  RSI crosses 50 (mean reversion) or divergence invalidated
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from agentic_trading.core.enums import SignalDirection
from agentic_trading.core.events import FeatureVector, RegimeState, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle

from .base import BaseStrategy
from .registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy("rsi_divergence")
class RSIDivergenceStrategy(BaseStrategy):
    """CMT Strategy 3: RSI Divergence with Trend Confirmation."""

    def __init__(self, strategy_id: str = "rsi_divergence", params: dict[str, Any] | None = None):
        defaults = {
            "rsi_period": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "lookback_bars": 30,
            "min_divergence_bars": 5,
            "rsi_exit_level": 50,
            "atr_multiplier": 2.0,
            "min_confidence": 0.4,
            "trend_filter": True,  # Only take divergence counter-trend
        }
        merged = {**defaults, **(params or {})}
        super().__init__(strategy_id, merged)
        self._current_regime: RegimeState | None = None
        # Rolling price/RSI history for divergence detection
        self._price_history: dict[str, list[float]] = {}
        self._rsi_history: dict[str, list[float]] = {}
        self._max_history = 60

    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        f = features.features

        rsi = f.get("rsi")
        atr = f.get("atr")
        adx = f.get("adx")
        close = candle.close

        if rsi is None:
            return None

        # Update rolling history
        self._update_history(candle.symbol, close, rsi)

        # ---- EXIT CHECK ----
        current_pos = self._position_direction(candle.symbol)
        if current_pos is not None:
            should_exit = False
            exit_reasons = []

            rsi_exit = self._get_param("rsi_exit_level")
            if current_pos == "long" and rsi >= rsi_exit:
                should_exit = True
                exit_reasons.append(f"RSI={rsi:.1f} crossed above {rsi_exit} (mean reversion)")
            elif current_pos == "short" and rsi <= rsi_exit:
                should_exit = True
                exit_reasons.append(f"RSI={rsi:.1f} crossed below {rsi_exit} (mean reversion)")

            if should_exit:
                self._record_exit(candle.symbol)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=candle.symbol,
                    direction=SignalDirection.FLAT,
                    confidence=0.5,
                    rationale=" | ".join(exit_reasons),
                    features_used={"rsi": rsi},
                    timeframe=candle.timeframe,
                    risk_constraints={},
                )
            return None

        # ---- ENTRY CHECK ----
        if self._on_cooldown(candle.symbol, candle.timestamp):
            return None

        prices = self._price_history.get(candle.symbol, [])
        rsis = self._rsi_history.get(candle.symbol, [])

        lookback = self._get_param("lookback_bars")
        min_bars = self._get_param("min_divergence_bars")

        if len(prices) < lookback:
            return None

        direction = SignalDirection.FLAT
        rationale_parts = []
        divergence_strength = 0.0

        rsi_oversold = self._get_param("rsi_oversold")
        rsi_overbought = self._get_param("rsi_overbought")

        # Check for bullish divergence: price lower low, RSI higher low
        if rsi < rsi_oversold:
            div = self._detect_bullish_divergence(prices, rsis, lookback, min_bars)
            if div is not None:
                direction = SignalDirection.LONG
                divergence_strength = div
                rationale_parts.append(f"Bullish RSI divergence (strength={div:.2f})")
                rationale_parts.append(f"RSI={rsi:.1f} < {rsi_oversold} (oversold)")

        # Check for bearish divergence: price higher high, RSI lower high
        elif rsi > rsi_overbought:
            div = self._detect_bearish_divergence(prices, rsis, lookback, min_bars)
            if div is not None:
                direction = SignalDirection.SHORT
                divergence_strength = div
                rationale_parts.append(f"Bearish RSI divergence (strength={div:.2f})")
                rationale_parts.append(f"RSI={rsi:.1f} > {rsi_overbought} (overbought)")

        if direction == SignalDirection.FLAT:
            return None

        # Trend filter: prefer divergence counter to prevailing trend
        if self._get_param("trend_filter") and adx is not None:
            ema_50 = f.get("ema_50")
            ema_200 = f.get("ema_200")
            if ema_50 is not None and ema_200 is not None:
                if direction == SignalDirection.LONG and ema_50 > ema_200 and adx > 25:
                    # Already in uptrend, bullish divergence less meaningful
                    divergence_strength *= 0.6
                    rationale_parts.append("Already in uptrend (reduced strength)")
                elif direction == SignalDirection.SHORT and ema_50 < ema_200 and adx > 25:
                    divergence_strength *= 0.6
                    rationale_parts.append("Already in downtrend (reduced strength)")

        # Confidence
        confidence = min(1.0, divergence_strength * 0.7 + 0.3)

        # Regime adjustment
        if self._current_regime and self._current_regime.regime.value == "trend":
            # Divergence at trend exhaustion is higher quality
            confidence = min(1.0, confidence + 0.1)
            rationale_parts.append("Trend regime (potential reversal)")

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        # Risk constraints
        risk_constraints = {
            "sizing_method": "fixed_fractional",
            "price": close,
        }
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            risk_constraints["stop_distance_atr"] = atr * atr_mult
            risk_constraints["atr"] = atr

        features_used = {
            "rsi": rsi,
            "close": close,
            "divergence_strength": divergence_strength,
        }
        if adx is not None:
            features_used["adx"] = adx
        if atr is not None:
            features_used["atr"] = atr

        self._record_entry(candle.symbol, direction.value)
        self._record_signal_time(candle.symbol, candle.timestamp)

        # Compute explicit TP/SL prices
        _take_profit: Decimal | None = None
        _stop_loss: Decimal | None = None
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            sl_distance = atr * atr_mult
            if direction == SignalDirection.LONG:
                _stop_loss = Decimal(str(close - sl_distance))
                _take_profit = Decimal(str(close + sl_distance * 2))
            elif direction == SignalDirection.SHORT:
                _stop_loss = Decimal(str(close + sl_distance))
                _take_profit = Decimal(str(close - sl_distance * 2))

        return Signal(
            strategy_id=self.strategy_id,
            symbol=candle.symbol,
            direction=direction,
            confidence=round(confidence, 3),
            rationale=" | ".join(rationale_parts),
            features_used=features_used,
            timeframe=candle.timeframe,
            risk_constraints=risk_constraints,
            take_profit=_take_profit,
            stop_loss=_stop_loss,
        )

    def _update_history(self, symbol: str, price: float, rsi: float) -> None:
        """Maintain rolling price and RSI history."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
            self._rsi_history[symbol] = []

        self._price_history[symbol].append(price)
        self._rsi_history[symbol].append(rsi)

        # Trim to max length
        if len(self._price_history[symbol]) > self._max_history:
            self._price_history[symbol] = self._price_history[symbol][-self._max_history:]
            self._rsi_history[symbol] = self._rsi_history[symbol][-self._max_history:]

    def _detect_bullish_divergence(
        self,
        prices: list[float],
        rsis: list[float],
        lookback: int,
        min_bars: int,
    ) -> float | None:
        """Detect bullish divergence: price lower low, RSI higher low.

        Returns divergence strength (0-1) or None.
        """
        recent_prices = prices[-lookback:]
        recent_rsis = rsis[-lookback:]

        # Find local lows in price (simple approach: compare to neighbors)
        lows = []
        for i in range(2, len(recent_prices) - 1):
            if (recent_prices[i] < recent_prices[i - 1]
                    and recent_prices[i] < recent_prices[i - 2]
                    and recent_prices[i] <= recent_prices[i + 1]):
                lows.append(i)

        # Also check the current bar as potential low
        if len(recent_prices) >= 3 and recent_prices[-1] < recent_prices[-2]:
            lows.append(len(recent_prices) - 1)

        if len(lows) < 2:
            return None

        # Check the most recent pair of lows
        prev_low_idx = lows[-2]
        curr_low_idx = lows[-1]

        if curr_low_idx - prev_low_idx < min_bars:
            return None

        # Bullish divergence: price makes lower low, RSI makes higher low
        if (recent_prices[curr_low_idx] < recent_prices[prev_low_idx]
                and recent_rsis[curr_low_idx] > recent_rsis[prev_low_idx]):
            price_diff = abs(recent_prices[prev_low_idx] - recent_prices[curr_low_idx])
            rsi_diff = abs(recent_rsis[curr_low_idx] - recent_rsis[prev_low_idx])
            avg_price = (abs(recent_prices[prev_low_idx]) + abs(recent_prices[curr_low_idx])) / 2
            strength = min(1.0, (price_diff / avg_price * 10 + rsi_diff / 30))
            return max(0.1, strength)

        return None

    def _detect_bearish_divergence(
        self,
        prices: list[float],
        rsis: list[float],
        lookback: int,
        min_bars: int,
    ) -> float | None:
        """Detect bearish divergence: price higher high, RSI lower high.

        Returns divergence strength (0-1) or None.
        """
        recent_prices = prices[-lookback:]
        recent_rsis = rsis[-lookback:]

        # Find local highs
        highs = []
        for i in range(2, len(recent_prices) - 1):
            if (recent_prices[i] > recent_prices[i - 1]
                    and recent_prices[i] > recent_prices[i - 2]
                    and recent_prices[i] >= recent_prices[i + 1]):
                highs.append(i)

        # Current bar as potential high
        if len(recent_prices) >= 3 and recent_prices[-1] > recent_prices[-2]:
            highs.append(len(recent_prices) - 1)

        if len(highs) < 2:
            return None

        prev_high_idx = highs[-2]
        curr_high_idx = highs[-1]

        if curr_high_idx - prev_high_idx < min_bars:
            return None

        # Bearish divergence: price makes higher high, RSI makes lower high
        if (recent_prices[curr_high_idx] > recent_prices[prev_high_idx]
                and recent_rsis[curr_high_idx] < recent_rsis[prev_high_idx]):
            price_diff = abs(recent_prices[curr_high_idx] - recent_prices[prev_high_idx])
            rsi_diff = abs(recent_rsis[prev_high_idx] - recent_rsis[curr_high_idx])
            avg_price = (abs(recent_prices[prev_high_idx]) + abs(recent_prices[curr_high_idx])) / 2
            strength = min(1.0, (price_diff / avg_price * 10 + rsi_diff / 30))
            return max(0.1, strength)

        return None
