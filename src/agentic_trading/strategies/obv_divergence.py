"""OBV Divergence Strategy (CMT Strategy 14).

On-Balance Volume divergence detection for accumulation/distribution:
  Bullish: Price makes lower low, OBV makes higher low (accumulation)
  Bearish: Price makes higher high, OBV makes lower high (distribution)

Entry requires price breakout confirmation after divergence.

Signal logic:
  LONG:  Bullish OBV divergence + price breaks above recent swing high
         + volume > average
  SHORT: Bearish OBV divergence + price breaks below recent swing low
         + volume > average
  EXIT:  OBV reversal (crosses below/above its EMA) or counter-divergence
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


@register_strategy("obv_divergence")
class OBVDivergenceStrategy(BaseStrategy):
    """CMT Strategy 14: OBV Divergence for Accumulation/Distribution."""

    def __init__(
        self,
        strategy_id: str = "obv_divergence",
        params: dict[str, Any] | None = None,
    ):
        defaults = {
            "lookback_bars": 30,
            "min_divergence_bars": 5,
            "volume_confirmation": 1.0,  # Volume must be >= average
            "atr_multiplier": 2.0,
            "min_confidence": 0.4,
            "obv_ema_exit": True,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(strategy_id, merged)
        self._current_regime: RegimeState | None = None
        # Rolling history
        self._price_history: dict[str, list[float]] = {}
        self._obv_history: dict[str, list[float]] = {}
        self._high_history: dict[str, list[float]] = {}
        self._low_history: dict[str, list[float]] = {}
        self._max_history = 60

    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        f = features.features
        symbol = candle.symbol

        obv = f.get("obv")
        obv_ema = f.get("obv_ema_20")
        atr = f.get("atr")
        volume_ratio = f.get("volume_ratio", 1.0)

        if obv is None:
            return None

        # Update history
        self._update_history(symbol, candle.close, candle.high, candle.low, obv)

        # ---- EXIT CHECK (before lookback gate) ----
        current_pos = self._position_direction(symbol)
        if current_pos is not None:
            should_exit = False
            exit_reasons = []

            # OBV EMA crossover exit
            if self._get_param("obv_ema_exit") and obv_ema is not None:
                if current_pos == "long" and obv < obv_ema:
                    should_exit = True
                    exit_reasons.append("OBV crossed below its 20-EMA (momentum fading)")
                elif current_pos == "short" and obv > obv_ema:
                    should_exit = True
                    exit_reasons.append("OBV crossed above its 20-EMA (momentum fading)")

            if should_exit:
                self._record_exit(symbol)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction=SignalDirection.FLAT,
                    confidence=0.5,
                    rationale=" | ".join(exit_reasons),
                    features_used={"obv": obv, "obv_ema_20": obv_ema},
                    timeframe=candle.timeframe,
                    risk_constraints={},
                )
            return None

        # ---- ENTRY CHECK ----
        prices = self._price_history.get(symbol, [])
        obvs = self._obv_history.get(symbol, [])
        highs_hist = self._high_history.get(symbol, [])
        lows_hist = self._low_history.get(symbol, [])

        lookback = self._get_param("lookback_bars")
        if len(prices) < lookback:
            return None

        if self._on_cooldown(symbol, candle.timestamp):
            return None

        min_bars = self._get_param("min_divergence_bars")
        vol_gate = self._get_param("volume_confirmation")

        # Volume gate
        if volume_ratio < vol_gate:
            return None

        direction = SignalDirection.FLAT
        rationale_parts = []
        divergence_strength = 0.0

        # Check for bullish OBV divergence
        bull_div = self._detect_bullish_divergence(
            prices[-lookback:], obvs[-lookback:], min_bars
        )
        if bull_div is not None:
            # Confirm with breakout: price above recent swing high
            recent_high = max(highs_hist[-10:]) if len(highs_hist) >= 10 else max(highs_hist)
            if candle.close > recent_high * 0.998:  # Near or above recent high
                direction = SignalDirection.LONG
                divergence_strength = bull_div
                rationale_parts.append(f"Bullish OBV divergence (strength={bull_div:.2f})")
                rationale_parts.append("Accumulation detected")
                rationale_parts.append(f"Volume {volume_ratio:.1f}x average")

        # Check for bearish OBV divergence
        if direction == SignalDirection.FLAT:
            bear_div = self._detect_bearish_divergence(
                prices[-lookback:], obvs[-lookback:], min_bars
            )
            if bear_div is not None:
                # Confirm with breakdown: price below recent swing low
                recent_low = min(lows_hist[-10:]) if len(lows_hist) >= 10 else min(lows_hist)
                if candle.close < recent_low * 1.002:  # Near or below recent low
                    direction = SignalDirection.SHORT
                    divergence_strength = bear_div
                    rationale_parts.append(f"Bearish OBV divergence (strength={bear_div:.2f})")
                    rationale_parts.append("Distribution detected")
                    rationale_parts.append(f"Volume {volume_ratio:.1f}x average")

        if direction == SignalDirection.FLAT:
            return None

        # Confidence
        vol_boost = min(0.2, (volume_ratio - 1.0) * 0.1)
        confidence = min(1.0, divergence_strength * 0.6 + 0.3 + vol_boost)

        # Regime adjustment
        if self._current_regime and self._current_regime.regime.value == "trend":
            # OBV divergence at trend exhaustion is high quality
            confidence = min(1.0, confidence + 0.1)
            rationale_parts.append("Trend regime (potential reversal)")

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        # Risk constraints
        risk_constraints = {
            "sizing_method": "fixed_fractional",
            "price": candle.close,
        }
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            risk_constraints["stop_distance_atr"] = atr * atr_mult
            risk_constraints["atr"] = atr

        features_used = {
            "obv": obv,
            "close": candle.close,
            "volume_ratio": volume_ratio,
            "divergence_strength": divergence_strength,
        }
        if obv_ema is not None:
            features_used["obv_ema_20"] = obv_ema
        if atr is not None:
            features_used["atr"] = atr

        self._record_entry(symbol, direction.value)
        self._record_signal_time(symbol, candle.timestamp)

        # Compute explicit TP/SL prices
        _take_profit: Decimal | None = None
        _stop_loss: Decimal | None = None
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            sl_distance = atr * atr_mult
            if direction == SignalDirection.LONG:
                _stop_loss = Decimal(str(candle.close - sl_distance))
                _take_profit = Decimal(str(candle.close + sl_distance * 2))
            elif direction == SignalDirection.SHORT:
                _stop_loss = Decimal(str(candle.close + sl_distance))
                _take_profit = Decimal(str(candle.close - sl_distance * 2))

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 3),
            rationale=" | ".join(rationale_parts),
            features_used=features_used,
            timeframe=candle.timeframe,
            risk_constraints=risk_constraints,
            take_profit=_take_profit,
            stop_loss=_stop_loss,
        )

    def _update_history(
        self, symbol: str, close: float, high: float, low: float, obv: float
    ) -> None:
        """Maintain rolling history."""
        for store, val in [
            (self._price_history, close),
            (self._high_history, high),
            (self._low_history, low),
            (self._obv_history, obv),
        ]:
            if symbol not in store:
                store[symbol] = []
            store[symbol].append(val)
            if len(store[symbol]) > self._max_history:
                store[symbol] = store[symbol][-self._max_history:]

    def _detect_bullish_divergence(
        self,
        prices: list[float],
        obvs: list[float],
        min_bars: int,
    ) -> float | None:
        """Detect bullish OBV divergence: price lower low, OBV higher low."""
        # Find local lows in price
        lows = []
        for i in range(2, len(prices) - 1):
            if (
                prices[i] < prices[i - 1]
                and prices[i] < prices[i - 2]
                and prices[i] <= prices[i + 1]
            ):
                lows.append(i)

        if len(prices) >= 3 and prices[-1] < prices[-2]:
            lows.append(len(prices) - 1)

        if len(lows) < 2:
            return None

        prev_idx = lows[-2]
        curr_idx = lows[-1]

        if curr_idx - prev_idx < min_bars:
            return None

        # Bullish: price lower low, OBV higher low
        if prices[curr_idx] < prices[prev_idx] and obvs[curr_idx] > obvs[prev_idx]:
            price_diff = abs(prices[prev_idx] - prices[curr_idx])
            avg_price = (abs(prices[prev_idx]) + abs(prices[curr_idx])) / 2
            if avg_price > 0:
                strength = min(1.0, price_diff / avg_price * 15)
                return max(0.1, strength)

        return None

    def _detect_bearish_divergence(
        self,
        prices: list[float],
        obvs: list[float],
        min_bars: int,
    ) -> float | None:
        """Detect bearish OBV divergence: price higher high, OBV lower high."""
        # Find local highs in price
        highs = []
        for i in range(2, len(prices) - 1):
            if (
                prices[i] > prices[i - 1]
                and prices[i] > prices[i - 2]
                and prices[i] >= prices[i + 1]
            ):
                highs.append(i)

        if len(prices) >= 3 and prices[-1] > prices[-2]:
            highs.append(len(prices) - 1)

        if len(highs) < 2:
            return None

        prev_idx = highs[-2]
        curr_idx = highs[-1]

        if curr_idx - prev_idx < min_bars:
            return None

        # Bearish: price higher high, OBV lower high
        if prices[curr_idx] > prices[prev_idx] and obvs[curr_idx] < obvs[prev_idx]:
            price_diff = abs(prices[curr_idx] - prices[prev_idx])
            avg_price = (abs(prices[prev_idx]) + abs(prices[curr_idx])) / 2
            if avg_price > 0:
                strength = min(1.0, price_diff / avg_price * 15)
                return max(0.1, strength)

        return None
