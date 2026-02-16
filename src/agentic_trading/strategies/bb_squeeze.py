"""Bollinger Band Squeeze Breakout (CMT Strategy 5).

Detects low-volatility compression (BB inside Keltner Channel)
followed by directional breakout.

Squeeze detection: BBW at 120-period low (BBW percentile < 0.10)
Direction: Determined by breakout above/below BB with ADX and RSI filters.
Trailing stop: Keltner Channel used as adaptive trailing stop.

Signal logic:
  LONG:  BB squeeze detected + price breaks above upper BB
         + ADX > 20 + RSI > 50
  SHORT: BB squeeze detected + price breaks below lower BB
         + ADX > 20 + RSI < 50
  EXIT:  Price crosses back inside Keltner Channel (trailing stop)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from agentic_trading.core.enums import SignalDirection
from agentic_trading.core.events import FeatureVector, RegimeState, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle

from .base import BaseStrategy
from .registry import register_strategy


@register_strategy("bb_squeeze")
class BBSqueezeStrategy(BaseStrategy):
    """CMT Strategy 5: Bollinger Band Squeeze Breakout."""

    def __init__(self, strategy_id: str = "bb_squeeze", params: dict[str, Any] | None = None):
        defaults = {
            "squeeze_percentile": 0.10,  # BBW must be in lowest 10% of 120 bars
            "adx_threshold": 20,
            "rsi_neutral": 50,
            "atr_multiplier": 2.0,
            "min_confidence": 0.4,
            "keltner_trailing": True,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(strategy_id, merged)
        self._current_regime: RegimeState | None = None
        # Track whether we were in a squeeze on the previous bar
        self._in_squeeze: dict[str, bool] = {}

    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        f = features.features

        # Required features
        bb_upper = f.get("bb_upper")
        bb_lower = f.get("bb_lower")
        bb_middle = f.get("bb_middle")
        bbw = f.get("bbw")
        bbw_percentile = f.get("bbw_percentile")
        keltner_upper = f.get("keltner_upper")
        keltner_lower = f.get("keltner_lower")
        adx = f.get("adx")
        rsi = f.get("rsi")
        atr = f.get("atr")

        if any(v is None for v in (bb_upper, bb_lower, bbw)):
            return None

        price = candle.close
        symbol = candle.symbol
        squeeze_pct = self._get_param("squeeze_percentile")

        # Determine if currently in squeeze
        is_squeeze = False
        if bbw_percentile is not None:
            is_squeeze = bbw_percentile <= squeeze_pct
        elif keltner_upper is not None and keltner_lower is not None:
            # Alternative: BB inside Keltner means squeeze
            is_squeeze = bb_upper < keltner_upper and bb_lower > keltner_lower

        was_in_squeeze = self._in_squeeze.get(symbol, False)
        self._in_squeeze[symbol] = is_squeeze

        # ---- EXIT CHECK ----
        current_pos = self._position_direction(symbol)
        if current_pos is not None:
            should_exit = False
            exit_reasons = []

            # Keltner trailing stop
            if (
                self._get_param("keltner_trailing")
                and keltner_upper is not None
                and keltner_lower is not None
            ):
                if current_pos == "long" and price < keltner_lower:
                    should_exit = True
                    exit_reasons.append(f"Price below Keltner lower ({keltner_lower:.2f})")
                elif current_pos == "short" and price > keltner_upper:
                    should_exit = True
                    exit_reasons.append(f"Price above Keltner upper ({keltner_upper:.2f})")
            elif bb_middle is not None:
                # Fallback: exit at middle BB
                if current_pos == "long" and price < bb_middle:
                    should_exit = True
                    exit_reasons.append(f"Price below BB middle ({bb_middle:.2f})")
                elif current_pos == "short" and price > bb_middle:
                    should_exit = True
                    exit_reasons.append(f"Price above BB middle ({bb_middle:.2f})")

            if should_exit:
                self._record_exit(symbol)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction=SignalDirection.FLAT,
                    confidence=0.5,
                    rationale=" | ".join(exit_reasons),
                    features_used={"close": price, "bbw": bbw},
                    timeframe=candle.timeframe,
                    risk_constraints={},
                )
            return None

        # ---- ENTRY CHECK ----
        if self._on_cooldown(symbol, candle.timestamp):
            return None

        # Must have been in squeeze and now breaking out
        if not was_in_squeeze:
            return None

        # Squeeze is releasing — check breakout direction
        direction = SignalDirection.FLAT
        rationale_parts = [f"BB Squeeze breakout (BBW percentile was <{squeeze_pct:.0%})"]

        adx_threshold = self._get_param("adx_threshold")
        rsi_neutral = self._get_param("rsi_neutral")

        if price > bb_upper:
            # Bullish breakout
            if adx is not None and adx < adx_threshold:
                return None
            if rsi is not None and rsi < rsi_neutral:
                return None  # RSI should confirm bullish momentum
            direction = SignalDirection.LONG
            rationale_parts.append(f"Breakout above upper BB ({bb_upper:.2f})")
        elif price < bb_lower:
            # Bearish breakout
            if adx is not None and adx < adx_threshold:
                return None
            if rsi is not None and rsi > rsi_neutral:
                return None  # RSI should confirm bearish momentum
            direction = SignalDirection.SHORT
            rationale_parts.append(f"Breakout below lower BB ({bb_lower:.2f})")
        else:
            return None  # No breakout yet

        if adx is not None:
            rationale_parts.append(f"ADX={adx:.1f}")
        if rsi is not None:
            rationale_parts.append(f"RSI={rsi:.1f}")

        # Confidence: based on squeeze tightness and ADX confirmation
        squeeze_tightness = (
            max(0.0, 1.0 - (bbw_percentile or 0.1))
            if bbw_percentile is not None
            else 0.7
        )
        adx_conf = min(1.0, (adx - adx_threshold) / 30.0) if adx is not None else 0.3
        confidence = min(1.0, squeeze_tightness * 0.5 + adx_conf * 0.3 + 0.2)

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        # Risk constraints
        risk_constraints = {
            "sizing_method": "volatility_adjusted",
            "price": price,
        }
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            risk_constraints["stop_distance_atr"] = atr * atr_mult
            risk_constraints["atr"] = atr
        if keltner_lower is not None and direction == SignalDirection.LONG:
            risk_constraints["trailing_stop"] = keltner_lower
        elif keltner_upper is not None and direction == SignalDirection.SHORT:
            risk_constraints["trailing_stop"] = keltner_upper

        features_used = {
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bbw": bbw,
            "close": price,
        }
        if bbw_percentile is not None:
            features_used["bbw_percentile"] = bbw_percentile
        if adx is not None:
            features_used["adx"] = adx
        if rsi is not None:
            features_used["rsi"] = rsi
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
                _stop_loss = Decimal(str(price - sl_distance))
                _take_profit = Decimal(str(price + sl_distance * 2))
            elif direction == SignalDirection.SHORT:
                _stop_loss = Decimal(str(price + sl_distance))
                _take_profit = Decimal(str(price - sl_distance * 2))

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
