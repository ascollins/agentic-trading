"""Mean reversion strategy.

Bollinger Band + RSI with range-regime filtering.
Only trades when regime detector says market is ranging.

Signal logic:
  LONG:  price < lower_bb AND rsi < oversold AND regime=range
  SHORT: price > upper_bb AND rsi > overbought AND regime=range
  Confidence scaled by distance from band and RSI extremity.
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


@register_strategy("mean_reversion")
class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band + RSI mean reversion, range-regime filtered."""

    def __init__(self, strategy_id: str = "mean_reversion", params: dict[str, Any] | None = None):
        defaults = {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "mean_reversion_score_threshold": 0.6,
            "require_range_regime": True,
            "min_confidence": 0.3,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(strategy_id, merged)
        self._current_regime: RegimeState | None = None

    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        f = features.features

        # Required features
        upper_bb = f.get("bb_upper")
        lower_bb = f.get("bb_lower")
        middle_bb = f.get("bb_middle")
        rsi = f.get("rsi")
        atr = f.get("atr")

        if any(v is None for v in (upper_bb, lower_bb, middle_bb, rsi)):
            return None

        price = candle.close
        bb_width = upper_bb - lower_bb
        if bb_width <= 0:
            return None

        # ---- EXIT CHECK: close when price reverts to middle band ----
        current_pos = self._position_direction(candle.symbol)
        if current_pos is not None:
            should_exit = False
            exit_reasons = []

            if current_pos == "long" and price >= middle_bb:
                should_exit = True
                exit_reasons.append(f"Price reached middle BB ({middle_bb:.2f}) — target hit")
            elif current_pos == "short" and price <= middle_bb:
                should_exit = True
                exit_reasons.append(f"Price reached middle BB ({middle_bb:.2f}) — target hit")

            if should_exit:
                self._record_exit(candle.symbol)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=candle.symbol,
                    direction=SignalDirection.FLAT,
                    confidence=0.5,
                    rationale=" | ".join(exit_reasons),
                    features_used={"bb_middle": middle_bb, "rsi": rsi},
                    timeframe=candle.timeframe,
                    risk_constraints={},
                )
            # Already positioned, don't re-signal
            return None

        # ---- ENTRY CHECK ----
        if self._on_cooldown(candle.symbol, candle.timestamp):
            return None

        # Regime filter: only trade in range/unknown regime
        if self._get_param("require_range_regime") and self._current_regime:
            if self._current_regime.regime.value == "trend":
                return None

        direction = SignalDirection.FLAT
        rationale_parts = []
        mr_score = 0.0

        rsi_oversold = self._get_param("rsi_oversold")
        rsi_overbought = self._get_param("rsi_overbought")

        # Mean reversion long: price below lower band + RSI oversold
        if price < lower_bb and rsi < rsi_oversold:
            direction = SignalDirection.LONG
            band_distance = (lower_bb - price) / bb_width
            rsi_distance = (rsi_oversold - rsi) / rsi_oversold
            mr_score = (band_distance + rsi_distance) / 2.0
            rationale_parts.append(f"Price below lower BB by {band_distance:.2%}")
            rationale_parts.append(f"RSI={rsi:.1f} (oversold)")

        # Mean reversion short: price above upper band + RSI overbought
        elif price > upper_bb and rsi > rsi_overbought:
            direction = SignalDirection.SHORT
            band_distance = (price - upper_bb) / bb_width
            rsi_distance = (rsi - rsi_overbought) / (100 - rsi_overbought)
            mr_score = (band_distance + rsi_distance) / 2.0
            rationale_parts.append(f"Price above upper BB by {band_distance:.2%}")
            rationale_parts.append(f"RSI={rsi:.1f} (overbought)")

        else:
            return None

        # Check mean reversion score threshold
        threshold = self._get_param("mean_reversion_score_threshold")
        if mr_score < threshold:
            return None

        # Confidence: based on MR score, capped at 1.0
        confidence = min(1.0, mr_score * 0.8 + 0.2)

        # Regime boost: higher confidence in confirmed range regime
        if self._current_regime and self._current_regime.regime.value == "range":
            confidence = min(1.0, confidence + 0.1)
            rationale_parts.append("Range regime confirmed")

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        # Risk: target is middle band, stop is ATR-based
        risk_constraints = {
            "target_price": middle_bb,
            "price": candle.close,
            "sizing_method": "fixed_fractional",
        }
        if atr is not None:
            risk_constraints["stop_distance_atr"] = atr * 2.0
            risk_constraints["atr"] = atr

        # Explicit TP/SL prices for execution layer
        _take_profit = Decimal(str(middle_bb))
        sl_distance = atr * 2.0 if atr is not None else bb_width / 2.0
        if direction == SignalDirection.LONG:
            _stop_loss = Decimal(str(candle.close - sl_distance))
        else:
            _stop_loss = Decimal(str(candle.close + sl_distance))

        features_used = {
            "bb_upper": upper_bb,
            "bb_lower": lower_bb,
            "bb_middle": middle_bb,
            "rsi": rsi,
            "mr_score": mr_score,
            "close": candle.close,
        }
        if atr is not None:
            features_used["atr"] = atr

        # Record position and cooldown
        self._record_entry(candle.symbol, direction.value)
        self._record_signal_time(candle.symbol, candle.timestamp)

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
