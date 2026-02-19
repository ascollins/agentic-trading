"""Multi-Timeframe Moving Average Trend System (CMT Strategy 1).

Three-timeframe EMA alignment with pullback entry:
  Weekly 20 EMA (trend direction) + Daily 50/200 EMA golden cross
  + 4H 21 EMA pullback with RSI 40-50 zone entry.

Signal logic:
  LONG:  1d_ema_50 > 1d_ema_200 AND ADX > 20 AND price pulls back to 4h_ema_21
         AND RSI in 40-50 zone (pullback confirmation)
  SHORT: 1d_ema_50 < 1d_ema_200 AND ADX > 20 AND price rallies to 4h_ema_21
         AND RSI in 50-60 zone (rally exhaustion)
  EXIT:  ADX < 15 (trend loss) OR counter-trend EMA crossover
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


@register_strategy("multi_tf_ma")
class MultiTFMAStrategy(BaseStrategy):
    """CMT Strategy 1: Multi-Timeframe Moving Average Trend System."""

    def __init__(self, strategy_id: str = "multi_tf_ma", params: dict[str, Any] | None = None):
        defaults = {
            "fast_ema": 50,
            "slow_ema": 200,
            "pullback_ema": 21,
            "adx_entry_threshold": 20,
            "adx_exit_threshold": 15,
            "rsi_pullback_low": 40,
            "rsi_pullback_high": 50,
            "rsi_rally_low": 50,
            "rsi_rally_high": 60,
            "atr_multiplier": 2.0,
            "min_confidence": 0.4,
            "pullback_tolerance_pct": 0.005,  # 0.5% tolerance around pullback EMA
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
        fast_ema = f.get(f"ema_{self._get_param('fast_ema')}")
        slow_ema = f.get(f"ema_{self._get_param('slow_ema')}")
        pullback_ema = f.get(f"ema_{self._get_param('pullback_ema')}")
        adx = f.get("adx")
        rsi = f.get("rsi")
        atr = f.get("atr")

        if any(v is None for v in (fast_ema, slow_ema, adx, rsi)):
            return None

        price = candle.close
        adx_entry = self._get_param("adx_entry_threshold")
        adx_exit = self._get_param("adx_exit_threshold")

        # ---- EXIT CHECK ----
        current_pos = self._position_direction(candle.symbol)
        if current_pos is not None:
            should_exit = False
            exit_reasons = []

            # Exit if trend strength dies
            if adx < adx_exit:
                should_exit = True
                exit_reasons.append(f"ADX={adx:.1f} below exit threshold {adx_exit}")

            # Exit on EMA crossover reversal
            if current_pos == "long" and fast_ema < slow_ema:
                should_exit = True
                exit_reasons.append("Death cross: EMA50 < EMA200")
            elif current_pos == "short" and fast_ema > slow_ema:
                should_exit = True
                exit_reasons.append("Golden cross: EMA50 > EMA200")

            if should_exit:
                self._record_exit(candle.symbol)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=candle.symbol,
                    direction=SignalDirection.FLAT,
                    confidence=0.5,
                    rationale=" | ".join(exit_reasons),
                    features_used={"adx": adx, "ema_50": fast_ema, "ema_200": slow_ema},
                    timeframe=candle.timeframe,
                    risk_constraints={},
                )
            return None

        # ---- ENTRY CHECK ----
        if self._on_cooldown(candle.symbol, candle.timestamp):
            return None

        # Step 1: Confirm trend direction via daily EMA alignment
        if adx < adx_entry:
            return None

        # Determine trend direction from EMA cross
        is_bullish = fast_ema > slow_ema
        is_bearish = fast_ema < slow_ema

        if not is_bullish and not is_bearish:
            return None

        # Step 2: Check higher TF alignment
        htf_aligned = self._check_htf_alignment(f, "long" if is_bullish else "short")

        # Step 3: Check pullback to 21 EMA (price near the pullback EMA)
        if pullback_ema is not None:
            tolerance = self._get_param("pullback_tolerance_pct")
            distance_pct = abs(price - pullback_ema) / pullback_ema if pullback_ema > 0 else 1.0

            if is_bullish:
                # For long: price should be near or just above the 21 EMA (pullback)
                near_pullback = distance_pct <= tolerance or (
                    price >= pullback_ema * (1 - tolerance)
                    and price <= pullback_ema * (1 + tolerance * 3)
                )
            else:
                # For short: price should be near or just below the 21 EMA (rally)
                near_pullback = distance_pct <= tolerance or (
                    price <= pullback_ema * (1 + tolerance)
                    and price >= pullback_ema * (1 - tolerance * 3)
                )

            if not near_pullback:
                return None

        # Step 4: RSI zone confirmation
        rsi_low = self._get_param("rsi_pullback_low")
        rsi_high = self._get_param("rsi_pullback_high")
        rally_low = self._get_param("rsi_rally_low")
        rally_high = self._get_param("rsi_rally_high")

        direction = SignalDirection.FLAT
        rationale_parts = []

        if is_bullish and rsi_low <= rsi <= rsi_high:
            direction = SignalDirection.LONG
            rationale_parts.append(
                f"Golden cross (EMA{self._get_param('fast_ema')}"
                f" > EMA{self._get_param('slow_ema')})"
            )
            rationale_parts.append(f"RSI={rsi:.1f} in pullback zone [{rsi_low}-{rsi_high}]")
            if pullback_ema is not None:
                rationale_parts.append(f"Price near EMA{self._get_param('pullback_ema')} pullback")
        elif is_bearish and rally_low <= rsi <= rally_high:
            direction = SignalDirection.SHORT
            rationale_parts.append(
                f"Death cross (EMA{self._get_param('fast_ema')}"
                f" < EMA{self._get_param('slow_ema')})"
            )
            rationale_parts.append(f"RSI={rsi:.1f} in rally zone [{rally_low}-{rally_high}]")
            if pullback_ema is not None:
                rationale_parts.append(f"Price near EMA{self._get_param('pullback_ema')} rally")
        else:
            return None

        # Confidence calculation
        adx_confidence = min((adx - adx_entry) / 40.0, 1.0)
        htf_boost = 0.2 if htf_aligned else -0.1
        confidence = max(0.0, min(1.0, adx_confidence * 0.6 + 0.3 + htf_boost))

        if htf_aligned:
            rationale_parts.append("HTF alignment confirmed")

        # Regime adjustment
        if self._current_regime and self._current_regime.regime.value == "range":
            confidence *= 0.5
            rationale_parts.append("Range regime (halved confidence)")

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        rationale_parts.append(f"ADX={adx:.1f}")

        # Risk constraints
        risk_constraints = {
            "sizing_method": "volatility_adjusted",
            "price": price,
        }
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            risk_constraints["stop_distance_atr"] = atr * atr_mult
            risk_constraints["atr"] = atr

        features_used = {
            f"ema_{self._get_param('fast_ema')}": fast_ema,
            f"ema_{self._get_param('slow_ema')}": slow_ema,
            "adx": adx,
            "rsi": rsi,
            "close": price,
        }
        if pullback_ema is not None:
            features_used[f"ema_{self._get_param('pullback_ema')}"] = pullback_ema
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
                _stop_loss = Decimal(str(price - sl_distance))
                _take_profit = Decimal(str(price + sl_distance * 2))
            elif direction == SignalDirection.SHORT:
                _stop_loss = Decimal(str(price + sl_distance))
                _take_profit = Decimal(str(price - sl_distance * 2))

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

    def _check_htf_alignment(self, features: dict[str, float], direction: str) -> bool:
        """Check if higher timeframe EMAs agree with trade direction."""
        for prefix in ("1h_", "4h_", "1d_"):
            htf_fast = features.get(f"{prefix}ema_{self._get_param('fast_ema')}")
            htf_slow = features.get(f"{prefix}ema_{self._get_param('slow_ema')}")
            if htf_fast is not None and htf_slow is not None:
                if direction == "long" and htf_fast <= htf_slow:
                    return False
                if direction == "short" and htf_fast >= htf_slow:
                    return False
        return True
