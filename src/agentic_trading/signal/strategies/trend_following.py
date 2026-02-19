"""Trend-following strategy.

Multi-timeframe EMA crossover with ADX trend strength filter.
Position sizing: volatility-adjusted via ATR.

Signal logic:
  LONG:  fast_ema > slow_ema AND adx > threshold AND volume confirms
  SHORT: fast_ema < slow_ema AND adx > threshold AND volume confirms
  Confidence scaled by ADX strength and multi-TF alignment.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from agentic_trading.core.enums import SignalDirection, Timeframe
from agentic_trading.core.events import FeatureVector, RegimeState, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle

from .base import BaseStrategy
from .registry import register_strategy


@register_strategy("trend_following")
class TrendFollowingStrategy(BaseStrategy):
    """Multi-timeframe trend-following with EMA crossover + ADX filter."""

    def __init__(self, strategy_id: str = "trend_following", params: dict[str, Any] | None = None):
        defaults = {
            "fast_ema": 12,
            "slow_ema": 26,
            "signal_ema": 9,
            "adx_threshold": 25,
            "atr_period": 14,
            "atr_multiplier": 1.5,
            "volume_filter": True,
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
        fast_key = f"ema_{self._get_param('fast_ema')}"
        slow_key = f"ema_{self._get_param('slow_ema')}"
        adx_key = "adx"
        atr_key = "atr"

        fast_ema = f.get(fast_key)
        slow_ema = f.get(slow_key)
        adx = f.get(adx_key)
        atr = f.get(atr_key)

        if any(v is None for v in (fast_ema, slow_ema, adx, atr)):
            return None  # Not enough data yet

        adx_threshold = self._get_param("adx_threshold")

        # ---- EXIT CHECK: close position if trend dies ----
        current_pos = self._position_direction(candle.symbol)
        if current_pos is not None:
            # Exit if ADX drops below threshold (trend exhausted)
            should_exit = False
            exit_reasons = []

            if adx < adx_threshold:
                should_exit = True
                exit_reasons.append(f"ADX={adx:.1f} below threshold {adx_threshold}")

            # Exit if EMA crossover reverses
            if current_pos == "long" and fast_ema < slow_ema:
                should_exit = True
                exit_reasons.append("EMA crossover reversed (bearish)")
            elif current_pos == "short" and fast_ema > slow_ema:
                should_exit = True
                exit_reasons.append("EMA crossover reversed (bullish)")

            if should_exit:
                self._record_exit(candle.symbol)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=candle.symbol,
                    direction=SignalDirection.FLAT,
                    confidence=0.5,
                    rationale=" | ".join(exit_reasons),
                    features_used={"adx": adx, fast_key: fast_ema, slow_key: slow_ema},
                    timeframe=candle.timeframe,
                    risk_constraints={},
                )
            # Already positioned, don't re-signal
            return None

        # ---- ENTRY CHECK ----
        # Cooldown: skip if recently signalled
        if self._on_cooldown(candle.symbol, candle.timestamp):
            return None

        # Check multi-timeframe alignment (higher TF trend confirmation)
        htf_bullish = self._check_higher_tf_alignment(f, direction="long")
        htf_bearish = self._check_higher_tf_alignment(f, direction="short")

        # ADX filter: only trade in strong trends
        if adx < adx_threshold:
            return None

        # Volume filter
        if self._get_param("volume_filter"):
            vol_ratio = f.get("volume_ratio", 1.0)
            if vol_ratio < 0.8:
                return None

        # Direction
        direction = SignalDirection.FLAT
        rationale_parts = []

        if fast_ema > slow_ema:
            direction = SignalDirection.LONG
            rationale_parts.append(f"EMA{self._get_param('fast_ema')} > EMA{self._get_param('slow_ema')}")
        elif fast_ema < slow_ema:
            direction = SignalDirection.SHORT
            rationale_parts.append(f"EMA{self._get_param('fast_ema')} < EMA{self._get_param('slow_ema')}")
        else:
            return None

        # Multi-TF confirmation boost
        if direction == SignalDirection.LONG and htf_bullish:
            rationale_parts.append("HTF aligned bullish")
        elif direction == SignalDirection.SHORT and htf_bearish:
            rationale_parts.append("HTF aligned bearish")
        elif direction == SignalDirection.LONG and not htf_bullish:
            rationale_parts.append("HTF not aligned (reduced confidence)")
        elif direction == SignalDirection.SHORT and not htf_bearish:
            rationale_parts.append("HTF not aligned (reduced confidence)")

        # Confidence: based on ADX strength + multi-TF alignment
        adx_confidence = min((adx - adx_threshold) / 50.0, 1.0)
        mtf_boost = 0.2 if (
            (direction == SignalDirection.LONG and htf_bullish)
            or (direction == SignalDirection.SHORT and htf_bearish)
        ) else -0.1
        confidence = max(0.0, min(1.0, adx_confidence * 0.7 + 0.3 + mtf_boost))

        # Regime check: reduce confidence in range regime
        if self._current_regime and self._current_regime.regime.value == "range":
            confidence *= 0.5
            rationale_parts.append("Range regime (halved confidence)")

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        rationale_parts.append(f"ADX={adx:.1f}")
        rationale_parts.append(f"ATR={atr:.4f}")

        # Risk constraints: ATR-based stop distance
        atr_mult = self._get_param("atr_multiplier")
        sl_distance = atr * atr_mult
        tp_distance = sl_distance * 2.0  # 2:1 reward-to-risk
        risk_constraints = {
            "stop_distance_atr": sl_distance,
            "atr": atr,
            "price": candle.close,
            "sizing_method": "volatility_adjusted",
        }

        # Compute explicit TP/SL price levels
        if direction == SignalDirection.LONG:
            _stop_loss = Decimal(str(candle.close - sl_distance))
            _take_profit = Decimal(str(candle.close + tp_distance))
        else:
            _stop_loss = Decimal(str(candle.close + sl_distance))
            _take_profit = Decimal(str(candle.close - tp_distance))

        features_used = {
            fast_key: fast_ema,
            slow_key: slow_ema,
            "adx": adx,
            "atr": atr,
            "close": candle.close,
        }

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

    def _check_higher_tf_alignment(self, features: dict[str, float], direction: str) -> bool:
        """Check if higher timeframe EMAs agree with the signal direction."""
        for prefix in ("1h_", "4h_"):
            htf_fast = features.get(f"{prefix}ema_{self._get_param('fast_ema')}")
            htf_slow = features.get(f"{prefix}ema_{self._get_param('slow_ema')}")
            if htf_fast is not None and htf_slow is not None:
                if direction == "long" and htf_fast <= htf_slow:
                    return False
                if direction == "short" and htf_fast >= htf_slow:
                    return False
        return True
