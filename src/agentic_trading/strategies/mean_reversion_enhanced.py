"""Enhanced Mean-Reversion with Weekly Filter (CMT Strategy 6).

Improves on the base mean-reversion by adding:
  - Weekly trend filter (don't mean-revert against strong weekly downtrend)
  - Time-based stop (10 bars max holding period)
  - Partial position scaling based on distance from band
  - ATR-based stop loss instead of fixed

Signal logic:
  LONG:  Price < lower BB + RSI < 30 + weekly NOT in strong downtrend
         Target = middle BB. Time stop = 10 bars.
  SHORT: Price > upper BB + RSI > 70 + weekly NOT in strong uptrend
         Target = middle BB. Time stop = 10 bars.
  EXIT:  Price reaches middle BB OR 10 bars elapsed (time stop)
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


@register_strategy("mean_reversion_enhanced")
class MeanReversionEnhancedStrategy(BaseStrategy):
    """CMT Strategy 6: Enhanced Mean-Reversion with Weekly Filter + Time Stop."""

    def __init__(
        self,
        strategy_id: str = "mean_reversion_enhanced",
        params: dict[str, Any] | None = None,
    ):
        defaults = {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "time_stop_bars": 10,
            "atr_multiplier": 2.5,
            "min_confidence": 0.35,
            "require_range_regime": False,
            "weekly_trend_filter": True,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(strategy_id, merged)
        self._current_regime: RegimeState | None = None
        # Track entry bar count for time stops
        self._entry_bar: dict[str, int] = {}
        self._bar_count: dict[str, int] = {}

    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        f = features.features
        symbol = candle.symbol

        # Increment bar counter
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1
        bar_num = self._bar_count[symbol]

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

        # ---- EXIT CHECK ----
        current_pos = self._position_direction(symbol)
        if current_pos is not None:
            should_exit = False
            exit_reasons = []

            # Target exit: price reaches middle BB
            if current_pos == "long" and price >= middle_bb:
                should_exit = True
                exit_reasons.append(f"Target hit: price={price:.2f} >= middle BB={middle_bb:.2f}")
            elif current_pos == "short" and price <= middle_bb:
                should_exit = True
                exit_reasons.append(f"Target hit: price={price:.2f} <= middle BB={middle_bb:.2f}")

            # Time stop: max holding period
            entry_bar = self._entry_bar.get(symbol, 0)
            bars_held = bar_num - entry_bar
            time_stop = self._get_param("time_stop_bars")
            if bars_held >= time_stop:
                should_exit = True
                exit_reasons.append(f"Time stop: {bars_held} bars held >= {time_stop} limit")

            if should_exit:
                self._record_exit(symbol)
                self._entry_bar.pop(symbol, None)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction=SignalDirection.FLAT,
                    confidence=0.5,
                    rationale=" | ".join(exit_reasons),
                    features_used={"bb_middle": middle_bb, "rsi": rsi, "bars_held": bars_held},
                    timeframe=candle.timeframe,
                    risk_constraints={},
                )
            return None

        # ---- ENTRY CHECK ----
        if self._on_cooldown(symbol, candle.timestamp):
            return None

        # Regime filter
        if (
            self._get_param("require_range_regime")
            and self._current_regime
            and self._current_regime.regime.value == "trend"
        ):
            return None

        # Weekly trend filter: don't mean-revert against strong weekly trend
        if self._get_param("weekly_trend_filter"):
            weekly_fast = f.get("1d_ema_50") or f.get("4h_ema_50")
            weekly_slow = f.get("1d_ema_200") or f.get("4h_ema_200")
            htf_adx = f.get("1d_adx") or f.get("4h_adx")

            if (
                weekly_fast is not None
                and weekly_slow is not None
                and htf_adx is not None
                and htf_adx > 30
            ):
                # Strong weekly downtrend: don't go long
                if weekly_fast < weekly_slow and price < lower_bb:
                    return None
                # Strong weekly uptrend: don't go short
                if weekly_fast > weekly_slow and price > upper_bb:
                    return None

        direction = SignalDirection.FLAT
        rationale_parts = []
        mr_score = 0.0

        rsi_oversold = self._get_param("rsi_oversold")
        rsi_overbought = self._get_param("rsi_overbought")

        # Mean reversion long
        if price < lower_bb and rsi < rsi_oversold:
            direction = SignalDirection.LONG
            band_distance = (lower_bb - price) / bb_width
            rsi_distance = (rsi_oversold - rsi) / rsi_oversold
            mr_score = (band_distance + rsi_distance) / 2.0
            rationale_parts.append(f"Price {band_distance:.1%} below lower BB")
            rationale_parts.append(f"RSI={rsi:.1f} (oversold)")

        # Mean reversion short
        elif price > upper_bb and rsi > rsi_overbought:
            direction = SignalDirection.SHORT
            band_distance = (price - upper_bb) / bb_width
            rsi_distance = (rsi - rsi_overbought) / (100 - rsi_overbought)
            mr_score = (band_distance + rsi_distance) / 2.0
            rationale_parts.append(f"Price {band_distance:.1%} above upper BB")
            rationale_parts.append(f"RSI={rsi:.1f} (overbought)")

        else:
            return None

        # Confidence: based on MR score
        confidence = min(1.0, mr_score * 0.7 + 0.3)

        # Regime boost
        if self._current_regime and self._current_regime.regime.value == "range":
            confidence = min(1.0, confidence + 0.1)
            rationale_parts.append("Range regime confirmed")

        # Volatility regime: higher vol = wider bands = better MR opportunity
        if self._current_regime and self._current_regime.volatility.value == "high":
            confidence = min(1.0, confidence + 0.05)

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        rationale_parts.append(f"Time stop: {self._get_param('time_stop_bars')} bars")

        # Risk constraints
        risk_constraints = {
            "target_price": middle_bb,
            "price": price,
            "sizing_method": "fixed_fractional",
            "time_stop_bars": self._get_param("time_stop_bars"),
        }
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            risk_constraints["stop_distance_atr"] = atr * atr_mult
            risk_constraints["atr"] = atr

        features_used = {
            "bb_upper": upper_bb,
            "bb_lower": lower_bb,
            "bb_middle": middle_bb,
            "rsi": rsi,
            "mr_score": mr_score,
            "close": price,
        }
        if atr is not None:
            features_used["atr"] = atr

        # Record position, entry bar, and cooldown
        self._record_entry(symbol, direction.value)
        self._entry_bar[symbol] = bar_num
        self._record_signal_time(symbol, candle.timestamp)

        # Compute explicit TP/SL prices
        _take_profit: Decimal | None = None
        _stop_loss: Decimal | None = None
        # TP = middle BB (the mean-reversion target)
        _take_profit = Decimal(str(middle_bb))
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            sl_distance = atr * atr_mult
            if direction == SignalDirection.LONG:
                _stop_loss = Decimal(str(price - sl_distance))
            else:
                _stop_loss = Decimal(str(price + sl_distance))

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
