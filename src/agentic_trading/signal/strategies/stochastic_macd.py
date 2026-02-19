"""Stochastic + MACD Momentum Confluence (CMT Strategy 4).

Dual-oscillator momentum confirmation:
  MACD histogram cross + Stochastic %K/%D cross within a 3-bar window,
  gated by volume confirmation (1.2x average).

Signal logic:
  LONG:  MACD histogram crosses above zero + Stochastic %K crosses above %D
         (both below 20 zone) + volume > 1.2x average
  SHORT: MACD histogram crosses below zero + Stochastic %K crosses below %D
         (both above 80 zone) + volume > 1.2x average
  EXIT:  Stochastic overbought/oversold reversal or MACD histogram reversal
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


@register_strategy("stochastic_macd")
class StochasticMACDStrategy(BaseStrategy):
    """CMT Strategy 4: Stochastic + MACD Momentum Confluence."""

    def __init__(
        self, strategy_id: str = "stochastic_macd", params: dict[str, Any] | None = None
    ):
        defaults = {
            "stoch_oversold": 20,
            "stoch_overbought": 80,
            "volume_gate": 1.2,
            "confluence_window": 3,  # Bars within which both signals must fire
            "atr_multiplier": 1.5,
            "min_confidence": 0.4,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(strategy_id, merged)
        self._current_regime: RegimeState | None = None
        # Track pending signals for confluence window
        self._macd_cross_bar: dict[str, tuple[str, int]] = {}  # symbol -> (direction, bar_count)
        self._stoch_cross_bar: dict[str, tuple[str, int]] = {}
        self._bar_count: dict[str, int] = {}

    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        f = features.features

        # Required features
        macd_hist = f.get("macd_histogram")
        stoch_k = f.get("stoch_k")
        stoch_d = f.get("stoch_d")
        stoch_k_prev = f.get("stoch_k_prev")
        stoch_d_prev = f.get("stoch_d_prev")
        macd_prev = f.get("macd_prev")
        macd_signal_prev = f.get("macd_signal_prev")
        macd = f.get("macd")
        macd_signal = f.get("macd_signal")
        volume_ratio = f.get("volume_ratio", 1.0)
        atr = f.get("atr")

        if any(v is None for v in (macd_hist, stoch_k, stoch_d, macd, macd_signal)):
            return None

        symbol = candle.symbol

        # Increment bar counter
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1
        bar_num = self._bar_count[symbol]

        # ---- EXIT CHECK ----
        current_pos = self._position_direction(symbol)
        if current_pos is not None:
            should_exit = False
            exit_reasons = []

            stoch_ob = self._get_param("stoch_overbought")
            stoch_os = self._get_param("stoch_oversold")

            # Exit on stochastic reversal
            if (
                current_pos == "long"
                and stoch_k > stoch_ob
                and stoch_k_prev is not None
                and stoch_d_prev is not None
                and stoch_k < stoch_d
                and stoch_k_prev >= stoch_d_prev
            ):
                should_exit = True
                exit_reasons.append(
                    f"Stochastic bearish cross in overbought zone "
                    f"(%K={stoch_k:.1f})"
                )
            elif (
                current_pos == "short"
                and stoch_k < stoch_os
                and stoch_k_prev is not None
                and stoch_d_prev is not None
                and stoch_k > stoch_d
                and stoch_k_prev <= stoch_d_prev
            ):
                should_exit = True
                exit_reasons.append(
                    f"Stochastic bullish cross in oversold zone "
                    f"(%K={stoch_k:.1f})"
                )

            # Exit on MACD histogram reversal
            if current_pos == "long" and macd_hist < 0:
                should_exit = True
                exit_reasons.append("MACD histogram turned negative")
            elif current_pos == "short" and macd_hist > 0:
                should_exit = True
                exit_reasons.append("MACD histogram turned positive")

            if should_exit:
                self._record_exit(symbol)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    direction=SignalDirection.FLAT,
                    confidence=0.5,
                    rationale=" | ".join(exit_reasons),
                    features_used={
                        "stoch_k": stoch_k,
                        "stoch_d": stoch_d,
                        "macd_histogram": macd_hist,
                    },
                    timeframe=candle.timeframe,
                    risk_constraints={},
                )
            return None

        # ---- ENTRY CHECK ----
        if self._on_cooldown(symbol, candle.timestamp):
            return None

        window = self._get_param("confluence_window")
        stoch_os = self._get_param("stoch_oversold")
        stoch_ob = self._get_param("stoch_overbought")

        # Detect MACD crossover
        if macd_prev is not None and macd_signal_prev is not None:
            # Bullish MACD cross: MACD crosses above signal
            if macd > macd_signal and macd_prev <= macd_signal_prev:
                self._macd_cross_bar[symbol] = ("long", bar_num)
            # Bearish MACD cross: MACD crosses below signal
            elif macd < macd_signal and macd_prev >= macd_signal_prev:
                self._macd_cross_bar[symbol] = ("short", bar_num)

        # Detect Stochastic crossover
        if stoch_k_prev is not None and stoch_d_prev is not None:
            # Bullish stoch cross in oversold zone
            if stoch_k > stoch_d and stoch_k_prev <= stoch_d_prev and stoch_k < stoch_os + 10:
                self._stoch_cross_bar[symbol] = ("long", bar_num)
            # Bearish stoch cross in overbought zone
            elif stoch_k < stoch_d and stoch_k_prev >= stoch_d_prev and stoch_k > stoch_ob - 10:
                self._stoch_cross_bar[symbol] = ("short", bar_num)

        # Check for confluence: both signals within the window
        macd_cross = self._macd_cross_bar.get(symbol)
        stoch_cross = self._stoch_cross_bar.get(symbol)

        if macd_cross is None or stoch_cross is None:
            return None

        macd_dir, macd_bar = macd_cross
        stoch_dir, stoch_bar = stoch_cross

        # Must be same direction and within window
        if macd_dir != stoch_dir:
            return None
        if abs(macd_bar - stoch_bar) > window:
            return None

        # Volume gate
        vol_gate = self._get_param("volume_gate")
        if volume_ratio < vol_gate:
            return None

        direction = SignalDirection.LONG if macd_dir == "long" else SignalDirection.SHORT
        rationale_parts = []

        if direction == SignalDirection.LONG:
            rationale_parts.append(
                f"MACD bullish cross + Stochastic bullish cross"
                f" (%K={stoch_k:.1f})"
            )
            rationale_parts.append(
                f"Volume {volume_ratio:.1f}x avg (gate={vol_gate}x)"
            )
        else:
            rationale_parts.append(
                f"MACD bearish cross + Stochastic bearish cross"
                f" (%K={stoch_k:.1f})"
            )
            rationale_parts.append(
                f"Volume {volume_ratio:.1f}x avg (gate={vol_gate}x)"
            )

        # Confidence: based on volume strength and zone quality
        vol_conf = min(1.0, (volume_ratio - 1.0) / 2.0)
        zone_quality = 0.0
        if direction == SignalDirection.LONG:
            zone_quality = (
                max(0.0, (stoch_os - stoch_k) / stoch_os)
                if stoch_k < stoch_os
                else 0.3
            )
        else:
            zone_quality = (
                max(0.0, (stoch_k - stoch_ob) / (100 - stoch_ob))
                if stoch_k > stoch_ob
                else 0.3
            )

        confidence = min(1.0, vol_conf * 0.3 + zone_quality * 0.3 + 0.4)

        # Regime bonus
        if self._current_regime and self._current_regime.regime.value == "trend":
            confidence = min(1.0, confidence + 0.1)
            rationale_parts.append("Trend regime (momentum aligned)")

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        # Clear the cross trackers after generating a signal
        self._macd_cross_bar.pop(symbol, None)
        self._stoch_cross_bar.pop(symbol, None)

        # Risk constraints
        risk_constraints = {
            "sizing_method": "volatility_adjusted",
            "price": candle.close,
        }
        if atr is not None:
            atr_mult = self._get_param("atr_multiplier")
            risk_constraints["stop_distance_atr"] = atr * atr_mult
            risk_constraints["atr"] = atr

        features_used = {
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": macd_hist,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "volume_ratio": volume_ratio,
            "close": candle.close,
        }

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
