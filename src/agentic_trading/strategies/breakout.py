"""Breakout / volatility expansion strategy.

Donchian channel breakout with volume confirmation, liquidity-aware sizing.

Signal logic:
  LONG:  price breaks above Donchian upper AND volume > avg * mult AND ATR expanding
  SHORT: price breaks below Donchian lower AND volume > avg * mult AND ATR expanding
  Confidence scaled by volume spike and ATR expansion rate.
  Liquidity awareness: reduce size in thin orderbook conditions.
"""

from __future__ import annotations

from typing import Any

from agentic_trading.core.enums import SignalDirection
from agentic_trading.core.events import FeatureVector, RegimeState, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle

from .base import BaseStrategy
from .registry import register_strategy


@register_strategy("breakout")
class BreakoutStrategy(BaseStrategy):
    """Donchian channel breakout with volume confirmation."""

    def __init__(self, strategy_id: str = "breakout", params: dict[str, Any] | None = None):
        defaults = {
            "donchian_period": 20,
            "volume_confirmation_multiplier": 1.5,
            "atr_period": 14,
            "breakout_atr_threshold": 1.0,
            "min_liquidity_score": 0.5,
            "min_confidence": 0.3,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(strategy_id, merged)
        self._current_regime: RegimeState | None = None
        self._prev_atr: float | None = None

    def on_candle(
        self,
        ctx: TradingContext,
        candle: Candle,
        features: FeatureVector,
    ) -> Signal | None:
        f = features.features

        # Required features
        donchian_upper = f.get("donchian_upper")
        donchian_lower = f.get("donchian_lower")
        atr = f.get("atr")
        volume_ratio = f.get("volume_ratio", 1.0)

        if any(v is None for v in (donchian_upper, donchian_lower, atr)):
            return None

        price = candle.close

        # ATR expansion check: current ATR should be increasing
        atr_expanding = True
        if self._prev_atr is not None:
            atr_change = (atr - self._prev_atr) / self._prev_atr if self._prev_atr > 0 else 0
            atr_threshold = self._get_param("breakout_atr_threshold")
            # ATR should be at least stable or expanding
            atr_expanding = atr_change > -0.1  # Allow small contraction
        self._prev_atr = atr

        # Volume confirmation
        vol_mult = self._get_param("volume_confirmation_multiplier")
        volume_confirmed = volume_ratio >= vol_mult

        direction = SignalDirection.FLAT
        rationale_parts = []

        # Breakout long: price above Donchian upper
        if price > donchian_upper:
            direction = SignalDirection.LONG
            rationale_parts.append(
                f"Price {price:.2f} > Donchian upper {donchian_upper:.2f}"
            )

        # Breakout short: price below Donchian lower
        elif price < donchian_lower:
            direction = SignalDirection.SHORT
            rationale_parts.append(
                f"Price {price:.2f} < Donchian lower {donchian_lower:.2f}"
            )

        else:
            return None  # No breakout

        # Must have volume confirmation
        if not volume_confirmed:
            return None
        rationale_parts.append(f"Volume ratio={volume_ratio:.2f}x (confirmed)")

        # ATR expansion adds conviction
        if atr_expanding:
            rationale_parts.append("ATR expanding")
        else:
            rationale_parts.append("ATR contracting (reduced confidence)")

        # Liquidity awareness
        liquidity_score = f.get("liquidity_score", 1.0)
        min_liq = self._get_param("min_liquidity_score")
        if liquidity_score < min_liq:
            return None  # Skip in illiquid conditions

        # Confidence
        vol_confidence = min(1.0, (volume_ratio - 1.0) / 2.0)  # 0-1 based on volume spike
        atr_confidence = 0.8 if atr_expanding else 0.4
        liq_factor = min(1.0, liquidity_score / 0.8)
        confidence = vol_confidence * 0.4 + atr_confidence * 0.4 + liq_factor * 0.2

        # Regime boost: breakouts work better in trend regimes
        if self._current_regime and self._current_regime.regime.value == "trend":
            confidence = min(1.0, confidence + 0.1)
            rationale_parts.append("Trend regime (boosted)")

        min_conf = self._get_param("min_confidence")
        if confidence < min_conf:
            return None

        # Risk constraints
        channel_width = donchian_upper - donchian_lower
        risk_constraints = {
            "stop_distance": channel_width * 0.5,  # Stop at mid-channel
            "atr": atr,
            "sizing_method": "liquidity_adjusted",
            "liquidity_score": liquidity_score,
        }

        features_used = {
            "donchian_upper": donchian_upper,
            "donchian_lower": donchian_lower,
            "atr": atr,
            "volume_ratio": volume_ratio,
            "liquidity_score": liquidity_score,
        }

        return Signal(
            strategy_id=self.strategy_id,
            symbol=candle.symbol,
            direction=direction,
            confidence=round(confidence, 3),
            rationale=" | ".join(rationale_parts),
            features_used=features_used,
            timeframe=candle.timeframe,
            risk_constraints=risk_constraints,
        )
